# Engine.py
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import opsvis as opsv


def build_rc_rect_fiber_section(secTag, D, b, cover, nBars, db,
                                GJ=1e6, mat_cover=1, mat_core=2, mat_steel=3):
    """RC rectangular fiber section (bottom-left corner at 0,0)."""

    # Geometry (origin at bottom-left)
    yb, yt = 0.0, D
    zl, zr = 0.0, b
    ycb, yct = cover, D - cover
    zcl, zcr = cover, b - cover

    # Mesh: core = 5x5; cover thickness = 2
    ny_core, nz_core = 20, 20
    n_cov_t = 5
    n_cov_face = 20

    fib = [['section', 'Fiber', secTag, '-GJ', GJ]]

    # Core (confined)
    fib += [['patch', 'rect', mat_core, ny_core, nz_core, ycb, zcl, yct, zcr]]

    # Cover strips
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_face, yct, zcl, yt, zcr]]   # top
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_face, yb,  zcl, ycb, zcr]]  # bottom
    fib += [['patch', 'rect', mat_cover, n_cov_face, n_cov_t, ycb, zl,  yct, zcl]]  # left
    fib += [['patch', 'rect', mat_cover, n_cov_face, n_cov_t, ycb, zcr, yct, zr ]]  # right

    # Cover corners
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_t, yct, zl,  yt,  zcl]]  # top-left
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_t, yct, zcr, yt,  zr ]]  # top-right
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_t, yb,  zl,  ycb, zcl]]  # bottom-left
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_t, yb,  zcr, ycb, zr ]]  # bottom-right

    # Steel layers: equally spaced along width in each layer
    nL = len(nBars)
    y_top = yt - cover - db[0]/2
    y_bot = yb + cover + db[-1]/2
    yL = [0.5*D] if nL == 1 else np.linspace(y_top, y_bot, nL)

    for i in range(nL):
        Ab = np.pi * (db[i]**2) / 4.0
        zL = cover + db[i]/2.0
        zR = b - cover - db[i]/2.0
        zi, zj = (0.5*b, 0.5*b) if int(nBars[i]) == 1 else (zL, zR)
        fib += [['layer', 'straight', mat_steel, int(nBars[i]), Ab, float(yL[i]), zi, float(yL[i]), zj]]

    return fib


def run_moment_curvature(secTag, axialLoad, maxK, numIncr,
                        D, b, cover, nBars, db,
                        fc_cover, epsc0_cover, fcu_cover, epscu_cover,
                        fc_core,  epsc0_core,  fcu_core,  epscu_core,
                        fy, Es, b_kin):
    """ZeroLengthSection M-ϕ: ϕ = rotation at node 2 (dof 3), M = base reaction (dof 3)."""

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # Materials
    ops.uniaxialMaterial('Concrete02', 1, fc_cover, epsc0_cover, fcu_cover, epscu_cover)
    ops.uniaxialMaterial('Concrete02', 2, fc_core,  epsc0_core,  fcu_core,  epscu_core)
    # ops.uniaxialMaterial('Steel02',    3, fy, Es, b_kin)
    
    
    FY = fy
    FU = fy
    FC = 0.1*fy
    
    DY = Es
    
    # print(DY)
    
    DU = b_kin
    DC = DU*1.1
    
    p1eb = [FY,DY]
    n1eb = [-FY,-DY]
    
    p2eb = [FU,DU]
    n2eb = [-FU,-DU]
    
    p3eb = [FC,DC]
    n3eb = [-FC,-DC]
    
    #Variables degradación
    pinchX = 1;
    pinchY = 1;
    damage1 = 0.0;
    damage2 = 0.0;
    beta = 0.0;
    
    ops.uniaxialMaterial('Hysteretic', 3, *p1eb, *p2eb, *p3eb, *n1eb, *n2eb, *n3eb, pinchX, pinchY, damage1, damage2, beta)

    
    # Section (define directly in OpenSees)
    fib_sec = build_rc_rect_fiber_section(secTag, D, b, cover, nBars, db)
    
    for cmd in fib_sec:
        if cmd[0] == 'section':
            ops.section(*cmd[1:])   # e.g., section('Fiber', secTag, '-GJ', GJ)
        elif cmd[0] == 'patch':
            ops.patch(*cmd[1:])     # e.g., patch('rect', mat, ny, nz, yI, zI, yJ, zJ)
        elif cmd[0] == 'layer':
            ops.layer(*cmd[1:])     # e.g., layer('straight', mat, nBar, Ab, yI, zI, yJ, zJ)


    # Two coincident nodes
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)

    # Fix: base fully fixed; top free in axial + rotation (uy locked)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 0)

    # Element
    ops.element('zeroLengthSection', 1, 1, 2, secTag)

    # Analysis core
    ops.system('SparseGeneral', '-piv')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.test('NormUnbalance', 1e-5, 200)
    ops.algorithm('Newton')
    ops.analysis('Static')

    # 1) Constant axial load
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(2, axialLoad, 0.0, 0.0)
    ops.integrator('LoadControl', 0.0)
    if ops.analyze(1) != 0:
        raise RuntimeError("Axial load step did not converge.")

    # 2) Reference unit moment (solver scales it via load factor)
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(2, 0.0, 0.0, 1.0)

    dK = maxK / float(numIncr)
    ops.integrator('DisplacementControl', 2, 3, dK)

    # Record
    K = np.zeros(numIncr + 1)
    M = np.zeros(numIncr + 1)

    ops.reactions()
    K[0] = ops.nodeDisp(2, 3)
    M[0] = -ops.nodeReaction(1, 3)

    for i in range(1, numIncr + 1):
        ok = ops.analyze(1)
        if ok != 0:
            K = K[:i]
            M = M[:i]
            break
        ops.reactions()
        K[i] = ops.nodeDisp(2, 3)        # curvature = rotation (zero length)
        M[i] = -ops.nodeReaction(1, 3)   # section moment

    return K, M, fib_sec


def plot_section_and_mc(fib_sec, D, b, K, M, tick_cm=5):
    """Section plot with 5 cm ticks + M-ϕ curve."""

    # ---------------- Section plot ----------------
    opsv.plot_fiber_section(fib_sec, matcolor=['lightgrey', 'skyblue', 'black'])
    ax = plt.gca()

    step = tick_cm / 100.0  # cm -> m
    y_ticks = np.arange(0.0, D + step/2, step)
    z_ticks = np.arange(0.0, b + step/2, step)

    ax.set_yticks(y_ticks)
    ax.set_xticks(z_ticks)
    ax.set_ylabel('Depth [cm]')
    ax.set_xlabel('Width [cm]')
    ax.set_yticklabels((y_ticks / 0.01).astype(int))
    ax.set_xticklabels((z_ticks / 0.01).astype(int))
    ax.invert_xaxis()

    plt.axis('equal')
    plt.tight_layout()

    # ---------------- Moment–curvature ----------------
    plt.figure()
    MkNm = M / 1000.0  # N·m -> kN·m
    plt.plot(K, MkNm)

    plt.xlabel('Curvature, ϕ [1/m]')
    plt.ylabel('Moment, M [kN·m]')

    # Y-axis ticks every 50 kN·m
    Mmax = np.ceil(np.max(np.abs(MkNm)) / 50.0) * 50.0
    plt.yticks(np.arange(0, Mmax + 50, 50))

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
