# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:19:51 2026

@author: jan181
"""


# EngineV3.py
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import opsvis as opsv

def compute_events_from_geometry(K, D, cover_clear, db, y_layers, fy, Es, epscu_cover_pos, eps_su):
    """
    Fallback event detection using curvature and section geometry (plane sections).
    Works even when fiber querying is unreliable.
    """
    K = np.asarray(K, dtype=float)

    epsy = fy / Es

    # concrete cover crush at top extreme fiber (compression)
    eps_top = -K * (D / 2.0)
    cover_crush_idx = None
    hit = np.where(eps_top <= -abs(epscu_cover_pos))[0]
    if hit.size > 0:
        cover_crush_idx = int(hit[0])

    # tension steel: use lowest layer (closest to bottom)
    i_tension = int(np.argmin(np.array(y_layers, dtype=float)))
    # distance from neutral axis ~ D/2 to bottom bar centroid:
    # approx: ybar_from_bottom = y_layers[i]
    # bottom extreme fiber is y=0, NA approx at D/2 => lever arm = D/2 - ybar
    # but we want tension strain magnitude at bar relative to NA:
    # if NA assumed mid-depth: eps = K*(D/2 - ybar)
    # Better: assume linear strain with zero at mid-depth:
    ybar = float(y_layers[i_tension])
    eps_bar = K * (D/2.0 - ybar)

    steel_yield_idx = None
    hit = np.where(eps_bar >= epsy)[0]
    if hit.size > 0:
        steel_yield_idx = int(hit[0])

    steel_fracture_idx = None
    hit = np.where(eps_bar >= abs(eps_su))[0]
    if hit.size > 0:
        steel_fracture_idx = int(hit[0])

    return {
        "steel_yield_idx": steel_yield_idx,
        "steel_fracture_idx": steel_fracture_idx,
        "cover_crush_idx": cover_crush_idx,
        "epsy": float(epsy),
        "eps_su": float(eps_su),
        "method": "geometry_fallback"
    }


def build_rc_rect_fiber_section_v2(
    secTag, D, b, cover, nBars, db, y_layers,
    GJ=1e6, mat_cover=1, mat_core=2, mat_steel=3
):
    """
    RC rectangular fiber section (bottom-left corner at 0,0).

    Steel layers:
      - y_layers: list of layer heights measured from bottom [m]
      - nBars[i]: number of bars in layer i
      - db[i]: bar diameter for layer i [m]

    Bars are distributed along width between (cover + db/2) and (b - cover - db/2).
    If nBars[i] == 1, bar is placed at mid-width.
    """

    if not (len(nBars) == len(db) == len(y_layers)):
        raise ValueError("nBars, db, and y_layers must have the same length.")

    # Geometry (origin at bottom-left)
    yb, yt = 0.0, D
    zl, zr = 0.0, b
    ycb, yct = cover, D - cover
    zcl, zcr = cover, b - cover

    # Mesh
    ny_core, nz_core = 20, 20
    n_cov_t = 5
    n_cov_face = 20

    fib = [['section', 'Fiber', secTag, '-GJ', GJ]]

    # Core (confined)
    fib += [['patch', 'rect', mat_core, ny_core, nz_core, ycb, zcl, yct, zcr]]

    # Cover strips
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_face, yct, zcl, yt, zcr]]    # top
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_face, yb,  zcl, ycb, zcr]]   # bottom
    fib += [['patch', 'rect', mat_cover, n_cov_face, n_cov_t, ycb, zl,  yct, zcl]]   # left
    fib += [['patch', 'rect', mat_cover, n_cov_face, n_cov_t, ycb, zcr, yct, zr]]    # right

    # Cover corners
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_t, yct, zl,  yt,  zcl]]      # top-left
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_t, yct, zcr, yt,  zr]]       # top-right
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_t, yb,  zl,  ycb, zcl]]      # bottom-left
    fib += [['patch', 'rect', mat_cover, n_cov_t, n_cov_t, yb,  zcr, ycb, zr]]       # bottom-right

    # Steel layers
    for i in range(len(nBars)):
        nbi = int(nBars[i])
        if nbi <= 0:
            continue

        y = float(y_layers[i])
        di = float(db[i])

        # safety checks
        if y <= 0.0 or y >= D:
            raise ValueError(f"Layer {i+1}: y must be inside (0, D). Got y={y:.4g}, D={D:.4g}")
        if di <= 0.0:
            raise ValueError(f"Layer {i+1}: db must be > 0. Got db={di:.4g}")

        Ab = np.pi * (di ** 2) / 4.0

        zL = cover + di / 2.0
        zR = b - cover - di / 2.0
        if zR < zL:
            raise ValueError(f"Layer {i+1}: cover+db/2 exceeds width/2. Check b/cover/db.")

        # If only 1 bar in layer: place at mid-width
        if nbi == 1:
            zi, zj = 0.5 * b, 0.5 * b
        else:
            zi, zj = zL, zR

        fib += [['layer', 'straight', mat_steel, nbi, Ab, y, zi, y, zj]]

    return fib

def _pick_bar_z(cover_clear, b, db_layer, nBars_layer):
    """Return a z coordinate that matches an actual bar fiber location."""
    if int(nBars_layer) == 1:
        return 0.5 * b
    return cover_clear + db_layer / 2.0  # first bar location (leftmost)

def _fiber_strain(eleTag, secNum, y, z):
    """
    Get fiber strain at (y,z). OpenSees returns [stress, strain] for stressStrain.
    """
    try:
        ss = ops.eleResponse(eleTag, 'section', secNum, 'fiber', y, z, 'stressStrain')
        # sometimes returns a flat list, sometimes numpy-ish
        return float(ss[1])
    except Exception:
        return None
def run_moment_curvature(
    secTag, axialLoad, maxK, numIncr,
    D, b, cover_clear,
    nBars, db, y_layers,
    fc_cover_pos, epsc0_cover_pos, fcu_cover_pos, epscu_cover_pos,
    fc_core_pos,  epsc0_core_pos,  fcu_core_pos,  epscu_core_pos,
    fy, Es,
    eps_su, fracture_tension_only=True
):
    """
    cover_clear = clear cover to START of bar [m]
    Concrete strengths/strains entered positive; internally converted to OpenSees compression sign convention.
    Steel fracture: MinMax wrapper using eps_su (tension-only by default).
    """

    # ---- Convert concrete signs (Concrete02 expects negative in compression) ----
    fc_cover  = -abs(fc_cover_pos)
    fcu_cover = -abs(fcu_cover_pos)
    epsc0_cover = -abs(epsc0_cover_pos)       # <-- FIX
    epscu_cover = -abs(epscu_cover_pos)

    fc_core   = -abs(fc_core_pos)
    fcu_core  = -abs(fcu_core_pos)
    epsc0_core = -abs(epsc0_core_pos)         # <-- FIX
    epscu_core = -abs(epscu_core_pos)

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # Materials
    ops.uniaxialMaterial('Concrete02', 1, fc_cover, epsc0_cover, fcu_cover, epscu_cover)
    ops.uniaxialMaterial('Concrete02', 2, fc_core,  epsc0_core,  fcu_core,  epscu_core)

    # Steel02 base
    b_hard = 0.01
    steel_base_tag = 100
    ops.uniaxialMaterial('Steel02', steel_base_tag, fy, Es, b_hard)

    # Steel fracture wrapper (MinMax)
    steel_tag = 3
    eps_su = float(eps_su)
    if eps_su <= 0:
        raise ValueError("eps_su (steel fracture strain) must be > 0")

    if fracture_tension_only:
        ops.uniaxialMaterial('MinMax', steel_tag, steel_base_tag, '-max', eps_su)
    else:
        ops.uniaxialMaterial('MinMax', steel_tag, steel_base_tag, '-max', eps_su, '-min', -eps_su)

    # Section
    fib_sec = build_rc_rect_fiber_section_v2(
        secTag=secTag, D=D, b=b, cover=cover_clear,
        nBars=nBars, db=db, y_layers=y_layers,
        mat_cover=1, mat_core=2, mat_steel=steel_tag
    )

    for cmd in fib_sec:
        if cmd[0] == 'section':
            ops.section(*cmd[1:])
        elif cmd[0] == 'patch':
            ops.patch(*cmd[1:])
        elif cmd[0] == 'layer':
            ops.layer(*cmd[1:])

    # Nodes / BCs
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 0)

    ops.element('zeroLengthSection', 1, 1, 2, secTag)

    # Analysis setup (more robust)
    ops.system('SparseGeneral', '-piv')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.test('NormUnbalance', 1e-6, 1000)

    # Try line search; if not available, fallback to Newton
    try:
        ops.algorithm('NewtonLineSearch', 0.8)
    except Exception:
        ops.algorithm('Newton')

    # 1) Axial load
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(2, axialLoad, 0.0, 0.0)

    ops.integrator('LoadControl', 0.0)
    ops.analysis('Static')  # <-- FIX: ensure integrator is attached

    if ops.analyze(1) != 0:
        raise RuntimeError("Axial load step did not converge.")

    # 2) Unit moment
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(2, 0.0, 0.0, 1.0)

    dK = maxK / float(numIncr)
    ops.integrator('DisplacementControl', 2, 3, dK)
    ops.analysis('Static')  # <-- FIX: ensure integrator is attached

    # Record
    K = np.zeros(numIncr + 1)
    M = np.zeros(numIncr + 1)

    # Event tracking
    epsy = fy / Es
    event = {
        "steel_yield_idx": None,
        "steel_fracture_idx": None,
        "cover_crush_idx": None,
        "epsy": float(epsy),
        "eps_su": float(eps_su),
    }

    # Probe locations
    eleTag = 1
    secNum = 1
    y_top_cover = D - cover_clear / 2.0
    z_mid = 0.5 * b

    i_tension = int(np.argmin(np.array(y_layers)))
    y_tension_bar = float(y_layers[i_tension])
    z_tension_bar = _pick_bar_z(cover_clear, b, float(db[i_tension]), int(nBars[i_tension]))

    ops.reactions()
    K[0] = ops.nodeDisp(2, 3)
    M[0] = -ops.nodeReaction(1, 3)

    # Step loop with step-halving fallback
    for i in range(1, numIncr + 1):
        ok = ops.analyze(1)

        if ok != 0:
            # try smaller step sizes
            dK_try = dK
            recovered = False
            for _ in range(8):
                dK_try *= 0.5
                ops.integrator('DisplacementControl', 2, 3, dK_try)
                ops.analysis('Static')
                ok = ops.analyze(1)
                if ok == 0:
                    dK = dK_try
                    recovered = True
                    break
            if not recovered:
                K = K[:i]
                M = M[:i]
                break

        ops.reactions()
        K[i] = ops.nodeDisp(2, 3)
        M[i] = -ops.nodeReaction(1, 3)

        # event probes
        eps_cov = _fiber_strain(eleTag, secNum, y_top_cover, z_mid)
        eps_stl = _fiber_strain(eleTag, secNum, y_tension_bar, z_tension_bar)

        if eps_cov is not None and event["cover_crush_idx"] is None and eps_cov <= epscu_cover:
            event["cover_crush_idx"] = i
        if eps_stl is not None and event["steel_yield_idx"] is None and eps_stl >= epsy:
            event["steel_yield_idx"] = i
        if eps_stl is not None and event["steel_fracture_idx"] is None and eps_stl >= eps_su:
            event["steel_fracture_idx"] = i
            
    # If fiber probing never detected anything, fall back to geometry-based detection
    if (event["steel_yield_idx"] is None and
        event["steel_fracture_idx"] is None and
        event["cover_crush_idx"] is None):
        event = compute_events_from_geometry(
            K=K, D=D, cover_clear=cover_clear,
            db=db, y_layers=y_layers,
            fy=fy, Es=Es,
            epscu_cover_pos=epscu_cover_pos,
            eps_su=eps_su
        )


    return K, M, fib_sec, event


def plot_section_and_mc(fib_sec, D, b, K, M, tick_cm=5, event=None,
                        unit_system="SI", mc_tick_every=50,
                        show_fiber=True, show_mc=True):
    import matplotlib.pyplot as plt
    import numpy as np

    # Times New Roman for plots
    plt.rcParams["font.family"] = "Times New Roman"

    # ----- Fiber plot -----
    if show_fiber:
        opsv.plot_fiber_section(fib_sec, matcolor=['lightgrey', 'skyblue', 'black'])
        ax = plt.gca()

        step = tick_cm / 100.0
        y_ticks = np.arange(0.0, D + step / 2, step)
        z_ticks = np.arange(0.0, b + step / 2, step)

        ax.set_yticks(y_ticks)
        ax.set_xticks(z_ticks)
        ax.set_ylabel('Depth [cm]')
        ax.set_xlabel('Width [cm]')
        ax.set_yticklabels((y_ticks / 0.01).astype(int))
        ax.set_xticklabels((z_ticks / 0.01).astype(int))
        ax.invert_xaxis()

        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    # ----- M-phi plot -----
    if not show_mc:
        return

    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)

    # Unit conversion for plotting
    if unit_system == "Imperial":
        NM_PER_KIPIN = 4448.2216152605 * 0.0254
        K_plot = K * 0.0254            # 1/m -> 1/in
        M_plot = M / NM_PER_KIPIN      # N·m -> kip·in
        xlab = "Curvature, ϕ [1/in]"
        ylab = "Moment, M [kip·in]"
    else:
        K_plot = K
        M_plot = M / 1000.0            # N·m -> kN·m
        xlab = "Curvature, ϕ [1/m]"
        ylab = "Moment, M [kN·m]"

    plt.figure()
    plt.plot(K_plot, M_plot)

    # Start plot at (0,0)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)

    # Y-axis major tick step (in plot units)
    if mc_tick_every is not None and mc_tick_every > 0:
        y_max = float(np.max(M_plot)) if len(M_plot) else 0.0
        step = float(mc_tick_every)
        if y_max > 0:
            plt.yticks(np.arange(0.0, y_max + step, step))


    # Mark ONLY steel yield (point + label)
    if event is not None and event.get("steel_yield_idx") is not None:
        i = int(event["steel_yield_idx"])
        i = max(0, min(i, len(K_plot) - 1))
        yield_handle = None
        if event is not None and event.get("steel_yield_idx") is not None:
            i = int(event["steel_yield_idx"])
            i = max(0, min(i, len(K_plot) - 1))
        
            # Create a handle for legend, no text on plot
            yield_handle = plt.scatter([K_plot[i]], [M_plot[i]], label="Steel yield")

        # ... later, right before show:
        if yield_handle is not None:
            leg = plt.legend(
                loc="lower right",
                frameon=True,
                framealpha=1.0,      # not transparent
                edgecolor="black"    # black border
            )
            leg.get_frame().set_linewidth(1.0)


    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
