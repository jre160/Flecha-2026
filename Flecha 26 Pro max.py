# Flecha 26 Pro max.py
import numpy as np
from units import mm, MPa, GPa, kN  # :contentReference[oaicite:0]{index=0}
from Engine import run_moment_curvature, plot_section_and_mc

# ============================================================
# INPUTS (edit here)
# ============================================================

# ---- Section geometry ----
D     = 700*mm   # Section depth
b     = 500*mm   # Section width
cover =  50*mm   # Concrete cover to bar centroid line

# ---- Longitudinal reinforcement ----
nBars = (4, 2, 2, 2 , 4)                 # Bars per layer (top → bottom)
db    = (25*mm, 25*mm, 25*mm,25*mm,25*mm)     # Bar diameters per layer (top → bottom)

# ---- Concrete02 (cover) ----
fc_cover     = -25*MPa   # Peak compressive strength (negative)
epsc0_cover  = -0.002    # Strain at fc
fcu_cover    =  -5*MPa   # Residual stress
epscu_cover  = -0.003    # Ultimate crushing strain

# ---- Concrete02 (core) ----
fc_core      = -25*MPa   # Peak compressive strength (negative)
epsc0_core   = -0.002    # Strain at fc
fcu_core     =  -5*MPa   # Residual stress
epscu_core   = -0.015    # Ultimate crushing strain

# ---- Steel02 ----
fy     = 420*MPa   # Yield strength
Es     = 0.0021   # Elastic strain
b_kin  = 0.01      # Rupture strain

# ---- Axial load (compression negative) ----
axialLoad = -700*kN  # Constant axial force

# ---- Curvature protocol ----
numIncr = 200     # Number of increments
maxK    = 0.03   # Target maximum curvature (rotation for zeroLength)

# ---- Plot formatting ----
tick_cm = 5       # Axis ticks every 5 cm (section plot)



#%%


# ============================================================
# RUN - DO NOT TOUCH THIS
# ============================================================
K, M, fib_sec = run_moment_curvature(
    secTag=1, axialLoad=axialLoad, maxK=maxK, numIncr=numIncr,
    D=D, b=b, cover=cover, nBars=nBars, db=db,
    fc_cover=fc_cover, epsc0_cover=epsc0_cover, fcu_cover=fcu_cover, epscu_cover=epscu_cover,
    fc_core=fc_core,   epsc0_core=epsc0_core,   fcu_core=fcu_core,   epscu_core=epscu_core,
    fy=fy, Es=Es, b_kin=b_kin
)

plot_section_and_mc(fib_sec, D, b, K, M, tick_cm=tick_cm)

print(f"Steel yield strain epsy = {fy/Es:.6f}")
print(f"Final curvature reached = {K[-1]:.6f}")
print(f"Final moment reached = {M[-1]:.3e} N·m")




