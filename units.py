# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:02:46 2025

@author: jre160
"""

# units.py
# Base SI units (used as fundamental units in your calculations)
m   = 1.0        # meter, base unit for length
kg  = 1.0        # kilogram, base unit for mass
sec = 1.0        # second, base unit for time
N   = 1.0        # Newton, base unit for force
Pa  = 1.0        # Pascal, base unit for pressure

# Derived force unit
kN  = 1e3 * N    # kiloNewton: 1 kN = 1000 N

# Length conversion factors
mm  = 0.001 * m  # 1 mm = 0.001 m
cm  = 0.01 * m   # 1 cm = 0.01 m
inch  = 0.0254 * m # 1 in = 0.0254 m (using 'in' for inches)
ft  = 0.3048 * m # 1 ft = 0.3048 m

# Mass conversion factor
ton = 1000 * kg  # 1 ton (metric ton or tonelada) = 1000 kg

# Pressure conversion factors
kPa = 1e3 * Pa   # 1 kPa = 1,000 Pa
MPa = 1e6 * Pa   # 1 MPa = 1,000,000 Pa
GPa = 1e9 * Pa   # 1 GPa = 1,000,000,000 Pa

# Additional useful conversions
# Pound-force (lbf) conversion: 1 lbf = 4.4482216152605 N
lbf = 4.4482216152605 * N
psi = lbf / (inch**2)  # 1 psi = 1 lbf/in², automatically in Pascals
