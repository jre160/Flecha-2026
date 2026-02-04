# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:27:21 2026

@author: jan181
"""

# Flecha 26 Pro max.py
# Flecha_UI.py
from ui_inputs import get_user_inputs, export_mc_to_csv
from EngineV2 import run_moment_curvature, plot_section_and_mc

ui = get_user_inputs()

K, M, fib_sec, event = run_moment_curvature(
    secTag=1,
    axialLoad=ui.axialLoad,
    maxK=ui.maxK,
    numIncr=ui.numIncr,
    D=ui.D,
    b=ui.b,
    cover_clear=ui.cover_clear,
    nBars=ui.nBars,
    db=ui.db,
    y_layers=ui.y_layers,
    fc_cover_pos=ui.fc_cover_pos,
    epsc0_cover_pos=ui.epsc0_cover,
    fcu_cover_pos=ui.fcu_cover_pos,
    epscu_cover_pos=ui.epscu_cover_pos,
    fc_core_pos=ui.fc_core_pos,
    epsc0_core_pos=ui.epsc0_core,
    fcu_core_pos=ui.fcu_core_pos,
    epscu_core_pos=ui.epscu_core_pos,
    fy=ui.fy,
    Es=ui.Es,
    eps_su=ui.eps_su,
    fracture_tension_only=ui.fracture_tension_only
)

plot_section_and_mc(
    fib_sec=fib_sec, D=ui.D, b=ui.b, K=K, M=M,
    tick_cm=5,
    event=event,
    unit_system=ui.unit_system,
    mc_tick_every=ui.mc_tick_every,
    show_fiber=ui.show_fiber_plot,
    show_mc=ui.show_mc_plot
)

print(f"Steel yield strain epsy = {ui.fy/ui.Es:.6f}")
print(f"Final curvature reached = {K[-1]:.6f}")
print(f"Final moment reached = {M[-1]:.3e} N·m")

if ui.export_csv and ui.csv_path:
    export_mc_to_csv(ui.csv_path, K, M, ui.unit_system)
    print(f"Exported CSV to: {ui.csv_path}")
    if ui.unit_system == "Imperial":
        print("CSV units: curvature [1/in], moment [kip·in]")
    else:
        print("CSV units: curvature [1/m], moment [N·m] and [kN·m]")

