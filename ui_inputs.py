# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:23:13 2026

@author: jan181
"""

# ui_inputs.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import Optional
import numpy as np
import csv
import os

# ---------- Unit conversions to SI ----------
IN_TO_M = 0.0254
KIP_TO_N = 4448.2216152605
KSI_TO_PA = 6.894757293168e6

MPA_TO_PA = 1e6
GPA_TO_PA = 1e9
KN_TO_N = 1e3

# For plotting/export in Imperial
NM_PER_KIPIN = KIP_TO_N * IN_TO_M  # 1 kip·in in N·m
INV_IN_PER_INV_M = IN_TO_M         # (1/m) -> (1/in) multiply by 0.0254


@dataclass
class UIResult:
    # Geometry (SI)
    D: float
    b: float
    cover_clear: float  # clear cover to START of bar [m]

    # Concrete inputs (SI) - user enters POSITIVE strengths and POSITIVE crushing strains
    fc_cover_pos: float
    epsc0_cover: float
    fcu_cover_pos: float
    epscu_cover_pos: float

    fc_core_pos: float
    epsc0_core: float
    fcu_core_pos: float
    epscu_core_pos: float

    # Steel (SI)
    fy: float
    Es: float

    # Steel fracture control (strain)
    eps_su: float
    fracture_tension_only: bool

    # Axial load (SI) [N] (compression negative)
    axialLoad: float

    # Curvature protocol
    numIncr: int
    maxK: float

    # Rebar layers
    nBars: list
    db: list         # [m]
    y_layers: list   # [m] from bottom to bar centroid

    distribute_evenly: bool

    # Plot options
    show_fiber_plot: bool
    show_mc_plot: bool
    mc_tick_every: int  # major ticks every N points on MC curve

    # Export
    export_csv: bool
    csv_path: Optional[str]

    # Unit system chosen (for reference)
    unit_system: str


def _to_float(s: str) -> float:
    return float(str(s).strip())


def _apply_tnr_style(root: tk.Tk):
    # Force Times New Roman across Tk + ttk
    root.option_add("*Font", ("Times New Roman", 11))
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", font=("Times New Roman", 11))
    style.configure("TLabelframe.Label", font=("Times New Roman", 11, "bold"))
    style.configure("TButton", font=("Times New Roman", 11))
    style.configure("TCheckbutton", font=("Times New Roman", 11))
    style.configure("TRadiobutton", font=("Times New Roman", 11))


def _popup_materials() -> dict:
    root = tk.Tk()
    root.withdraw()
    _apply_tnr_style(root)

    win = tk.Toplevel(root)
    win.title("Step 1/4 — Units, Geometry, Materials")
    win.resizable(False, False)

    unit_var = tk.StringVar(value="SI")
    frac_tension_only_var = tk.BooleanVar(value=True)

    entries = {}
    unit_labels = {}

    # NOTE: Concrete strengths/strains entered POSITIVE in UI
    defaults_si = {
        "D": "700",        # mm
        "b": "500",        # mm
        "axial": "-700",   # kN
        "numIncr": "200",
        "maxK": "0.03",

        "fc_cover": "25", "epsc0_cover": "0.002", "fcu_cover": "5", "epscu_cover": "0.003",
        "fc_core":  "25", "epsc0_core":  "0.002", "fcu_core":  "5", "epscu_core":  "0.015",

        "fy": "420",       # MPa
        "Es": "200",       # GPa
        "eps_su": "0.10",  # fracture strain
    }

    defaults_imp = {
        "D": "27.56",      # in
        "b": "19.69",      # in
        "axial": "-157.4", # kip
        "numIncr": "200",
        "maxK": "0.03",

        "fc_cover": "3.63", "epsc0_cover": "0.002", "fcu_cover": "0.73", "epscu_cover": "0.003",
        "fc_core":  "3.63", "epsc0_core":  "0.002", "fcu_core":  "0.73", "epscu_core":  "0.015",

        "fy": "60.9",      # ksi
        "Es": "29000",     # ksi
        "eps_su": "0.10",
    }

    def current_units():
        if unit_var.get() == "SI":
            return {"L": "mm", "F": "kN", "fc": "MPa", "fy": "MPa", "Es": "GPa", "K": "1/m"}
        return {"L": "in", "F": "kip", "fc": "ksi", "fy": "ksi", "Es": "ksi", "K": "1/in"}


    frm = ttk.Frame(win, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    # Unit system
    ufrm = ttk.LabelFrame(frm, text="Unit system", padding=8)
    ufrm.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))

    def add_row(r, label, key, unit_key=None):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        e = ttk.Entry(frm, width=16)
        e.grid(row=r, column=1, sticky="w", pady=2)
        entries[key] = e

        ul = ttk.Label(frm, text="")
        ul.grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)
        unit_labels[key] = (ul, unit_key)

    def set_defaults():
        d = defaults_si if unit_var.get() == "SI" else defaults_imp
        for k, e in entries.items():
            e.delete(0, tk.END)
            e.insert(0, d[k])

    def refresh_unit_text():
        u = current_units()
        for k, (lab, unit_key) in unit_labels.items():
            lab.config(text="" if unit_key is None else u[unit_key])

    ttk.Radiobutton(
        ufrm, text="SI (mm, kN, MPa/GPa)", variable=unit_var, value="SI",
        command=lambda: (set_defaults(), refresh_unit_text())
    ).grid(row=0, column=0, sticky="w")

    ttk.Radiobutton(
        ufrm, text="Imperial (in, kip, ksi)", variable=unit_var, value="Imperial",
        command=lambda: (set_defaults(), refresh_unit_text())
    ).grid(row=0, column=1, sticky="w")

    # Geometry / protocol
    ttk.Label(frm, text="Geometry / Loads", font=("Times New Roman", 12, "bold")).grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(0, 4)
    )
    add_row(2, "Depth D", "D", "L")
    add_row(3, "Width b", "b", "L")
    add_row(4, "Axial load P (compression is −)", "axial", "F")
    add_row(5, "Increments (numIncr)", "numIncr", None)
    add_row(6, "Max curvature ϕ", "maxK", "K")

    # ---------------- Concrete (split into two boxes) ----------------
    ttk.Label(frm, text="Concrete", font=("Times New Roman", 12, "bold")).grid(
        row=7, column=0, columnspan=3, sticky="w", pady=(10, 4)
    )
    
    def add_row_in(parent, r, label, key, unit_key=None):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        e = ttk.Entry(parent, width=16)
        e.grid(row=r, column=1, sticky="w", pady=2)
        entries[key] = e
    
        ul = ttk.Label(parent, text="")
        ul.grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)
        unit_labels[key] = (ul, unit_key)
    
    # Cover concrete box
    cover_box = ttk.LabelFrame(frm, text="Cover concrete", padding=8)
    cover_box.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(2, 8))
    
    add_row_in(cover_box, 0, "f′c,cover",   "fc_cover",    "fc")
    add_row_in(cover_box, 1, "ε₀,cover",    "epsc0_cover", None)
    add_row_in(cover_box, 2, "f′cu,cover",  "fcu_cover",   "fc")
    add_row_in(cover_box, 3, "εcu,cover",   "epscu_cover", None)
    
    # Core concrete box
    core_box = ttk.LabelFrame(frm, text="Core concrete", padding=8)
    core_box.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    
    add_row_in(core_box, 0, "f′c,core",     "fc_core",     "fc")
    add_row_in(core_box, 1, "ε₀,core",      "epsc0_core",  None)
    add_row_in(core_box, 2, "f′cu,core",    "fcu_core",    "fc")
    add_row_in(core_box, 3, "εcu,core",     "epscu_core",  None)

    # Steel
    ttk.Label(frm, text="Steel", font=("Times New Roman", 12, "bold")).grid(
        row=16, column=0, columnspan=3, sticky="w", pady=(10, 4)
    )
    add_row(17, "fy", "fy", "fy")
    add_row(18, "Es", "Es", "Es")
    add_row(19, "εsu", "eps_su", None)

    ttk.Checkbutton(
        frm,
        text="Fracture only in tension (recommended)",
        variable=frac_tension_only_var
    ).grid(row=20, column=0, columnspan=3, sticky="w", pady=(6, 0))

    result = {}

    def on_ok():
        try:
            raw = {k: entries[k].get() for k in entries}
            raw["unit_system"] = unit_var.get()
            raw["fracture_tension_only"] = bool(frac_tension_only_var.get())
            result.update(raw)
            win.destroy()
        except Exception as e:
            messagebox.showerror("Input error", f"Check inputs.\n\n{e}")

    btns = ttk.Frame(frm)
    btns.grid(row=21, column=0, columnspan=3, sticky="e", pady=(12, 0))
    ttk.Button(btns, text="Cancel", command=lambda: (result.clear(), win.destroy())).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="OK", command=on_ok).grid(row=0, column=1)

    set_defaults()
    refresh_unit_text()
    win.grab_set()
    root.wait_window(win)
    root.destroy()

    if not result:
        raise SystemExit("Cancelled by user.")
    return result


def _popup_layers_count() -> int:
    root = tk.Tk()
    root.withdraw()
    _apply_tnr_style(root)

    win = tk.Toplevel(root)
    win.title("Step 2/4 — Number of rebar layers")
    win.resizable(False, False)

    frm = ttk.Frame(win, padding=12)
    frm.grid(row=0, column=0)

    ttk.Label(frm, text="How many rebar layers? (max 10)").grid(row=0, column=0, sticky="w")
    var = tk.StringVar(value="5")
    sp = ttk.Spinbox(frm, from_=1, to=10, textvariable=var, width=6)
    sp.grid(row=0, column=1, padx=(10, 0))

    out = {"n": None}

    def ok():
        out["n"] = int(var.get())
        win.destroy()

    ttk.Button(frm, text="Cancel", command=lambda: win.destroy()).grid(row=1, column=0, pady=(10, 0), sticky="w")
    ttk.Button(frm, text="OK", command=ok).grid(row=1, column=1, pady=(10, 0), sticky="e")

    win.grab_set()
    root.wait_window(win)
    root.destroy()

    if out["n"] is None:
        raise SystemExit("Cancelled by user.")
    return out["n"]


def _popup_layers_data(n_layers: int, unit_system: str):
    root = tk.Tk()
    root.withdraw()
    _apply_tnr_style(root)

    win = tk.Toplevel(root)
    win.title("Step 3/4 — Reinforcement layers")
    win.geometry("800x520")
    win.minsize(800, 520)

    distribute_var = tk.BooleanVar(value=True)

    len_unit = "mm" if unit_system == "SI" else "in"

    frm = ttk.Frame(win, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Enter layers from TOP → BOTTOM", font=("Times New Roman", 12, "bold")).pack(anchor="w")

    ttk.Checkbutton(
        frm,
        text="Distribute layers evenly along height (only ask clear cover)",
        variable=distribute_var
    ).pack(anchor="w", pady=(6, 0))

    ttk.Label(frm, text="If unchecked, enter each layer y from bottom (to bar centroid).").pack(anchor="w")

    cover_frame = ttk.Frame(frm)
    cover_frame.pack(fill="x", pady=(10, 6))
    ttk.Label(cover_frame, text=f"Clear cover to start of bar [{len_unit}]").pack(side="left")
    cover_entry = ttk.Entry(cover_frame, width=10)
    cover_entry.pack(side="left", padx=(10, 0))
    cover_entry.insert(0, "50" if unit_system == "SI" else "2.0")

    table_frame = ttk.LabelFrame(frm, text="Layer data", padding=8)
    table_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(table_frame, highlightthickness=0)
    scroll = ttk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)

    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)

    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    ttk.Label(inner, text="Layer").grid(row=0, column=0, sticky="w", padx=6)
    ttk.Label(inner, text="# bars").grid(row=0, column=1, sticky="w", padx=6)
    ttk.Label(inner, text=f"Bar diameter [{len_unit}]").grid(row=0, column=2, sticky="w", padx=6)
    ttk.Label(inner, text=f"y from bottom to centroid [{len_unit}]").grid(row=0, column=3, sticky="w", padx=6)

    nbar_entries, db_entries, y_entries = [], [], []
    for i in range(n_layers):
        ttk.Label(inner, text=str(i + 1)).grid(row=i + 1, column=0, sticky="w", padx=6, pady=2)

        en = ttk.Entry(inner, width=10)
        en.grid(row=i + 1, column=1, sticky="w", padx=6, pady=2)
        en.insert(0, "2")

        ed = ttk.Entry(inner, width=14)
        ed.grid(row=i + 1, column=2, sticky="w", padx=6, pady=2)
        ed.insert(0, "25" if unit_system == "SI" else "1.0")

        ey = ttk.Entry(inner, width=14)
        ey.grid(row=i + 1, column=3, sticky="w", padx=6, pady=2)
        ey.insert(0, "0")

        nbar_entries.append(en)
        db_entries.append(ed)
        y_entries.append(ey)

    def toggle_y_inputs(*_):
        state = "disabled" if distribute_var.get() else "normal"
        for ey in y_entries:
            ey.configure(state=state)

    toggle_y_inputs()
    distribute_var.trace_add("write", toggle_y_inputs)

    result = {"nBars": None, "db": None, "y": None, "cover": None, "distribute": None}

    def on_ok():
        try:
            cover_val = float(cover_entry.get())
            if cover_val <= 0:
                raise ValueError("Clear cover must be > 0.")

            nBars = [int(float(e.get())) for e in nbar_entries]
            if any(n < 0 for n in nBars):
                raise ValueError("Number of bars must be >= 0.")

            db = [float(e.get()) for e in db_entries]
            if any(d <= 0 for d in db):
                raise ValueError("Bar diameters must be > 0.")

            if distribute_var.get():
                y_vals = [0.0] * n_layers
            else:
                y_vals = [float(e.get()) for e in y_entries]
                if any(y <= 0 for y in y_vals):
                    raise ValueError("All y values must be > 0.")

            result.update({
                "nBars": nBars,
                "db": db,
                "y": y_vals,
                "cover": cover_val,
                "distribute": bool(distribute_var.get()),
            })
            win.destroy()
        except Exception as e:
            messagebox.showerror("Input error", f"Check layer inputs.\n\n{e}")

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(10, 0))
    ttk.Button(btns, text="Cancel", command=lambda: win.destroy()).pack(side="left")
    ttk.Button(btns, text="OK", command=on_ok).pack(side="right")

    win.grab_set()
    root.wait_window(win)
    root.destroy()

    if result["nBars"] is None:
        raise SystemExit("Cancelled by user.")
    return result


def _popup_plot_and_export(unit_system: str) -> dict:
    root = tk.Tk()
    root.withdraw()
    _apply_tnr_style(root)

    win = tk.Toplevel(root)
    win.title("Step 4/4 — Plots + Export")
    win.resizable(False, False)

    show_fiber_var = tk.BooleanVar(value=True)
    show_mc_var = tk.BooleanVar(value=True)
    tick_every_var = tk.StringVar(value="50")

    export_var = tk.BooleanVar(value=False)

    frm = ttk.Frame(win, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    ttk.Label(frm, text="Plots", font=("Times New Roman", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))

    ttk.Checkbutton(frm, text="Show fiber section plot", variable=show_fiber_var).grid(row=1, column=0, sticky="w")
    ttk.Checkbutton(frm, text="Show moment–curvature plot", variable=show_mc_var).grid(row=2, column=0, sticky="w")

    ttk.Label(frm, text="M–ϕ plot: Y-axis tick step (kN·m in SI, kip·in in Imperial)").grid(row=3, column=0, sticky="w", pady=(10, 2))
    ttk.Entry(frm, textvariable=tick_every_var, width=10).grid(row=4, column=0, sticky="w")


    ttk.Separator(frm).grid(row=5, column=0, sticky="ew", pady=10)

    units_note = "SI units" if unit_system == "SI" else "Imperial units (kip·in and 1/in)"
    ttk.Label(frm, text=f"Export", font=("Times New Roman", 12, "bold")).grid(row=6, column=0, sticky="w", pady=(0, 6))
    ttk.Checkbutton(frm, text=f"Export M–ϕ to CSV using {units_note}", variable=export_var).grid(row=7, column=0, sticky="w")

    out = {"show_fiber": None, "show_mc": None, "tick_every": None, "export_csv": None, "csv_path": None}

    def pick_csv_path():
        default_name = "moment_curvature.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV file", "*.csv")],
            title="Save moment–curvature CSV"
        )
        return path if path else None

    def on_ok():
        try:
            tick_every = int(float(tick_every_var.get()))
            if tick_every <= 0:
                raise ValueError("Tick step must be > 0.")

            csv_path = None
            if export_var.get():
                csv_path = pick_csv_path()
                if not csv_path:
                    export_var.set(False)

            out.update({
                "show_fiber": bool(show_fiber_var.get()),
                "show_mc": bool(show_mc_var.get()),
                "tick_every": tick_every,
                "export_csv": bool(export_var.get()),
                "csv_path": csv_path,
            })
            win.destroy()
        except Exception as e:
            messagebox.showerror("Input error", f"Check plot/export options.\n\n{e}")

    btns = ttk.Frame(frm)
    btns.grid(row=8, column=0, sticky="e", pady=(12, 0))
    ttk.Button(btns, text="Cancel", command=lambda: win.destroy()).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="OK", command=on_ok).grid(row=0, column=1)

    win.grab_set()
    root.wait_window(win)
    root.destroy()

    if out["show_fiber"] is None:
        raise SystemExit("Cancelled by user.")
    return out


def get_user_inputs() -> UIResult:
    # ---- Step 1 ----
    raw = _popup_materials()
    unit_system = raw["unit_system"]

    # Convert to SI
    if unit_system == "SI":
        D = _to_float(raw["D"]) * 1e-3
        b = _to_float(raw["b"]) * 1e-3
        axialLoad = _to_float(raw["axial"]) * KN_TO_N

        fc_cover_pos = abs(_to_float(raw["fc_cover"])) * MPA_TO_PA
        fcu_cover_pos = abs(_to_float(raw["fcu_cover"])) * MPA_TO_PA
        fc_core_pos = abs(_to_float(raw["fc_core"])) * MPA_TO_PA
        fcu_core_pos = abs(_to_float(raw["fcu_core"])) * MPA_TO_PA

        fy = _to_float(raw["fy"]) * MPA_TO_PA
        Es = _to_float(raw["Es"]) * GPA_TO_PA
    else:
        D = _to_float(raw["D"]) * IN_TO_M
        b = _to_float(raw["b"]) * IN_TO_M
        axialLoad = _to_float(raw["axial"]) * KIP_TO_N

        fc_cover_pos = abs(_to_float(raw["fc_cover"])) * KSI_TO_PA
        fcu_cover_pos = abs(_to_float(raw["fcu_cover"])) * KSI_TO_PA
        fc_core_pos = abs(_to_float(raw["fc_core"])) * KSI_TO_PA
        fcu_core_pos = abs(_to_float(raw["fcu_core"])) * KSI_TO_PA

        fy = _to_float(raw["fy"]) * KSI_TO_PA
        Es = _to_float(raw["Es"]) * KSI_TO_PA

    # strains entered positive
    epsc0_cover = abs(_to_float(raw["epsc0_cover"]))
    epscu_cover_pos = abs(_to_float(raw["epscu_cover"]))
    epsc0_core = abs(_to_float(raw["epsc0_core"]))
    epscu_core_pos = abs(_to_float(raw["epscu_core"]))

    numIncr = int(_to_float(raw["numIncr"]))
    maxK_raw = _to_float(raw["maxK"])
    if unit_system == "SI":
        maxK = maxK_raw                 # already 1/m
    else:
        maxK = maxK_raw / IN_TO_M       # (1/in) -> (1/m)  because 1/in = 1/0.0254 1/m

    eps_su = abs(_to_float(raw["eps_su"]))
    fracture_tension_only = bool(raw["fracture_tension_only"])

    # ---- Step 2 ----
    n_layers = _popup_layers_count()

    # ---- Step 3 ----
    lay = _popup_layers_data(n_layers, unit_system)
    distribute = lay["distribute"]

    nBars = lay["nBars"]
    db_raw = lay["db"]
    y_raw = lay["y"]

    if unit_system == "SI":
        cover_clear = float(lay["cover"]) * 1e-3
        db = [float(d) * 1e-3 for d in db_raw]        # mm -> m
        y_layers = [float(y) * 1e-3 for y in y_raw]   # mm -> m
    else:
        cover_clear = float(lay["cover"]) * IN_TO_M
        db = [float(d) * IN_TO_M for d in db_raw]     # in -> m
        y_layers = [float(y) * IN_TO_M for y in y_raw]

    # Evenly distribute by BAR CENTROID lines using clear cover + db/2
    if distribute:
        if n_layers == 1:
            y_layers = [0.5 * D]
        else:
            y_top = D - cover_clear - db[0] / 2.0
            y_bot = cover_clear + db[-1] / 2.0
            y_layers = list(np.linspace(y_top, y_bot, n_layers))

    # ---- Step 4 ----
    pe = _popup_plot_and_export(unit_system)

    return UIResult(
        D=D, b=b, cover_clear=cover_clear,
        fc_cover_pos=fc_cover_pos, epsc0_cover=epsc0_cover, fcu_cover_pos=fcu_cover_pos, epscu_cover_pos=epscu_cover_pos,
        fc_core_pos=fc_core_pos, epsc0_core=epsc0_core, fcu_core_pos=fcu_core_pos, epscu_core_pos=epscu_core_pos,
        fy=fy, Es=Es,
        eps_su=eps_su, fracture_tension_only=fracture_tension_only,
        axialLoad=axialLoad,
        numIncr=numIncr, maxK=maxK,
        nBars=nBars, db=db, y_layers=y_layers,
        distribute_evenly=distribute,
        show_fiber_plot=pe["show_fiber"],
        show_mc_plot=pe["show_mc"],
        mc_tick_every=pe["tick_every"],
        export_csv=pe["export_csv"],
        csv_path=pe["csv_path"],
        unit_system=unit_system
    )


def export_mc_to_csv(path: str, K, M, unit_system: str):
    """
    Exports M-phi data to CSV in the same unit system selected in the UI.
      - SI: curvature [1/m], moment [N·m], [kN·m]
      - Imperial: curvature [1/in], moment [kip·in]
    """
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if unit_system == "Imperial":
        K_out = K * INV_IN_PER_INV_M
        M_out = M / NM_PER_KIPIN
        headers = ["curvature_1_per_in", "moment_kip_in"]
        rows = zip(K_out.tolist(), M_out.tolist())
    else:
        headers = ["curvature_1_per_m", "moment_Nm", "moment_kNm"]
        rows = zip(K.tolist(), M.tolist(), (M / 1000.0).tolist())

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)

