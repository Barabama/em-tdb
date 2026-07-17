#!/usr/bin/env python
"""EM-TDB Fitting GUI — tkinter frontend for ``emtdb.fitters``.

Modes
-----
- Gibbs  (SGTE polynomial ``G(T)``)
- BM3    (Birch-Murnaghan 3rd-order EOS ``E(V)``)

Each mode supports single-file and batch-folder workflows.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import urllib.request
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from emtdb.config import PHASE_METRICS, VERSION
from emtdb.fitters import (
    Bm3Fitter,
    FitResult,
    GibbsFitter,
    expand_results,
    normalize_metrics,
    parse_folder_name,
    write_tdb_file,
)

# ── constants ────────────────────────────────────────────────────────────

PHASE_NAMES = sorted(PHASE_METRICS, key=lambda p: (p != "SER", p))
PHASE_NAMES_ALL = ["All"] + PHASE_NAMES

_FILE_PATTERNS = {
    "gibbs": "gibbs[-_]temperature.dat",
    "bm3": "*v-e.dat",
}

_DEFAULT_METRICS: dict[str, str] = {
    k: " ".join(str(x) for x in v) for k, v in PHASE_METRICS.items()
}

_GITHUB_REPO = "Barabama/em-tdb"
_GITHUB_API_URL = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"
_GITHUB_WEB_URL = f"https://github.com/{_GITHUB_REPO}"

plt.style.use("ggplot")


# ========================================================================
# Background workers
# ========================================================================

class _FitWorker:
    """Execute a single fitting job on a background thread."""

    def __init__(self, q: queue.SimpleQueue, fitter_type: str,
                 path: str, name: str, phase: str, elements: list[str],
                 metrics: list[float], atom_num: int) -> None:
        self.q = q
        self.fitter_type = fitter_type
        self.path = path
        self.name = name
        self.phase = phase
        self.elements = elements
        self.metrics = metrics
        self.atom_num = atom_num

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            if self.fitter_type == "gibbs":
                fitter: GibbsFitter | Bm3Fitter = GibbsFitter(max_trials=100)
            else:
                fitter = Bm3Fitter(max_trials=30)
            result = fitter.fit_one(
                self.path, self.name, self.phase,
                self.elements, self.metrics, self.atom_num,
            )
            self.q.put(("done", result))
        except Exception as exc:
            self.q.put(("error", f"{self.name}: {exc}"))


class _UpdateChecker:
    """Check for newer EM-TDB releases via the GitHub API."""

    def __init__(self, q: queue.SimpleQueue) -> None:
        self.q = q

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            req = urllib.request.Request(
                _GITHUB_API_URL,
                headers={"User-Agent": "EM-TDB", "Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            latest_tag = data.get("tag_name", "")
            release_url = data.get("html_url", _GITHUB_WEB_URL)
            latest_digits = tuple(int(x) for x in re.findall(r"\d+", latest_tag)[:3])
            installed_digits = tuple(int(x) for x in re.findall(r"\d+", VERSION)[:3])
            has_update = bool(latest_digits and installed_digits and latest_digits > installed_digits)
            self.q.put(("update_check", {
                "ok": True, "latest_tag": latest_tag,
                "has_update": has_update, "release_url": release_url,
            }))
        except Exception as exc:
            self.q.put(("update_check", {"ok": False, "error": str(exc)}))


# ========================================================================
# Main application
# ========================================================================

class App(ttk.Frame):
    """Main GUI application."""

    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root, padding=8)
        self.root = root
        root.title("EM-TDB Fitter")
        root.minsize(960, 640)

        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._workers: list[_FitWorker] = []
        self._batch_results: list[FitResult] = []
        self._current_result: FitResult | None = None
        self._running = False
        self._error_count: int = 0
        self._tree_data: list[dict] = []
        self._tree_item_to_data: dict[str, int] = {}
        self._updates_checked: bool = False

        self._build_ui()
        self.pack(fill=tk.BOTH, expand=True)
        self._poll_queue()

        root.update_idletasks()
        w, h = 960, 640
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")

    # ── layout ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Top-level notebook: Fit | Help | About
        self._nb = ttk.Notebook(self)
        self._nb.grid(row=0, column=0, sticky="nsew")
        self._build_fit_tab()
        self._build_help_tab()
        self._build_about_tab()
        self._nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Status bar
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W, padding=4).grid(
            row=1, column=0, sticky="ew", pady=(4, 0))

    # ── Fit tab ─────────────────────────────────────────────────────────

    def _build_fit_tab(self) -> None:
        fit_frame = ttk.Frame(self._nb, padding=6)
        fit_frame.columnconfigure(0, weight=1)
        fit_frame.rowconfigure(1, weight=1)

        # Input panel
        self._input_frame = ttk.LabelFrame(fit_frame, text="Input", padding=8)
        self._input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        # Mode row (always visible at top of Input)
        mode_row = ttk.Frame(self._input_frame)
        mode_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(mode_row, text="Mode:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        self._fitter_var = tk.StringVar(value="gibbs")
        ttk.Radiobutton(mode_row, text="Gibbs-T", variable=self._fitter_var,
                         value="gibbs", command=self._on_mode_change).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(mode_row, text="EOS E-V", variable=self._fitter_var,
                         value="bm3", command=self._on_mode_change).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Separator(mode_row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        self._mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(mode_row, text="Single", variable=self._mode_var,
                         value="single", command=self._on_mode_change).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(mode_row, text="Batch", variable=self._mode_var,
                         value="batch", command=self._on_mode_change).pack(side=tk.LEFT)

        self._build_input_single()
        self._build_input_batch()
        self._on_mode_change()  # set initial visibility from current mode selection

        # Results pane (PanedWindow: text + chart)
        results_frame = ttk.Frame(fit_frame)
        results_frame.grid(row=1, column=0, sticky="nsew")
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        self._build_result_pane(results_frame)

        self._nb.add(fit_frame, text="DataFit")

    def _build_input_single(self) -> None:
        self._single_frame = ttk.Frame(self._input_frame)

        # Row 0 — file path
        row0 = ttk.Frame(self._single_frame)
        row0.pack(fill=tk.X)
        ttk.Label(row0, text="File:").pack(side=tk.LEFT, padx=(0, 4))
        self._file_var = tk.StringVar()
        ttk.Entry(row0, textvariable=self._file_var, width=58).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row0, text="Browse…", command=self._browse_file).pack(side=tk.LEFT)

        # Row 1 — params
        row1 = ttk.Frame(self._single_frame)
        row1.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row1, text="Phase:").pack(side=tk.LEFT, padx=(0, 4))
        self._phase_var = tk.StringVar(value="BCC")
        cb = ttk.Combobox(row1, textvariable=self._phase_var, values=PHASE_NAMES,
                          state="readonly", width=6)
        cb.pack(side=tk.LEFT, padx=(0, 12))
        cb.bind("<<ComboboxSelected>>", self._on_phase_changed)
        ttk.Label(row1, text="Elements:").pack(side=tk.LEFT, padx=(0, 4))
        self._elem_var = tk.StringVar(value="Fe Cr")
        ttk.Entry(row1, textvariable=self._elem_var, width=14).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(row1, text="Metrics:").pack(side=tk.LEFT, padx=(0, 4))
        self._metrics_var = tk.StringVar(value="1 1")
        ttk.Entry(row1, textvariable=self._metrics_var, width=10).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(row1, text="Atoms:").pack(side=tk.LEFT, padx=(0, 4))
        self._atom_var = tk.StringVar(value="2")
        ttk.Spinbox(row1, from_=1, to=20, textvariable=self._atom_var, width=4).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row1, text="▶ Run Fit", command=self._run_single).pack(side=tk.LEFT)

        self._single_frame.pack(fill=tk.X)

    def _build_input_batch(self) -> None:
        self._batch_frame = ttk.Frame(self._input_frame)
        # batch_frame is NOT packed here — _on_mode_change() controls visibility

        # Row 0 — folder + mode radios
        row0 = ttk.Frame(self._batch_frame)
        row0.pack(fill=tk.X)
        ttk.Label(row0, text="Folder:").pack(side=tk.LEFT, padx=(0, 4))
        self._folder_var = tk.StringVar()
        ttk.Entry(row0, textvariable=self._folder_var, width=58).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row0, text="Browse…", command=self._browse_folder).pack(side=tk.LEFT)

        # Row 1 — phase filter + metrics + scan + run
        row1 = ttk.Frame(self._batch_frame)
        row1.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row1, text="Phase:").pack(side=tk.LEFT, padx=(0, 4))
        self._phase_filter_var = tk.StringVar(value="All")
        self._phase_filter_cb = ttk.Combobox(row1, textvariable=self._phase_filter_var,
                                              values=PHASE_NAMES_ALL, width=8)
        self._phase_filter_cb.pack(side=tk.LEFT, padx=(0, 4))
        self._phase_filter_cb.bind("<<ComboboxSelected>>", self._on_batch_phase_changed)

        ttk.Label(row1, text="Metrics:").pack(side=tk.LEFT, padx=(0, 4))
        self._batch_metrics_var = tk.StringVar(value="1 1")
        ttk.Entry(row1, textvariable=self._batch_metrics_var, width=10).pack(side=tk.LEFT, padx=(0, 8))
        self._batch_metrics_var.trace_add("write", self._on_batch_metrics_changed)

        ttk.Button(row1, text="Scan", command=self._scan_folder).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row1, text="All", command=self._select_all).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row1, text="None", command=self._select_none).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row1, text="▶ Run Selected", command=self._run_batch).pack(side=tk.LEFT)

        # Row 2 — treeview
        tree_frame = ttk.Frame(self._batch_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        cols = ("name", "phase", "elements", "metrics", "status")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="extended", height=5)
        widths = {"name": 130, "phase": 55, "elements": 80, "metrics": 70, "status": 70}
        for col in cols:
            self._tree.heading(col, text=col.capitalize())
            self._tree.column(col, width=widths.get(col, 80))
        self._tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scroll.set)

    def _build_result_pane(self, parent: ttk.Frame) -> None:
        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)

        # Text panel
        text_frame = ttk.Frame(pane)
        self._r2_var = tk.StringVar()
        ttk.Label(text_frame, textvariable=self._r2_var,
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._expr_label = ttk.Label(text_frame, text="", wraplength=400)
        self._expr_label.pack(anchor="w", fill=tk.X)
        self._params_label = ttk.Label(text_frame, text="", wraplength=400,
                                        font=("Consolas", 9))
        self._params_label.pack(anchor="w", fill=tk.X, pady=(2, 0))
        ttk.Label(text_frame, text="TDB:", font=("", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self._tdb_text = scrolledtext.ScrolledText(text_frame, height=6, wrap=tk.WORD)
        self._tdb_text.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
        btn_row = ttk.Frame(text_frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="\U0001f4cb Copy TDB", command=self._copy_tdb).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="\U0001f4be Save TDB", command=self._save_tdb).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="\U0001f4ca Save Plot", command=self._save_plot).pack(side=tk.RIGHT)
        pane.add(text_frame, weight=1)

        # Chart panel
        chart_frame = ttk.Frame(pane)
        self._fig = plt.Figure(figsize=(5, 3.5), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.draw()
        self._toolbar = NavigationToolbar2Tk(self._canvas, chart_frame, pack_toolbar=False)
        self._toolbar.pack(fill=tk.X)
        pane.add(chart_frame, weight=2)

        pane.grid(row=0, column=0, sticky="nsew")

    # ── Help tab ────────────────────────────────────────────────────────

    def _build_help_tab(self) -> None:
        f = ttk.Frame(self._nb, padding=16)
        text = scrolledtext.ScrolledText(f, wrap=tk.WORD, state=tk.NORMAL)
        text.pack(fill=tk.BOTH, expand=True)

        text.tag_configure("h1", font=("", 14, "bold"), spacing3=8)
        text.tag_configure("h2", font=("", 11, "bold"), spacing1=12, spacing3=4)
        text.tag_configure("body", font=("", 10), lmargin1=0, lmargin2=16)
        text.tag_configure("bullet", font=("", 10), lmargin1=16, lmargin2=16)

        def _ins(tag: str, t: str) -> None:
            text.insert(tk.END, t + "\n", tag)

        _ins("h1", "EM-TDB Fitting GUI")
        _ins("body", "Fit Gibbs-temperature and E-V data to obtain parameters for CALPHAD-type thermodynamic databases (TDB).")
        _ins("h2", "Fitting Modes")
        _ins("h2", "Gibbs-T  (SGTE polynomial G(T))")
        _ins("body", "Fits Gibbs free energy vs temperature using the SGTE polynomial:\n"
             "    G(T) = A + B·T + C·T·ln(T) + D·T² + E·T³ + F/T")
        _ins("body", "Input: gibbs-temperature.dat (Temp K, Energy J/mol)")
        _ins("h2", "EOS E-V  (Birch-Murnaghan 3rd order)")
        _ins("body", "Fits DFT total energy vs volume:\n"
             "    E(V) = E0 + (9/16)·V0·B0·[...]  (BM3 equation)")
        _ins("body", "Input: *v-e.dat  (Volume A3, Energy eV)")
        _ins("h2", "Workflow")
        _ins("h2", "Single mode")
        _ins("body", "1.  Choose fitting mode (Gibbs-T or EOS E-V)\n"
             "2.  Click Browse to select a data file\n"
             "3.  Phase, Elements, Metrics, Atoms are auto-filled from folder name\n"
             "4.  Click Run Fit  ->  Results tab shows R-squared, expression, TDB line, and plot")
        _ins("h2", "Batch mode")
        _ins("body", "1.  Choose fitting mode and select Batch\n"
             "2.  Click Browse to choose the root folder\n"
             "3.  Click Scan  ->  treeview shows discovered datasets\n"
             "4.  Select rows, click Run Selected  ->  results accumulate in Results tab")
        _ins("h2", "Folder naming")
        _ins("body", "Folder names follow: PHASE-elem1-elem2[-atoms]")
        _ins("bullet", "BCC-TiNb      (Ti+Nb, 2 atoms from BCC default)")
        _ins("bullet", "FCC-CrAl3      (Cr+Al, 4 atoms from FCC default)")
        _ins("bullet", "SER-Re-8a      (Re, 8 atoms explicit)")
        _ins("h2", "Parameters")
        _ins("bullet", "Phase   : crystal structure (BCC, FCC, HCP, SER, OTH)")
        _ins("bullet", "Elements: space-separated symbols (e.g. Ti Nb)")
        _ins("bullet", "Metrics : sublattice ratios (default from PHASE_METRICS)")
        _ins("bullet", "Atoms   : atoms per formula unit (default from metrics sum)")
        _ins("h2", "Output")
        _ins("bullet", "R-squared   : goodness of fit (1.0 = perfect)")
        _ins("bullet", "Expression  : SGTE polynomial with numeric coefficients")
        _ins("bullet", "TDB line    : ready-to-use FUNCTION (SER) or PARAMETER (other)")
        _ins("bullet", "Plot        : data vs fitted curve (zoomable toolbar)")

        text.config(state=tk.DISABLED)
        self._nb.add(f, text="Help")

    # ── About tab ───────────────────────────────────────────────────────

    def _build_about_tab(self) -> None:
        f = ttk.Frame(self._nb, padding=24)
        f.columnconfigure(0, weight=1)

        ttk.Label(f, text="EM-TDB", font=("", 18, "bold")).grid(row=0, column=0, pady=(0, 2))
        ttk.Label(f, text="Thermodynamic Data Fitting", font=("", 10)).grid(row=1, column=0, pady=(0, 12))
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(row=2, column=0, sticky="ew", pady=8)

        vrow = ttk.Frame(f)
        vrow.grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(vrow, text="Version:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(vrow, text=VERSION).pack(side=tk.LEFT)

        grow = ttk.Frame(f)
        grow.grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(grow, text="GitHub:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        gh = tk.Label(grow, text=_GITHUB_WEB_URL, fg="blue", cursor="hand2")
        gh.pack(side=tk.LEFT)
        gh.bind("<Button-1>", lambda _: webbrowser.open(_GITHUB_WEB_URL))

        ttk.Separator(f, orient=tk.HORIZONTAL).grid(row=5, column=0, sticky="ew", pady=12)
        ttk.Label(f, text="Updates", font=("", 11, "bold")).grid(row=6, column=0, sticky="w")
        self._about_update_label = tk.Label(f, text="", anchor="w", justify=tk.LEFT)
        self._about_update_label.grid(row=7, column=0, sticky="w", pady=(4, 0))

        btn_row = ttk.Frame(f)
        btn_row.grid(row=8, column=0, sticky="w", pady=(4, 0))
        ttk.Button(btn_row, text="Check for Updates",
                   command=self._check_updates_now).pack(side=tk.LEFT, padx=(0, 4))
        self._about_download_btn = ttk.Button(btn_row, text="View on GitHub",
                                               command=lambda: webbrowser.open(_GITHUB_WEB_URL))

        ttk.Separator(f, orient=tk.HORIZONTAL).grid(row=9, column=0, sticky="ew", pady=12)
        ttk.Label(f, text="Python 3.10+ | tkinter | matplotlib | numpy",
                  font=("", 8), foreground="gray").grid(row=10, column=0, sticky="w")

        self._nb.add(f, text="About")

    # ── mode switching ──────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        if self._mode_var.get() == "single":
            self._batch_frame.pack_forget()
            self._single_frame.pack(fill=tk.X)
        else:
            self._single_frame.pack_forget()
            self._batch_frame.pack(fill=tk.X)

    def _on_phase_changed(self, _event=None) -> None:
        phase = self._phase_var.get()
        if phase in _DEFAULT_METRICS:
            self._metrics_var.set(_DEFAULT_METRICS[phase])

    def _on_batch_phase_changed(self, _event=None) -> None:
        phase = self._phase_filter_var.get()
        if phase == "All":
            return
        if phase in _DEFAULT_METRICS:
            self._batch_metrics_var.set(_DEFAULT_METRICS[phase])

    def _on_batch_metrics_changed(self, *_args) -> None:
        """Update metrics column in all ready tree rows to match the Metrics field."""
        val = self._batch_metrics_var.get().strip()
        for child in self._tree.get_children():
            if self._tree.set(child, "status") == "ready":
                self._tree.set(child, "metrics", val)
        # Unknown phases: leave existing value for manual edit

    def _on_tab_changed(self, _event=None) -> None:
        if self._nb.tab("current", "text") == "About" and not self._updates_checked:
            self._check_updates_now()

    def _browse_file(self) -> None:
        ft = self._fitter_var.get()
        if ft == "gibbs":
            path = filedialog.askopenfilename(
                title="Select gibbs-temperature.dat or QHA JSON",
                filetypes=[("Data", "*.dat *.DAT *.json"), ("All", "*.*")])
        else:
            path = filedialog.askopenfilename(
                title="Select v-e.dat",
                filetypes=[("Data", "*.dat *.DAT"), ("All", "*.*")])
        if path:
            self._file_var.set(path)
            self._autofill_from_path(path)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select folder containing end-member subfolders")
        if path:
            self._folder_var.set(path)
            self._scan_folder()

    def _autofill_from_path(self, path: str) -> None:
        parsed = parse_folder_name(Path(path).parent.name)
        if parsed is None:
            return
        phase, elements, atom_num = parsed
        self._phase_var.set(phase)
        self._elem_var.set(" ".join(elements))
        self._atom_var.set(str(atom_num))
        self._on_phase_changed()

    # ── scan ────────────────────────────────────────────────────────────

    def _scan_folder(self) -> None:
        folder = Path(self._folder_var.get())
        if not folder.is_dir():
            messagebox.showerror("Error", f"Not a valid directory:\n{folder}")
            return
        pattern = _FILE_PATTERNS[self._fitter_var.get()]
        phase_filter = self._phase_filter_var.get()
        self._tree.delete(*self._tree.get_children())
        self._tree_data = []
        self._tree_item_to_data = {}
        found_phases: set[str] = set()

        for subdir in sorted(folder.iterdir()):
            if not subdir.is_dir():
                continue
            parsed = parse_folder_name(subdir.name)
            if parsed is None:
                self._tree.insert("", tk.END, values=(subdir.name, "?", "", "", "skip: name"))
                continue
            p, elems, atom_num = parsed
            found_phases.add(p)
            if phase_filter != "All" and p != phase_filter:
                continue
            metrics_raw = list(PHASE_METRICS.get(p, (1,)))
            metrics_str = " ".join(str(int(m)) for m in metrics_raw)
            dat_files = sorted(subdir.rglob(pattern))
            if not dat_files:
                self._tree.insert("", tk.END, values=(
                    subdir.name, p, ",".join(elems), metrics_str, "skip: no data"))
                continue
            item_id = self._tree.insert("", tk.END, values=(
                subdir.name, p, ",".join(elems), metrics_str, "ready"))
            self._tree_item_to_data[item_id] = len(self._tree_data)
            self._tree_data.append({
                "name": subdir.name, "phase": p,
                "elements": elems, "atom_num": atom_num,
                "dat_path": str(dat_files[0]),
            })

        # Update phase filter dropdown with all found phases + All
        phase_values = ["All"] + sorted(found_phases)
        self._phase_filter_cb["values"] = phase_values
        cur = self._phase_filter_var.get()
        if cur not in phase_values:
            self._phase_filter_var.set(phase_values[0])

        self._status_var.set(f"Scanned: {len(self._tree_data)} dataset(s) ready")
        # Select first row with status "ready" (may not be the first child)
        children = self._tree.get_children()
        for child in children:
            if self._tree.set(child, "status") == "ready":
                self._tree.selection_set(child)
                break

    def _select_all(self) -> None:
        self._tree.selection_set(*self._tree.get_children())

    def _select_none(self) -> None:
        self._tree.selection_remove(*self._tree.get_children())

    # ── run ─────────────────────────────────────────────────────────────

    def _run_single(self) -> None:
        if self._running:
            return
        filepath = self._file_var.get().strip()
        if not filepath or not Path(filepath).exists():
            messagebox.showerror("Error", "Please select a valid data file")
            return
        try:
            raw_metrics = [float(x) for x in self._metrics_var.get().split()]
            atom_num = int(self._atom_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid metrics or atoms value")
            return
        if sum(raw_metrics) == 0:
            messagebox.showerror("Error", "Metrics sum to zero")
            return
        elements = [e.strip().upper() for e in self._elem_var.get().split()]
        self._running = True
        self._status_var.set("Fitting…")
        self._update_ui_busy(True)
        worker = _FitWorker(
            self._q, self._fitter_var.get(), filepath,
            Path(filepath).stem,
            self._phase_var.get(), elements,
            normalize_metrics(raw_metrics), atom_num,
        )
        self._workers = [worker]
        worker.start()

    def _run_batch(self) -> None:
        if self._running:
            return
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select datasets from the list")
            return
        ft = self._fitter_var.get()
        datasets = [self._tree_data[self._tree_item_to_data[i]] for i in sel
                    if i in self._tree_item_to_data]
        if not datasets:
            return

        # Validate per-dataset metrics from PHASE_METRICS
        for ds in datasets:
            raw = list(PHASE_METRICS.get(ds["phase"], (1,)))
            if sum(raw) == 0:
                messagebox.showerror("Error", f"Metrics sum to zero for {ds['name']}")
                return

        self._running = True
        self._batch_results.clear()
        self._error_count = 0
        n = len(datasets)
        self._status_var.set(f"Fitting 0/{n}…")
        self._update_ui_busy(True)
        self._workers = []
        for ds in datasets:
            raw = list(PHASE_METRICS.get(ds["phase"], (1,)))
            worker = _FitWorker(
                self._q, ft, ds["dat_path"], ds["name"],
                ds["phase"], ds["elements"], normalize_metrics(raw), ds["atom_num"],
            )
            self._workers.append(worker)
            worker.start()

    # ── queue polling ───────────────────────────────────────────────────

    def _poll_queue(self) -> None:
        try:
            while True:
                msg, payload = self._q.get_nowait()
                try:
                    if msg == "done":
                        self._on_result(payload)
                    elif msg == "error":
                        self._on_error(payload)
                    elif msg == "update_check":
                        self._on_update_result(payload)
                except Exception:
                    import traceback; traceback.print_exc()
                    self._running = False
                    self._update_ui_busy(False)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _finalize_batch(self) -> None:
        all_exp: list[FitResult] = []
        for r in self._batch_results:
            all_exp.extend(expand_results(r))
        self._batch_results = all_exp

        worst = min(self._batch_results, key=lambda r: r.r2)
        self._current_result = worst
        self._display_result(worst)

        self._tdb_text.delete("1.0", tk.END)
        for r in self._batch_results:
            self._tdb_text.insert(tk.END, f"$ {r.name}\n{r.tdb_line}\n\n")

        r2s = [r.r2 for r in self._batch_results]
        self._status_var.set(
            f"Fitted {len(self._batch_results)} result(s), "
            f"R² [{min(r2s):.4f}, {max(r2s):.4f}]")
        self._running = False
        self._update_ui_busy(False)

    def _on_result(self, result: FitResult) -> None:
        if self._mode_var.get() == "single":
            all_exp = expand_results(result)
            self._batch_results = list(all_exp)
            self._current_result = self._batch_results[0]
            self._display_result(self._current_result)
            self._status_var.set(f"Fitted: R² = {result.r2:.6f}")
            self._running = False
            self._update_ui_busy(False)
            return
        self._batch_results.append(result)
        total = len(self._workers)
        self._status_var.set(
            f"Fitted {len(self._batch_results)}/{total}…")
        if len(self._batch_results) + self._error_count >= total:
            self._finalize_batch()

    def _on_error(self, message: str) -> None:
        self._error_count += 1
        self._status_var.set(f"Error: {message}")
        if self._mode_var.get() == "batch":
            total = len(self._workers)
            if self._error_count + len(self._batch_results) >= total:
                self._finalize_batch()
        else:
            self._running = False
            self._update_ui_busy(False)

    def _on_update_result(self, data: dict) -> None:
        self._updates_checked = True
        if not data.get("ok"):
            self._about_update_label.config(
                text=f"Update check failed: {data.get('error', '?')}", fg="red")
            self._about_download_btn.pack_forget()
            return
        if data["has_update"]:
            self._about_update_label.config(
                text=f"Update available: {data['latest_tag']}  (installed: {VERSION})",
                fg="#cc3300")
            self._about_download_btn.config(
                text=f"Download {data['latest_tag']}",
                command=lambda: webbrowser.open(data["release_url"]))
            self._about_download_btn.pack(anchor="w", pady=(4, 0))
        else:
            self._about_update_label.config(text=f"Up to date ({VERSION})", fg="green")
            self._about_download_btn.pack_forget()

    # ── display ─────────────────────────────────────────────────────────

    def _display_result(self, result: FitResult) -> None:
        self._r2_var.set(f"R² = {result.r2:.6f}")
        self._expr_label.configure(text=f"Expression:\n{result.expression}")

        # Uniform params display
        ft = self._fitter_var.get()
        if ft == "gibbs":
            labels = ["A", "B", "C", "D", "E", "F"]
            params_str = "\n".join(f"  {lbl} = {v:+.4E}" for lbl, v in zip(labels, result.params))
        else:
            e0, v0, b0, b1 = result.params
            params_str = (f"  E₀ = {e0:.6f} eV\n"
                          f"  V₀ = {v0:.4f} Å³\n"
                          f"  B₀ = {b0:.4f} GPa\n"
                          f"  B₁ = {b1:.4f}")
        self._params_label.configure(text=f"Parameters:\n{params_str}")

        self._tdb_text.delete("1.0", tk.END)
        self._tdb_text.insert(tk.END, result.tdb_line)

        # Chart — unified for both fitters
        self._ax.clear()
        x, y, yf = result.x_data, result.y_data, result.y_fit
        self._ax.plot(x, y, "o", ms=4, label="Data", zorder=3)
        self._ax.plot(x, yf, "-", lw=2, label=f"Fit  R²={result.r2:.6f}", zorder=2)

        if ft == "gibbs":
            self._ax.set_xlabel("Temperature (K)")
            self._ax.set_ylabel("G (J/mol)")
            title = f"{result.phase}  {','.join(result.elements)}"
        else:
            self._ax.set_xlabel("Volume (Å³)")
            self._ax.set_ylabel("Energy (eV)")
            e0, v0, b0, _ = result.params
            title = f"{result.phase}  {','.join(result.elements)}\n" \
                    f"E₀={e0:.4f} eV  V₀={v0:.2f} Å³  B₀={b0:.1f} GPa"

        self._ax.set_title(title)
        self._ax.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw()

    # ── helpers ─────────────────────────────────────────────────────────

    def _update_ui_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for child in self._input_frame.winfo_children():
            self._set_state_recursive(child, state)

    @staticmethod
    def _set_state_recursive(widget: tk.Widget, state: str) -> None:
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            App._set_state_recursive(child, state)

    def _copy_tdb(self) -> None:
        text = self._tdb_text.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._status_var.set("TDB line copied to clipboard")

    def _save_tdb(self) -> None:
        if self._current_result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".tdb",
            filetypes=[("TDB files", "*.tdb"), ("All files", "*.*")])
        if not path:
            return
        results = self._batch_results if self._batch_results else [self._current_result]
        write_tdb_file(path, results, description="fitted via tkgui")
        self._status_var.set(f"Saved TDB to {path}")

    def _save_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")])
        if path:
            self._fig.savefig(path, dpi=150)
            self._status_var.set(f"Plot saved to {path}")

    def _check_updates_now(self) -> None:
        if VERSION == "x.x.x":
            self._updates_checked = True
            self._about_update_label.config(
                text=f"Version unknown ({VERSION!r}) — cannot check", fg="gray")
            return
        self._about_update_label.config(text="Checking…", fg="gray")
        self._updates_checked = True
        _UpdateChecker(self._q).start()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
