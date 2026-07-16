#!/usr/bin/env python
"""EM-TDB Fitting GUI — tkinter frontend for ``emtdb.fitters``.

Modes
-----
- Gibbs  (SGTE polynomial ``G(T)``)
- BM3    (Birch–Murnaghan 3rd-order EOS ``E(V)``)

Each mode supports single-file and batch-folder workflows.
"""

from __future__ import annotations

import queue
import threading
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

from emtdb.config import PHASE_METRICS
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

# Matplotlib style
plt.style.use("ggplot")


# ========================================================================
# Fitter worker — runs in a background thread
# ========================================================================

class _FitWorker:
    """Execute a single fitting job on a background thread.

    Results are posted to *queue* as ``("done", FitResult)`` or
    ``("error", message)`` tuples.
    """

    def __init__(self, queue: queue.SimpleQueue, fitter_type: str,
                 path: str, name: str, phase: str, elements: list[str],
                 metrics: list[float], atom_num: int) -> None:
        self.queue = queue
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
            self.queue.put(("done", result))
        except Exception as exc:
            self.queue.put(("error", f"{self.name}: {exc}"))


# ========================================================================
# Main application
# ========================================================================

class App(ttk.Frame):
    """Main GUI application."""

    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root, padding=8)
        self.root = root
        root.title("EM-TDB Fitter")
        root.minsize(900, 580)

        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._workers: list[_FitWorker] = []
        self._batch_results: list[FitResult] = []
        self._current_result: FitResult | None = None
        self._running = False
        self._error_count: int = 0
        self._tree_data: list = []

        self._build_ui()
        self.pack(fill=tk.BOTH, expand=True)
        self._poll_queue()

        # Centre window
        root.update_idletasks()
        w, h = 960, 640
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")

    # ── layout ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Status bar (outside PanedWindow, fixed at bottom)
        self._status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self, textvariable=self._status_var,
                           relief=tk.SUNKEN, anchor=tk.W, padding=4)
        status.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        # PanedWindow: resizable split between top (Mode+Input) and Results
        paned = tk.PanedWindow(self, orient=tk.VERTICAL,
                               sashwidth=6, sashrelief=tk.RAISED,
                               sashcursor="sb_v_double_arrow",
                               borderwidth=2)
        paned.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # ── Top pane: Mode (fixed) + Input (fixed height) ──
        top = ttk.Frame(paned)
        paned.add(top, minsize=150)
        top.columnconfigure(0, weight=0, minsize=140)
        top.columnconfigure(1, weight=1, minsize=500)
        top.rowconfigure(0, weight=1)

        # ── Mode (2×2 grid) ──
        self._mode_frame = ttk.LabelFrame(top, text="Mode", padding=6)
        self._mode_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self._fitter_var = tk.StringVar(value="gibbs")
        ttk.Radiobutton(self._mode_frame, text="Gibbs-T", variable=self._fitter_var,
                         value="gibbs", command=self._on_mode_change).grid(
            row=0, column=0, padx=2, pady=2, sticky="w")
        ttk.Radiobutton(self._mode_frame, text="EOS E-V", variable=self._fitter_var,
                         value="bm3", command=self._on_mode_change).grid(
            row=0, column=1, padx=2, pady=2, sticky="w")

        self._mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(self._mode_frame, text="Single", variable=self._mode_var,
                         value="single", command=self._on_mode_change).grid(
            row=1, column=0, padx=2, pady=2, sticky="w")
        ttk.Radiobutton(self._mode_frame, text="Batch", variable=self._mode_var,
                         value="batch", command=self._on_mode_change).grid(
            row=1, column=1, padx=2, pady=2, sticky="w")

        # ── Input panel ──
        self._input_frame = ttk.LabelFrame(top, text="Input", padding=8)
        self._input_frame.grid(row=0, column=1, sticky="nsew")

        self._single_frame = ttk.Frame(self._input_frame)
        self._batch_frame = ttk.Frame(self._input_frame)

        self._build_single()
        self._build_batch()

        self._input_frame.columnconfigure(0, weight=1)
        self._input_frame.rowconfigure(0, weight=1)
        self._single_frame.grid(row=0, column=0, sticky="nsew")

        # ── Bottom pane: Results ──
        results_pane = ttk.Frame(paned)
        paned.add(results_pane)

        self._result_nb = ttk.Notebook(results_pane)
        self._result_nb.pack(fill=tk.BOTH, expand=True)

        self._build_result_tab()

    # ── single input ────────────────────────────────────────────────────

    def _build_single(self) -> None:
        f = self._single_frame
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=0)
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=0)

        # Row 0 — col 0: file path, col 1: Browse
        file_row = ttk.Frame(f)
        file_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(file_row, text="File:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._file_var = tk.StringVar()
        ttk.Entry(file_row, textvariable=self._file_var).grid(row=0, column=1, sticky="ew")
        file_row.columnconfigure(1, weight=1)
        ttk.Button(f, text="Browse…", command=self._browse_file).grid(row=0, column=1, sticky="w", padx=(4, 0))

        # Row 1 — col 0: params, col 1: Run Fit
        params_row = ttk.Frame(f)
        params_row.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Label(params_row, text="Phase:").pack(side=tk.LEFT, padx=(0, 4))
        self._phase_var = tk.StringVar(value="BCC")
        cb = ttk.Combobox(params_row, textvariable=self._phase_var, values=PHASE_NAMES,
                          state="readonly", width=8)
        cb.pack(side=tk.LEFT, padx=(0, 8))
        cb.bind("<<ComboboxSelected>>", self._on_phase_changed)
        ttk.Label(params_row, text="Elements:").pack(side=tk.LEFT, padx=(0, 4))
        self._elem_var = tk.StringVar(value="Fe Cr")
        ttk.Entry(params_row, textvariable=self._elem_var, width=10).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(params_row, text="Metrics:").pack(side=tk.LEFT, padx=(0, 4))
        self._metrics_var = tk.StringVar(value="1 1")
        ttk.Entry(params_row, textvariable=self._metrics_var, width=8).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(params_row, text="Atoms:").pack(side=tk.LEFT, padx=(0, 4))
        self._atom_var = tk.StringVar(value="2")
        ttk.Spinbox(params_row, from_=1, to=20, textvariable=self._atom_var, width=4).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(f, text="▶ Run Fit", command=self._run_single).grid(row=1, column=1, sticky="w", padx=(4, 0))

        # spacer — pushes height to match batch-mode natural size
        tk.Frame(f, height=60).grid(row=2, column=0, columnspan=2, sticky="nsew")

    # ── batch input ─────────────────────────────────────────────────────

    def _build_batch(self) -> None:
        f = self._batch_frame
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=0)
        f.rowconfigure(2, weight=1)

        # Row 0 — col 0: folder path, col 1: Browse
        folder_frame = ttk.Frame(f)
        folder_frame.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Label(folder_frame, text="Folder:").pack(side=tk.LEFT, padx=(0, 4))
        self._folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self._folder_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(f, text="Browse…", command=self._browse_folder).grid(row=0, column=1, sticky="w")

        # Row 1 — col 0: Phase + Scan, col 1: Run Selected
        action_frame = ttk.Frame(f)
        action_frame.grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(action_frame, text="Phase:").pack(side=tk.LEFT, padx=(0, 4))
        self._phase_filter_var = tk.StringVar(value="All")
        ttk.Combobox(action_frame, textvariable=self._phase_filter_var, values=PHASE_NAMES_ALL,
                     state="readonly", width=6).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_frame, text="Scan", command=self._scan_folder).pack(side=tk.LEFT)
        ttk.Button(f, text="▶ Run Selected", command=self._run_batch).grid(row=1, column=1, sticky="w", pady=4)

        # Row 2 — dataset treeview (spans both columns)
        tree_frame = ttk.Frame(f)
        tree_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=4)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        cols = ("name", "phase", "elements", "status")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                  selectmode="extended", height=6)
        for col in cols:
            self._tree.heading(col, text=col.capitalize())
            self._tree.column(col, width=140 if col == "name" else 80)
        self._tree.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=scroll.set)

    # ── result tab (text + chart) ───────────────────────────────────────

    def _build_result_tab(self) -> None:
        pane = ttk.PanedWindow(self._result_nb, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        # Left — text
        text_frame = ttk.Frame(pane)
        self._r2_var = tk.StringVar()
        ttk.Label(text_frame, textvariable=self._r2_var,
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 4))

        self._expr_label = ttk.Label(text_frame, text="", wraplength=400)
        self._expr_label.pack(anchor="w", fill=tk.X)

        ttk.Label(text_frame, text="TDB:", font=("", 9, "bold")).pack(anchor="w", pady=(8, 2))
        self._tdb_text = scrolledtext.ScrolledText(text_frame, height=6, wrap=tk.WORD)
        self._tdb_text.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        btn_row = ttk.Frame(text_frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="\U0001f4cb Copy TDB", command=self._copy_tdb).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="\U0001f4be Save TDB", command=self._save_tdb).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="\U0001f4ca Save Plot", command=self._save_plot).pack(side=tk.RIGHT)

        pane.add(text_frame, weight=1)

        # Right — chart
        chart_frame = ttk.Frame(pane)
        self._fig = plt.Figure(figsize=(5, 3.5), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.draw()
        self._toolbar = NavigationToolbar2Tk(self._canvas, chart_frame, pack_toolbar=False)
        self._toolbar.pack(fill=tk.X)

        pane.add(chart_frame, weight=2)

        self._result_nb.add(pane, text="Results")

    # ── event handlers ──────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        if self._mode_var.get() == "single":
            self._batch_frame.grid_forget()
            self._single_frame.grid(row=0, column=0, sticky="nsew")
        else:
            self._single_frame.grid_forget()
            self._batch_frame.grid(row=0, column=0, sticky="nsew")

    def _on_phase_changed(self, _event=None) -> None:
        phase = self._phase_var.get()
        if phase in _DEFAULT_METRICS:
            self._metrics_var.set(_DEFAULT_METRICS[phase])

    def _browse_file(self) -> None:
        ft = self._fitter_var.get()
        if ft == "gibbs":
            path = filedialog.askopenfilename(
                title="Select gibbs-temperature.dat or QHA JSON",
                filetypes=[("Data files", "*.dat *.DAT *.json"), ("All files", "*.*")],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select v-e.dat",
                filetypes=[("Data files", "*.dat *.DAT"), ("All files", "*.*")],
            )
        if path:
            self._file_var.set(path)
            self._autofill_from_path(path)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select folder containing end-member subfolders")
        if path:
            self._folder_var.set(path)
            self._scan_folder()

    def _autofill_from_path(self, path: str) -> None:
        """Parse parent folder name and pre-fill phase / elements / atom_num."""
        folder_name = Path(path).parent.name
        parsed = parse_folder_name(folder_name)
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

        ft = self._fitter_var.get()
        phase_filter = self._phase_filter_var.get()
        pattern = _FILE_PATTERNS[ft]

        self._tree.delete(*self._tree.get_children())
        self._tree_data: list[dict] = []

        for subdir in sorted(folder.iterdir()):
            if not subdir.is_dir():
                continue
            parsed = parse_folder_name(subdir.name)
            if parsed is None:
                self._tree.insert("", tk.END, values=(subdir.name, "?", "", "skip: name"))
                continue
            p, elems, atom_num = parsed
            if phase_filter != "All" and p != phase_filter:
                continue

            dat_files = sorted(subdir.rglob(pattern))
            if not dat_files:
                self._tree.insert("", tk.END, values=(subdir.name, p, ",".join(elems), "skip: no data"))
                continue

            elems_str = ",".join(elems)
            self._tree.insert("", tk.END, values=(subdir.name, p, elems_str, "ready"))
            self._tree_data.append({
                "name": subdir.name,
                "phase": p,
                "elements": elems,
                "atom_num": atom_num,
                "dat_path": str(dat_files[0]),
            })

        self._status_var.set(f"Scanned: {len(self._tree_data)} dataset(s) ready")
        if self._tree_data:
            self._tree.selection_set(self._tree.get_children()[0])

    # ── run (single) ────────────────────────────────────────────────────

    def _run_single(self) -> None:
        if self._running:
            return
        filepath = self._file_var.get().strip()
        if not filepath or not Path(filepath).exists():
            messagebox.showerror("Error", "Please select a valid data file")
            return

        phase = self._phase_var.get()
        raw_metrics = [float(x) for x in self._metrics_var.get().split()]
        elements = [e.strip().upper() for e in self._elem_var.get().split()]
        atom_num = int(self._atom_var.get())

        if sum(raw_metrics) == 0:
            messagebox.showerror("Error", "Metrics sum to zero")
            return
        metrics = normalize_metrics(raw_metrics)

        self._running = True
        self._status_var.set("Fitting…")
        self._update_ui_busy(True)

        worker = _FitWorker(
            self._queue, self._fitter_var.get(), filepath,
            Path(filepath).stem, phase, elements, metrics, atom_num,
        )
        self._workers = [worker]
        worker.start()

    # ── run (batch) ─────────────────────────────────────────────────────

    def _run_batch(self) -> None:
        if self._running:
            return
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select datasets from the list")
            return

        ft = self._fitter_var.get()
        indices = [self._tree.index(item) for item in sel]
        datasets = [self._tree_data[i] for i in indices if i < len(self._tree_data)]

        if not datasets:
            return

        self._running = True
        self._batch_results.clear()
        self._error_count = 0
        n = len(datasets)
        self._status_var.set(f"Fitting 0/{n}…")
        self._update_ui_busy(True)

        # Dispatch one worker per dataset.
        self._workers = []
        for ds in datasets:
            raw_m = list(PHASE_METRICS.get(ds["phase"], (1,)))
            metrics = normalize_metrics(raw_m)
            worker = _FitWorker(
                self._queue, ft, ds["dat_path"], ds["name"],
                ds["phase"], ds["elements"], metrics, ds["atom_num"],
            )
            self._workers.append(worker)
            worker.start()

    # ── queue polling ───────────────────────────────────────────────────

    def _poll_queue(self) -> None:
        try:
            while True:
                msg, payload = self._queue.get_nowait()
                if msg == "done":
                    self._on_result(payload)
                elif msg == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _on_result(self, result: FitResult) -> None:
        # Single mode: just one result.
        if self._mode_var.get() == "single":
            expanded = expand_results(result)
            self._current_result = expanded[0]
            self._batch_results = expanded
            self._display_result(expanded[0])
            self._status_var.set(f"Fitted: R² = {result.r2:.6f}")
            self._running = False
            self._update_ui_busy(False)
            return

        # Batch mode: accumulate.
        self._batch_results.append(result)
        done = len(self._batch_results)
        total = len(self._workers)
        self._status_var.set(f"Fitted {done}/{total}…")

        if done + self._error_count >= total:
            # Expand all results.
            all_expanded: list[FitResult] = []
            for r in self._batch_results:
                all_expanded.extend(expand_results(r))
            self._batch_results = all_expanded

            # Show the first result.
            self._current_result = self._batch_results[0]
            self._display_result(self._batch_results[0])

            r2s = [r.r2 for r in self._batch_results]
            self._status_var.set(
                f"Fitted {done} dataset(s), "
                f"R² range [{min(r2s):.4f}, {max(r2s):.4f}]"
            )
            self._running = False
            self._update_ui_busy(False)

    def _on_error(self, message: str) -> None:
        self._error_count += 1
        self._status_var.set(f"Error: {message}")
        messagebox.showerror("Fit Error", message)

        if self._mode_var.get() == "batch":
            total = len(self._workers)
            done = len(self._batch_results)
            if done + self._error_count >= total:
                self._running = False
                self._update_ui_busy(False)
        else:
            self._running = False
            self._update_ui_busy(False)

    # ── display ─────────────────────────────────────────────────────────

    def _display_result(self, result: FitResult) -> None:
        self._r2_var.set(f"R² = {result.r2:.6f}")

        # Expression
        self._expr_label.configure(text=f"Expression:\n{result.expression}")

        # TDB
        self._tdb_text.delete("1.0", tk.END)
        self._tdb_text.insert(tk.END, result.tdb_line)

        # Chart
        self._ax.clear()
        ft = self._fitter_var.get()

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
            e0, v0, b0, b1 = result.params
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
        for child in self._mode_frame.winfo_children():
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
            filetypes=[("TDB files", "*.tdb"), ("All files", "*.*")],
        )
        if not path:
            return
        results = self._batch_results if self._batch_results else [self._current_result]
        write_tdb_file(path, results, description="fitted via tkgui")
        self._status_var.set(f"Saved TDB to {path}")

    def _save_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self._fig.savefig(path, dpi=150)
            self._status_var.set(f"Plot saved to {path}")


# ========================================================================
# Entry point
# ========================================================================

def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
