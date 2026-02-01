# src/core/gibbsfit.py

import re
import glob
import json
import logging
from pathlib import Path
import traceback
from itertools import groupby
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src.tdb.tdbi import Func, Param, Phase
from src.tdb.tdbmgr import ParsedData

log = logging.getLogger(__name__)


class QhaData(TypedDict):
    name: str
    state: str
    structure: dict[str, Any]
    bulk_modulus_dft_GPa: float  # DFT E0 GPa
    volume_EOS: float  # EOS
    temperatures_K: list[float]  # T K
    thermal_expansion_t: list[float]  # K^(-1)
    bulk_modulus_GPa_t: list[float]  # B(T) GPa
    heat_capp_J_mol_K_t: list[float]  # Cp(T) J/mol/K
    gibbs_t: list[float]  # G(T)
    gruneisen_t: list[float]  # γ(T)
    volume_t: list[float]  # V(T)
    energy_kJ_mol_t: list[float]  # F(V,T) kJ/mol
    entropy_v_t: list[list[float]]  # S(V,T)
    heat_capp_J_mol_K_v_t: list[list[float]]  # Cv(V,T) J/mol/K
    helmholtz_v_t: list[list[float]]  # A(V,T)


# 原来的E-V.dat, 即DFT静态能量E0(V), 近似于 energy_kJ_mol_t[0]
# 或从store取"store_inputs eos deformation *" output["output"]["energy"]


class FitResult(TypedDict):
    name: str
    elements: list[str]
    metrics: list[float]
    phase: str
    is_ser: bool
    expression: str
    params: list
    r2: float
    data: pd.DataFrame | None


class GTFitter:
    def __init__(self, phase_metrics: dict[str, tuple]):
        self.formula = "+A+B*T+C*T*LN(T)+D*T**2+E*T**3+F*T**(-1)"
        self.phase_metrics = phase_metrics
        self.func_map = {
            "dat": self.handle_dat,
            "json": self.handle_json,
        }

    def _fit_func(self, x, A, B, C, D, E, F):
        return A + B * x + C * x * np.log(x) + D * x**2 + E * x**3 + F * x ** (-1)

    def _formula2str(self, params: list[float]) -> str:
        """Formula to string.

        Args:
            params: List of parameters.
        Returns:
            str: Formula string.
        """
        return (
            self.formula.replace("+A", f"{params[0]:+E}")
            .replace("+B", f"{params[1]:+E}")
            .replace("+C", f"{params[2]:+E}")
            .replace("+D", f"{params[3]:+E}")
            .replace("+E", f"{params[4]:+E}")
            .replace("+F", f"{params[5]:+E}")
        )

    def _read_dat(self, file: Path, atom_num: int) -> pd.DataFrame:
        """Read the gibbs-temperature.dat file.

        Args:
            file: Path to the gibbs-temperature.dat file.
            atom_num: Number of atoms.
        Returns:
            pd.DataFrame: DataFrame with T and G columns.
        """
        data = pd.read_csv(file, sep="\\s+", skiprows=1, header=None, names=["T", "G"])
        data = data[(data["T"] >= 100) & (data["T"] <= 2900)]
        data["G"] = data["G"] * 96485 / atom_num  # F=eNa=96485(C/mol)
        return data

    def _read_json(self, x: list[float], y: list[float], atom_num: int) -> pd.DataFrame:
        """Read the atomate QHA Flow json data.

        Args:
            x: List of temperatures.
            y: List of Gibbs free energies.
            atom_num: Number of atoms.
        Returns:
            pd.DataFrame: DataFrame with T and G columns.
        """
        data = pd.DataFrame({"T": x, "G": y})
        data = data[(data["T"] >= 100) & (data["T"] <= 2900)]
        data["G"] = data["G"] * 96485 / atom_num  # F=eNa=96485(C/mol)
        return data

    def _fit_data(self, data: pd.DataFrame) -> tuple[list, float]:
        """Fit the Gibbs-Temperature data.

        Args:
            data: DataFrame with T and G columns.
        Returns:
            tuple: Fitted parameters and R² value.
        """
        x = data["T"].values
        y = data["G"].values
        params, _ = curve_fit(self._fit_func, x, y)
        residuals = y - self._fit_func(x, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return params.tolist(), r2

    def _gen_fit_results(
        self,
        gt_data: pd.DataFrame,
        name: str,
        phase: str,
        elems: list[str],
        metrics: list[float],
    ) -> list[FitResult]:
        """Generate fit results.

        Args:
            gt_data: Gibbs-Temperature data.
            name: Name of the fit.
            phase: Phase name.
            elems: Elements in the phase.
            metrics: Metrics of the fit.
        Returns:
            list: List of fit results.
        """
        # Fit the Gibbs-Temperature data
        times = 100
        best_params = []
        best_r2 = 0
        for i in range(0, times):
            params, r2 = self._fit_data(gt_data)
            if r2 > best_r2:
                best_params = params
                best_r2 = r2
            if 1 - r2 < 1e-3:  # stop if r2 is already high
                break
        log.info(f"{name} {phase} {elems} {metrics} {best_params} {best_r2:.3f}")
        results = [
            FitResult(
                name=name,
                elements=elems,
                metrics=metrics,
                phase=phase,
                is_ser=(phase == "SER"),
                expression=self._formula2str(best_params),
                params=best_params,
                r2=best_r2,
                data=gt_data,
            ),
        ]
        # Handle BCC exchanged elements
        if len(elems) == 2 and elems[0] != elems[1] and metrics[0] == metrics[1]:
            results.append(
                FitResult(
                    name=f"{name}-ex",
                    elements=[elems[1], elems[0]],
                    metrics=[metrics[1], metrics[0]],
                    phase=phase,
                    is_ser=(phase == "SER"),
                    expression=self._formula2str(best_params),
                    params=best_params,
                    r2=best_r2,
                    data=gt_data,
                ),
            )
        return results

    def handle_dat(self, folder: Path | str) -> list[FitResult]:
        """Handle the gibbs-temperature.dat file in the folder.

        Args:
            folder: Path to the folder containing the gibbs-temperature.dat file.
        Returns:
            list: List of FitResult objects.
        """
        folder = Path(folder) if isinstance(folder, str) else folder
        files = glob.glob(str(folder.joinpath("**", "gibbs-temperature.dat")), recursive=True)
        if len(files) <= 0:
            raise FileNotFoundError(f"gibbs-temperature.dat not found in {folder}")
        file = files[0]
        name = folder.name
        parts = name.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid name format: {name}")
        phase = parts[0]
        if phase not in self.phase_metrics:
            raise ValueError(f"{phase} not valid in phase_metrics")
        metrics = self.phase_metrics[phase]
        m_sum = sum(m for m in metrics)
        metrics = [m / m_sum for m in metrics]
        elems = parts[1:1+len(metrics)]

        if match := re.search(r"\-(\d+)(?:atoms?)?", name):
            atom_num = int(match.group(1))
        else:
            atom_num = 1

        gt_data = self._read_dat(Path(file), atom_num)

        return self._gen_fit_results(gt_data, name, phase, elems, metrics)

    def handle_json(self, folder: Path | str) -> list[FitResult]:
        """Handle the atomate QHA Flow json file in the folder.

        Args:
            folder: Path to the folder containing the atomate QHA Flow json file.
        Returns:
            list: List of FitResult objects.
        """
        folder = Path(folder) if isinstance(folder, str) else folder
        files = glob.glob(str(folder.joinpath("*.json")), recursive=False)
        if len(files) <= 0:
            raise FileNotFoundError(f"*.json not found in {folder}")
        json_path = Path(files[0])

        if not json_path.exists():
            raise FileNotFoundError(f"{json_path} does not exist")
        with open(json_path, "r", encoding="utf-8") as jf:
            qha_data = QhaData(**json.load(jf))
        if qha_data.get("state", "failed") == "failed":
            raise ValueError(f"{json_path} is failed")

        name = folder.name
        parts = name.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid name format: {name}")
        phase = parts[0]
        if phase not in self.phase_metrics:
            raise ValueError(f"Invalid phase: {phase}")
        metrics = self.phase_metrics[phase]
        m_sum = sum(m for m in metrics)
        metrics = [m / m_sum for m in metrics]
        elems = parts[1:1+len(metrics)]

        struct = dict(qha_data["structure"])
        atoms = struct.get("sites", [])
        atom_num = len(atoms) or 1

        temps = qha_data["temperatures_K"]
        gibbs = qha_data["gibbs_t"]
        gt_data = self._read_json(temps, gibbs, atom_num)

        return self._gen_fit_results(gt_data, name, phase, elems, metrics)

    def process_folders(
        self,
        directory: Path | str,
        data_type: str,
    ) -> list[FitResult]:
        """Search for folders to fit Gibbs-Temperature data.

        Args:
            directory: Path to the directory to search.
            file_type: Type of file to handle.
        Returns:
            list: List of FitResult objects.
        """
        directory = Path(directory) if isinstance(directory, str) else directory
        if data_type not in self.func_map:
            raise ValueError(f"Invalid file_type: {data_type}")

        fit_results: list[FitResult] = []
        for folder in sorted(directory.iterdir(), key=lambda x: x.name):
            log.info(f"Processing {folder.name}")
            try:
                results: list[FitResult] = self.func_map[data_type](folder)
                fit_results.extend(results)
            except Exception as e:
                log.error(traceback.format_exc())
                log.error(f"Error processing {folder.name}: {e}")

        return fit_results

    def plot_fits(self, fit_results: list[FitResult], output: Path | str):
        """Plot all the fits in one image.

        Args:
            fit_results: The results to plot.
            output: The output image file path.
        """
        fit_results = sorted(fit_results, key=lambda x: (not x["is_ser"], x["name"]))
        output = Path(output) if isinstance(output, str) else output
        num = int(np.ceil(np.sqrt(len(fit_results))))
        fig, axes = plt.subplots(num, num, figsize=(num * 8, num * 8))
        if num == 1:
            axes = np.array([axes])

        log.info(f"Plotting {len(fit_results)} fits")
        for ax, res in zip(axes.flatten(), fit_results):
            if res["data"] is None:
                ax.set_title(res["name"])
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            data = res["data"]
            x = data["T"].values
            y = data["G"].values
            ax.plot(x, y, "o", label="Data")
            ax.plot(
                x,
                self._fit_func(x, *res["params"]),
                "-",
                label=f"R²={res['r2']:.3f}",
            )
            ax.set_title(res["name"])
            ax.legend()

        # Hide unused subplots
        for ax in axes.flatten()[len(fit_results) :]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(output)
        plt.close(fig)

    def fit2db(self, fit_results: list[FitResult], tdb_name: str) -> ParsedData:
        """Convert fit results to objects for ThermoDB.

        Args:
            fit_results: List of FitResult objects.
            tdb_name: TDB name to be added.
        Returns:
            ParsedData: ParsedData object.
        """
        funcs: list[Func] = []
        phases: list[Phase] = []
        params: list[Param] = []

        fit_results = [f for f in fit_results if f["data"] is not None]
        ser_groups = {
            is_ser: list(group)
            for is_ser, group in groupby(
                sorted(fit_results, key=lambda x: not x["is_ser"]),
                key=lambda x: not x["is_ser"],
            )
        }

        # Get ser functions
        funcs = [
            Func(
                func=f"SER{f['elements'][0]}",
                elem=f["elements"][0],
                temp_start=1.0,
                temp_end=6000.0,
                expression=f["expression"],
                is_continued="N",
            )
            for f in ser_groups.get(True, [])
        ]

        # Get non-ser parameters
        fit_params = ser_groups.get(False, [])
        phase_groups = {
            phase: list(group)
            for phase, group in groupby(
                sorted(fit_params, key=lambda x: x["phase"]),
                key=lambda x: x["phase"],
            )
        }

        for phase, fits in phase_groups.items():
            # Get the phase
            # The first metrics would be used to define the stoichiometry
            stoichiometry = set(tuple(f["metrics"]) for f in fits).pop()
            components = [
                ",".join(sorted(set(f["elements"][i] for f in fits)))
                for i in range(len(stoichiometry))
            ]
            phases.append(
                Phase(
                    phase=phase,
                    sub_lattices=len(stoichiometry),
                    stoichiometry=" ".join([str(s) for s in stoichiometry]),
                    components=":".join(components),
                    tdb=tdb_name,
                )
            )

            # Get the parameters
            for f in fits:
                elems = f["elements"]
                ser_expr = "".join(
                    f"-{stoichiometry[i]}*SER{elems[i]}#" for i in range(len(elems))
                )
                params.append(
                    Param(
                        param="",
                        ptype="G",
                        phase=phase,
                        components=":".join(elems),
                        order_num=0,
                        temp_start=1.0,
                        temp_end=6000.0,
                        tdb=tdb_name,
                        expression=f["expression"] + ser_expr,
                        is_continued="N",
                    )
                )

        return ParsedData(elems=[], funcs=funcs, phases=phases, params=params, tdb=tdb_name)
