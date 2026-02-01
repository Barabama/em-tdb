# gibbs_fit.py

import os
import re
import sys
import time
import glob
import json
import logging
from typing import TypedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm


VERSION = "0.3.1"
INFO = f"""
------------------------------------------
Gibbs-Temperature Fitting (ver{VERSION})
------------------------------------------
功能描述:
- 搜索gibbs-temperature.dat文件并拟合
- 生成拟合结果和拟合图
- 生成适用于Thermo-Calc和Pandat的Gibbs-Temperature关系式

开发信息:
- 开发团队: FZU-MCMF
- 主要开发者: Chen Xing-Yu, Gao Min-Liang, Wu Bo*
- 联系方式:
  - 邮箱: wubo@fzu.edu.cn
  - QQ: 654489521

使用说明:
1. 程序自动搜索目标文件夹下所有gibbs-temperature.dat文件并拟合
  - 文件夹结构:
    <目标文件夹>
    ├── <QHA-Cu-pure>
    │   └── gibbs-temperature.dat
    ├── <QHA-FeCu>
    │   └── gibbs-temperature.dat
    ├── <QHA-CuFe>
    │   └── gibbs-temperature.dat
2. 程序自动生成拟合结果:
  - FitResult.json: 拟合结果
  - all_fits.png: 拟合图集合
3. 程序生成G-T关系式并保存为tdb.txt
------------------------------------------
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s]%(message)s")


class FitResult(TypedDict):
    name: str
    prefix: str
    elements: list[str]
    is_pure: bool
    expression: str
    params: list
    r_squared: float
    formula: str
    data: pd.DataFrame | None


def _fit_func(x, A, B, C, D, E, F):
    return A + B * x + C * x * np.log(x) + D * x**2 + E * x**3 + F * x ** (-1)


def _formula2str(A="A", B="B", C="C", D="D", E="E", F="F"):
    if isinstance(A, str):
        s = f"{A}+{B}*T+{C}*T*LN(T)+{D}*T**2+{E}*T**3+{F}*T**(-1)"
    else:
        s = f"{A:+E}{B:+E}*T+{C:+E}*T*LN(T){D:+E}*T**2{E:+E}*T**3{F:+E}*T**(-1)"
    return s


def read_gibbs_data(filepath: str, structure: str) -> pd.DataFrame:
    """Read the Gibbs-Temperature data."""
    atom_num = 2 if structure == "BCC" else 4  # atoms per sell, BCC: 2, HCP: 2, FCC: 4
    data = pd.read_csv(filepath, sep="\\s+", skiprows=1, header=None, names=["T", "G"])
    data = data[(data["T"] <= 2000) & (data["T"] >= 100)]
    data["G"] = data["G"] * 96485 / atom_num  # F = eNa = 96485 (C/mol)
    return data


def fit_gibbs_data(data: pd.DataFrame) -> tuple[list, float]:
    """Fit the Gibbs-Temperature data."""
    x = data["T"].values
    y = data["G"].values
    params, _ = curve_fit(_fit_func, x, y)
    residuals = y - _fit_func(x, *params)
    ss_res = np.sum(residuals**2)  # error sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    return params.tolist(), r_squared


def process_folder(folderpath: str, structure: str, times: int = 100) -> FitResult:
    """Process a single folder to fit Gibbs-Temperature data."""
    filepaths = glob.glob(os.path.join(folderpath, "gibbs-temperature.dat"))
    if not filepaths:
        logging.error(f"No gibbs-temperature.dat found in {folderpath}")
        return FitResult(
            name=os.path.basename(folderpath),
            prefix="",
            elements=[],
            is_pure=False,
            expression="",
            params=[],
            r_squared=0,
            formula="",
            data=None,
        )

    file = filepaths[0]
    name = os.path.basename(os.path.dirname(file))  # dirname of file
    prefix, elems = name.split("-", 1)
    elems = [e.upper() for e in re.findall(r"[A-Z][a-z]?", elems)]
    data = read_gibbs_data(file, structure)

    # Fit the Gibbs-Temperature data
    best_params = []
    best_r2 = 0
    for i in tqdm(
        range(0, times),
        desc=f"Fitting {name}",
        total=times,
        ncols=80,
        postfix={"R²": f"{best_r2:.3f}"},
    ):
        params, r2 = fit_gibbs_data(data)
        if r2 > best_r2:
            best_params = params
            best_r2 = r2
        if best_r2 > 0.999:  # stop if r2 is already high
            break

    return FitResult(
        name=name,
        prefix=prefix,
        elements=elems,
        is_pure=True if name.endswith("pure") else False,
        expression=_formula2str(),
        params=best_params,
        r_squared=best_r2,
        formula=_formula2str(*best_params),
        data=data,
    )


# def get_gibbs_files(directory: str) -> list:
#     """Generate the Gibbs-Temperature data paths."""
#     filepaths = glob.glob(os.path.join(directory, "**", "gibbs-temperature.dat"))
#     filepaths.sort(key=lambda x: ("pure" not in x, x))
#     return filepaths


def fit_gibbs(directory: str, structure: str):
    fit_results: list[FitResult] = []
    for foldername in os.listdir(directory):
        folderpath = os.path.join(directory, foldername)
        if not os.path.isdir(folderpath):  # skip files
            continue
        result = process_folder(folderpath, structure)
        fit_results.append(result)

    # Sort the results
    fit_results.sort(key=lambda x: (not x["is_pure"], x["name"]))

    return fit_results


def save_fit_results(fit_results: list[FitResult], output_json: str):
    fit_results = fit_results.copy()
    _ = [result.pop("data", None) for result in fit_results]  # del data
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(fit_results, f, indent=4)


def plot_fits(fit_results: list[FitResult], num: int, output_img: str):
    """Plot all the fits in one image."""
    fig, axes = plt.subplots(num, num, figsize=(num * 8, num * 8))

    # Ensure axes is always a 2D array
    if num == 1:
        axes = np.array([[axes]])

    for ax, result in tqdm(
        zip(axes.flatten(), fit_results),
        desc="Plotting fits",
        total=len(fit_results),
        ncols=80,
    ):
        if result["data"] is None:
            ax.set_title(result["name"])
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            data = result["data"]
            x = data["T"].values
            y = data["G"].values
            ax.plot(x, y, "o", label="Data")
            ax.plot(x, _fit_func(x, *result["params"]), "-", label="Fit")
            ax.set_title(result["name"])
            ax.set_xlabel("T")
            ax.set_ylabel("G")
            ax.legend()

    # Hide unused subplots
    for ax in axes.flatten()[len(fit_results) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_img)
    plt.close(fig)


def save_tdb(fit_results: list[FitResult], structure: str, output_tdb: str):
    def gen_tdb_text(keywd: str, ending: str, params: list) -> list[str]:
        indent1 = " " * 1  # indentation for the first line
        indent2 = " " * 5  # indentation for the second and third lines
        A, B, C, D, E, F = params
        return [
            f"{indent1}{keywd} 1.0 {A:+E}{B:+E}*T",
            f"{indent2}{C:+E}*T*LN(T)*{D:+E}*T**2{E:+E}*T**3",
            f"{indent2}{F:+E}*T**(-1); 6000 N  !",
            "",
        ]

    texts = [""]
    for result in fit_results:
        if not result["params"]:
            continue
        params = result["params"]
        if result["is_pure"]:
            keywd = f"FUNCTION SER{result['elements'][0]}"
            ending = ""
            texts.extend(gen_tdb_text(keywd, ending, params))
        else:
            e1, e2 = result["elements"]
            keywd = f"PARAMETER G({structure},{e1}:{e2};0)"
            c1, c2 = [-0.5, -0.5] if structure == "BCC" else [-0.25, -0.75]
            ending = f"{c1}*SER{e1}#{c2}*SER{e2}#"
            texts.extend(gen_tdb_text(keywd, ending, params))

            # Exchange elements and again if BCC
            if structure == "BCC" and e1 != e2:
                keywd = f"PARAMETER G({structure},{e2}:{e1};0)"
                ending = f"{c2}*SER{e2}#{c1}*SER{e1}#"
                texts.extend(gen_tdb_text(keywd, ending, params))

    with open(output_tdb, "w", encoding="utf-8") as f:
        f.writelines("\n".join(texts))


def get_directory(directory: str) -> str:
    """Get the working directory."""
    while True:
        directory = directory or input("Enter the directory >>> ").strip()
        if os.path.isdir(directory):
            return os.path.abspath(directory)
        else:
            logging.warning(f"{directory} not found. Please try again.")
            directory = ""


def handle_structure(structure: str = "") -> str:
    """return a structure (FCC, BCC, HCP) information from the configuration."""
    while True:
        structure = structure or input("Enter Structure (fcc, bcc, hcp) >>> ").upper()
        if structure in ["FCC", "BCC", "HCP"]:
            return structure
        else:
            logging.warning(f"{structure} not found. Please try again.")


def main():
    print(INFO)
    directory = sys.argv[1] if len(sys.argv) > 1 else ""
    while True:
        try:
            directory = get_directory(directory)
            structure = handle_structure()

            output_json = os.path.join(directory, "FitResult.json")
            output_img = os.path.join(directory, "all_fits.png")
            output_tdb = os.path.join(directory, "TDB.txt")
            fit_results = fit_gibbs(directory, structure)

            # plot all the fits in one image
            elem_num = int(np.ceil(np.sqrt(len(fit_results))))
            plot_fits(fit_results, elem_num, output_img)
            logging.info(f"All fit imgs saved to {output_img}")

            # save fit results
            save_fit_results(fit_results, output_json)
            logging.info(f"Fit results saved to {output_json}")

            # save tdb file
            save_tdb(fit_results, structure, output_tdb)
            logging.info(f"TDB file saved to {output_tdb}")

            time.sleep(1)
            directory = ""

        except KeyboardInterrupt:
            print("\n")
            logging.info("Thank you for using Gibbs-Tem! Exiting...")
            sys.exit(0)
        except Exception as e:
            print("\n")
            logging.exception(e)
            input("\nPress Enter to Exit >>> ")
            sys.exit(1)


if __name__ == "__main__":
    main()

# nuitka --standalone --onefile --output-dir=dist --jobs=2 --lto=yes `
# --follow-imports --enable-plugin=no-qt --onefile-no-compression `
# --enable-plugin=upx --upx-binary="D:\\Programs\\upx-5.0.2-win64\\upx.exe" `
# --nofollow-import-to=matplotlib.tests --nofollow-import-to=pandas.tests `
# --nofollow-import-to=pytest --nofollow-import-to=setuptools.tests `
# --output-filename=gibbsfit-0.3.1.exe `
# --file-version=0.3.1 `
# --copyright="(C) 2026 MCMF, Fuzhou University" `
# old/gibbs_fit.py
