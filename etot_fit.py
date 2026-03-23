import re

from pathlib import Path


from emtdb.config import PHASE_METRICS


def find_deepest_ve_dat(folder: Path | str) -> Path | None:
    folder = Path(folder) if isinstance(folder, str) else folder

    # 直接在folder中以及其所有子目录中查找v-e.dat文件
    files = []
    for item in folder.rglob("*"):
        if item.is_file() and item.name.lower() == "v-e.dat":
            files.append(item)

    if files:
        # 返回最深的文件
        deepest_file = max(files, key=lambda x: len(x.parts))
        return deepest_file
    return None


def parse_ve_dat(filepath: Path | str, atom_num: int = 1) -> float:
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    try:
        # e_data = [float(line.strip().split()[1]) for line in lines if not line.startswith("#")]
        e_data = []
        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                e_data.append(float(parts[1]))
            except ValueError:
                raise ValueError(f"Failed to parse v-e.dat file: {filepath}, parts: {parts} ")
    except (ValueError, IndexError):
        raise ValueError(f"Failed to parse v-e.dat file: {filepath}, lines: {lines} ")
    return min(e_data) * 96485 / atom_num


def export_expr(phase: str, e_data: float, elems: list[str], metrics: list[float]):

    temps = (1.00, 6000.00)
    phase = phase.upper()

    elems_str = ":".join(f"{elem.upper():2}" for elem in elems)

    if phase == "SER":
        res = [
            f"FUNCTION ETOT_SER_{elems_str} {temps[0]} {e_data:+E}; {temps[1]} N !",
            ]
    else:
        sub_expr = "".join(f"-{m}*ETOT_SER_{e.upper()}#" for e, m in zip(elems, metrics))
        res = [
            f"PARAMETER G({phase},{elems_str};0) {temps[0]} {e_data:+E}\n    {sub_expr}; {temps[1]} N !"
            ]
        if (metrics[0] == metrics[1]) and elems[0] != elems[1]:
            elems_ex = list(reversed(elems))
            elems_str_ex = ":".join(f"{elem.upper():2}" for elem in elems_ex)
            sub_expr_ex = "".join(f"-{m}*ETOT_SER_{e.upper()}#" for e, m in zip(elems_ex, metrics))
            res.append(
                f"PARAMETER G({phase},{elems_str_ex};0) {temps[0]} {e_data:+E}\n    {sub_expr_ex}; {temps[1]} N !"
            )
    return res

def process_folder(folder: Path | str):
    folder = Path(folder) if isinstance(folder, str) else folder

    name = folder.name
    parts = name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Invalid folder name: {name}")
    phase = parts[0]
    if phase not in PHASE_METRICS:
        raise ValueError(f"Invalid phase: {phase}")
    metrics = PHASE_METRICS[phase]
    m_sum = sum(metrics)
    metrics = [m / m_sum for m in metrics]
    elems = parts[1 : 1 + len(metrics)]

    if match := re.search(r"\-(\d+)(?:atoms?)?", name):
        atom_num = int(match.group(1))
    else:
        atom_num = 1

    deepest_file = find_deepest_ve_dat(folder)
    if not deepest_file:
        raise FileNotFoundError(f"No v-e.dat file found in {folder}")

    e_data = parse_ve_dat(deepest_file, atom_num)

    exprs = export_expr(phase, e_data, elems, metrics)

    return exprs


def main():
    root = Path(r"D:\Documents\Projects\SOFsTools\mpea-tdb-fit\data\16-elems-BCC+FCC-cxy-hmd")
    results = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        exprs = process_folder(folder)
        results.extend(exprs)
    results.sort()
    print("\n".join(results))


if __name__ == "__main__":
    main()
