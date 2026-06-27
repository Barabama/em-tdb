"""
EM-TDB - Manager for parsing and managing thermodynamic database files.
"""

import re

from pathlib import Path
from itertools import groupby
from dataclasses import dataclass, asdict

from emtdb.tdb import Elem, Func, Tdb, Phase, Param, ThermoDBI


@dataclass
class ParsedData:
    elems: list[Elem]
    funcs: list[Func]
    phases: list[Phase]
    params: list[Param]
    tdb: str

    def to_dict(self):
        return asdict(self)

    def append(self, data: "ParsedData"):
        if data.tdb != self.tdb:
            raise ValueError("TDB mismatch")
        self.elems.extend(data.elems)
        self.funcs.extend(data.funcs)
        self.phases.extend(data.phases)
        self.params.extend(data.params)


class TDBParser:
    def _parse_expression(self, line: str) -> str:
        """Parse expression.

        Args:
            line: Expression string
        Returns:
            str: Parsed expression string
        """
        digital = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
        pattern_map = [
            (
                "A",
                rf"(?<!\*){digital}(?=\+|\-)",
                lambda m: f"{float(m.group(1)):+E}",
            ),
            (
                "B",
                rf"{digital}\*T(?!\*)",
                lambda m: f"{float(m.group(1)):+E}*T",
            ),
            (
                "C",
                rf"{digital}\*T\*LN\(T\)",
                lambda m: f"{float(m.group(1)):+E}*T*LN(T)",
            ),
            (
                "D",
                rf"{digital}\*T\*\*2",
                lambda m: f"{float(m.group(1)):+E}*T**2",
            ),
            (
                "E",
                rf"{digital}\*T\*\*3",
                lambda m: f"{float(m.group(1)):+E}*T**3",
            ),
            (
                "F",
                rf"{digital}\*T\*\*\(-1\)",
                lambda m: f"{float(m.group(1)):+E}*T**(-1)",
            ),
            (
                "X",
                rf"{digital}\*([^#\+\-][A-Za-z\_]+#)",
                lambda m: f"{float(m.group(1))}*{m.group(2)}",
            ),
        ]
        new_line = ""
        for key, pattern, formatter in pattern_map:
            for match in re.finditer(pattern, line):
                new_line += formatter(match)

        return new_line

    def _parse_elem(self, line: str) -> Elem | None:
        """Parse ELEMENT.

        Args:
            line: ELEMENT line
        Return:
            Elem | None: Elem dict or None
        """
        match = re.match(r"ELEMENT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\!", line)
        if not match:
            return None
        elem, ref_state, atomic_mass, h298_h0, s298 = match.groups()
        return Elem(
            elem=str(elem),
            ref_state=str(ref_state),
            atomic_mass=float(atomic_mass),
            h298_h0=float(h298_h0),
            s298=float(s298),
        )

    # Known element symbols for extracting element from function names.
    # Suffix match (longest first): SERFE → FE, GHSERCR → CR, GHCPNI → NI
    _KNOWN_ELEMENTS = frozenset({
        "AL", "CO", "CR", "CU", "FE", "HF", "MN", "MO", "NB", "NI",
        "RE", "TA", "TI", "V", "W", "ZR",
        "C", "N", "O", "B", "SI", "P", "S", "SN", "H", "VA",
    })

    @classmethod
    def _extract_elem(cls, func_name: str) -> str:
        """Extract element symbol from a function name suffix.

        Try 3-char, then 2-char, then 1-char suffixes against known elements.
        Returns empty string if no match.
        """
        upper = func_name.upper()
        for l in (3, 2, 1):
            suffix = upper[-l:]
            if suffix in cls._KNOWN_ELEMENTS:
                return suffix
        return ""

    def _parse_func(self, line: str) -> Func | None:
        """Parse FUNCTION.

        Args:
            line: FUNCTION line
        Return:
            Func | None: Func dict or None
        """
        match = re.match(
            r"FUNCTION\s+(\S+)\s+(\S+)\s+([^;]+?)(?:\s*\;\s*(\S+)\s*([YN]?)\s*\!)?\s*$",
            line.strip(),
        )
        if not match:
            return None
        func, temp_start, expression, temp_end, is_continued = match.groups()
        # Guard against split artifacts: temp_end must be a valid float,
        # and continuation markers like "N" without ! are not FUNCTION lines.
        if temp_end is not None:
            try:
                float(temp_end)
            except ValueError:
                # This is likely a continuation-split artifact (e.g., "N")
                return None
        elem = self._extract_elem(func)
        if not elem:
            return None  # skip functions without a recognizable element suffix
        return Func(
            func=str(func),
            elem=str(elem),
            temp_start=float(temp_start),
            temp_end=float(temp_end or "6000.00"),
            expression=self._parse_expression(expression or ""),
            is_continued=str(is_continued or "N"),
        )

    def _parse_param(self, line: str, tdb: str) -> Param | None:
        """Parse PARAMETER.

        Args:
            line: PARAMETER line
        Return:
            Param | None: Param dict or None
        """
        pattern = r"PARAMETER\s+([GL])\((\S+)\,(\S+)\;(\d)\)\s+(\S+)\s+([^\;].+)\s*\;\s+(\S+)\s+([YN]?)\s*\!"
        match = re.match(pattern, line)
        if not match:
            return None
        (
            ptype,
            phase,
            components,
            order_num,
            temp_start,
            expression,
            temp_end,
            is_continues,
        ) = match.groups()
        param = f"{ptype}({phase},{components};{order_num})"
        return Param(
            ptype=str(ptype),
            phase=str(phase),
            components=str(components),
            order_num=int(order_num),
            temp_start=float(temp_start),
            temp_end=float(temp_end),
            expression=self._parse_expression(expression),
            is_continued=str(is_continues),
            tdb=str(tdb),
            param=str(param),
        )

    def _parse_phase(self, line: str) -> dict | None:
        """Parse PHASE.

        Args:
            line: PHASE line
        Return:
            dict | None: Phase dict or None
        """
        match = re.match(r"PHASE\s+(\S+)\s+\%\s+(\d+)\s+([^\!].+)\!", line)
        if not match:
            return None
        phase, sub_lattices, stoichiometry = match.groups()
        sub_lattices = int(sub_lattices)
        metrics = stoichiometry.split()
        sub_lattices = len(metrics) if len(metrics) != sub_lattices else sub_lattices
        return {
            "phase": str(phase),
            "sub_lattices": int(sub_lattices),
            "stoichiometry": " ".join(metrics),
        }

    def _parse_const(self, line: str) -> dict | None:
        """Parse CONSTITUENT.

        Args:
            line: CONSTITUENT line
        Return:
            dict | None: Constituent dict or None
        """
        match = re.match(r"CONSTITUENT\s+(\S+)\s+([^\!].+)\!", line)
        if not match:
            return None
        phase, components = match.groups()
        components = components.strip(": ").split(":")
        components_list = [",".join(sorted(comp.split(","))) for comp in components]
        return {"phase": str(phase), "components": ":".join(components_list)}

    def _export_elements(self, elements: list[Elem]) -> list[str]:
        """Export ELEMENT.

        Args:
            elements: list of Elem dicts
        Return:
            list: list of ELEMENT lines formatted
        """
        lines = ["$ ELEMENT NAME REF_STATE ATOMIC_MASS H298-H0 S298 !"]
        for e in elements:
            s1 = f"ELEMENT {e['elem']:2s} {e['ref_state']:21s}"
            s2 = f"{e['atomic_mass']:E} {e['h298_h0']:E} {e['s298']:E}"
            lines.append(f"{s1} {s2} !")

        lines.extend(["$ END ELEMENT !", ""])
        return lines

    def _export_functions(self, functions: list[Func]) -> list[str]:
        """Export FUNCTION.

        Args:
            functions: list of Func dicts
        Return:
            list: list of FUNCTION lines formatted
        """
        lines = ["$ FUNCTION FUNC TEMP_START EXPRESSION TEMP_END IS_CONTINUED !"]
        for f in sorted(functions, key=lambda x: x["func"]):
            ex1, ex2 = f["expression"].split("*T*LN(T)")
            s1 = f"FUNCTION {f['func']:7s} {f['temp_start']:.2f} {ex1}*T*LN(T)"
            s2 = f"    {ex2}; {f['temp_end']:.2f} {f['is_continued']:1s}"
            lines.extend([s1, f"{s2} !"])

        lines.extend(["$ END FUNCTION !", ""])
        return lines

    def _export_phase_and_params(self, phase: Phase, params: list[Param]) -> list[str]:
        """Export PHASE and PARAMETER.

        Args:
            phase: Phase dict
            params: list of Param dicts
        Return:
            list: list of PHASE and PARAMETER lines formatted
        """
        lines = [
            "$ PHASE SUB_LATTICES STOICHIOMETRY !",
            "$ CONSTITUENT PHASE COMPONENTS !",
        ]
        s1 = f"PHASE {phase['phase']:7s} % {phase['sub_lattices']} {phase['stoichiometry']} !"
        lines.append(s1)
        comps = phase["components"].split(":")
        cur = f"CONSTITUENT {phase['phase']:7s} :"

        # Split to new lines
        for c in comps:
            if len(c) + len(cur) < 70:
                cur += f" {c}:"
            else:
                lines.append(cur)
                cur = f"    {c}:"
        lines.append(f"{cur} !")

        # PARAMETER
        for p in sorted(params, key=lambda x: x["param"]):
            parts = p["expression"].split("*T", 1)
            if len(parts) >= 2:
                ex1, ex2 = parts[0], parts[1]
            else:
                ex1 = parts[0]
                ex2 = ""
            if "*T**3" in ex2:
                ex2, ex3 = ex2.split("*T**3", 1)
                ex2 += "*T**3"
            else:
                ex3 = ""
            s1 = f"PARAMETER {p['param']} {p['temp_start']:.2f} {ex1}*T"
            s2 = f"    {ex2}"
            s3 = f"    {ex3}; {p['temp_end']:.2f} {p['is_continued']:1s}"
            lines.extend([s1, s2, f"{s3} !"])

        lines.extend(["$ PHASE AND PARAMETER DATA END !", ""])
        return lines


class TDBManager(TDBParser):
    def __init__(self, db: ThermoDBI):
        self.db = db

    def __del__(self):
        self.db.close()

    def parse_tdb(self, tdb_file: Path | str, tdb_name: str = "") -> ParsedData:
        """Parse TheromDynamics TDB file.

        Args:
            tdb_file: TDB file path.
            tdb_name: TDB name. Defaults to  TDB filename.
        Returns:
            dict: Parsed data.
        """
        tdb_file = Path(tdb_file) if isinstance(tdb_file, str) else tdb_file
        tdb_name = tdb_name or tdb_file.stem
        with open(tdb_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Clean notes
        text = "".join(s.split("$", 1)[0].strip() for s in lines)
        elems: list[Elem] = []
        funcs: list[Func] = []
        phases: list[Phase] = []
        params: list[Param] = []
        merged = []
        for line in [s.strip() for s in text.split("!")]:
            line += " !"
            # Only treat as FUNCTION/PARAMETER if line STARTS with the keyword.
            # (Continuation lines indented with spaces may contain ; temp_end N !
            #  but those are not independent FUNCTION/PARAMETER declarations.)
            if line.startswith("ELEMENT"):
                if e := self._parse_elem(line):
                    elems.append(e)
            elif line.startswith("FUNCTION"):
                if f := self._parse_func(line):
                    funcs.append(f)
            elif line.startswith("PARAMETER"):
                if p := self._parse_param(line, tdb_name):
                    params.append(p)
            elif line.startswith("PHASE"):
                if p := self._parse_phase(line):
                    merged.append(p)
            elif line.startswith("CONSTITUENT"):
                if c := self._parse_const(line):
                    merged.append(c)
            else:
                pass

        # Merge phases and constituents
        merged.sort(key=lambda x: x["phase"])
        for p, group in groupby(merged, lambda x: x["phase"]):
            data = {}
            for g in group:
                data.update({"tdb": tdb_name, **g})
            phases.append(Phase(**data))

        return ParsedData(
            elems=elems,
            funcs=funcs,
            phases=phases,
            params=params,
            tdb=tdb_name,
        )

    def save_elements(self, new_elems: list[Elem], keep: bool = True):
        """Save Elements to the database.

        Args:
            new_elems: List of Elem objects to save.
            keep: Whether to keep existing elements. Defaults to True.
        """
        db_elem_map = {e["elem"]: e for e in self.db.read_element()}

        # Create
        to_create = [e for e in new_elems if e["elem"] not in db_elem_map]
        if len(to_create) > 0:
            with self.db.conn:
                self.db.create_elements(to_create)

        # Update
        to_update = [e for e in new_elems if e["elem"] in db_elem_map]
        if keep and len(to_update) > 0:
            for e in to_update:
                with self.db.conn:
                    self.db.update_element(**e.__dict__)

    def save_functions(self, new_funcs: list[Func], keep: bool = True):
        """Save functions to database.

        Args:
            new_funcs (list[Func]): List of functions to save.
            keep (bool, optional): Whether to keep existing functions. Defaults to True.
        """
        db_func_map = {f["func"]: f for f in self.db.read_function()}

        # Create
        to_create = [f for f in new_funcs if f["func"] not in db_func_map]
        if len(to_create) > 0:
            with self.db.conn:
                self.db.create_functions(to_create)
        # Update
        to_update = [f for f in new_funcs if f["func"] in db_func_map]
        if keep and len(to_update) > 0:
            for f in to_update:
                with self.db.conn:
                    self.db.update_function(**f.__dict__)

    def import_tdb(
        self,
        phases: list[Phase],
        params: list[Param],
        tdb_name: str,
        desc: str = "",
        ver: str = "1.0",
    ):
        """Import tdb file into database.

        Args:
            phases: List of phases to import.
            params: List of parameters to import.
            tdb_name: TDB name to import.
            desc: Description of tdb. Defaults to blank.
            ver: Version of tdb. Defaults to '1.0'.
        """
        with self.db.conn:
            if not self.db.read_tdb(tdb=tdb_name):
                self.db.create_tdb(tdb_name, desc, ver)
            if len(phases) > 0:
                self.db.create_phases(phases)
            if len(params) > 0:
                self.db.create_parameters(params)

    def export_tdb(self, tdb_name: str, output: Path | str):
        """Export tdb to file.

        Args:
            tdb_name: TDB name.
            output: Output file path.
        """
        lines = []
        with self.db.conn:
            elems = self.db.read_element()
            funcs = self.db.read_function()
            tdb_data = self.db.read_tdb(tdb=tdb_name)[0]
            phases = self.db.read_phase(tdb=tdb_name)

        lines.extend([f"$ {tdb_data['description']}", ""])
        lines.extend(
            [
                "TYPE_DEFINITION % SEQ *!",
                "DEFINE_SYSTEM_DEFAULT ELEMENT 2 !",
                "DEFAULT_COMMAND DEF_SYS_ELEMENT VA !",
                "",
            ]
        )
        lines.extend(self._export_elements(elems))
        lines.extend(self._export_functions(funcs))

        for p in phases:
            with self.db.conn:
                params = self.db.read_parameter(phase=p["phase"], tdb=tdb_name)
            lines.extend(self._export_phase_and_params(p, params))
            lines.append("")

        with open(output, "w", encoding="utf-8") as f:
            f.writelines("\n".join(lines))
