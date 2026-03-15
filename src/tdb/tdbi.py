# src/tdbi.py

from copy import deepcopy
import sqlite3
from pathlib import Path
from typing import Any, TypedDict, Union, Sequence


class Elem(TypedDict):
    elem: str
    ref_state: str
    atomic_mass: float
    h298_h0: float
    s298: float


class Func(TypedDict):
    func: str
    elem: str
    temp_start: float
    temp_end: float
    expression: str
    is_continued: str


class Tdb(TypedDict):
    tdb: str
    description: str
    version: str


class Phase(TypedDict):
    phase: str
    sub_lattices: int
    stoichiometry: str
    components: str
    tdb: str


class Param(TypedDict):
    param: str
    ptype: str
    phase: str
    components: str
    order_num: int
    temp_start: float
    temp_end: float
    expression: str
    is_continued: str
    tdb: str


AllData = Union[Elem, Func, Tdb, Phase, Param]


class ThermoDBI:
    _table_meta = {
        "elements": {
            "fields": [
                "elem",
                "ref_state",
                "atomic_mass",
                "h298_h0",
                "s298",
            ],
            "primary_key": "elem",
        },
        "functions": {
            "fields": [
                "func",
                "elem",
                "temp_start",
                "temp_end",
                "expression",
                "is_continued",
            ],
            "primary_key": "func",
            "foreign_key": "elem",
        },
        "tdbs": {
            "fields": [
                "tdb",
                "description",
                "version",
                "update_time",
            ],
            "primary_key": "tdb",
        },
        "phases": {
            "fields": [
                "phase",
                "sub_lattices",
                "stoichiometry",
                "components",
                "tdb",
            ],
            "primary_key": [
                "phase",
                "tdb",
            ],
            "foreign_key": "tdb",
        },
        "parameters": {
            "fields": [
                "param",
                "ptype",
                "phase",
                "components",
                "order_num",
                "temp_start",
                "temp_end",
                "expression",
                "is_continued",
                "tdb",
            ],
            "primary_key": [
                "param",
                "tdb",
            ],
            "foreign_key": [
                "phase",
                "tdb",
            ],
        },
    }

    def __init__(self, db_path: str | Path = ":memory:"):
        is_in_memory = str(db_path) == ":memory:"
        db_path = Path(db_path) if isinstance(db_path, str) else db_path
        
        if is_in_memory:
            # For in-memory database, create connection first then initialize
            self.conn = sqlite3.connect(":memory:")
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            # Initialize tables directly using the same connection
            sql_path = Path(__file__).parent.joinpath("schema.sql")
            if not sql_path.exists():
                raise FileNotFoundError(f"{sql_path} not found")
            with open(sql_path, "r", encoding="utf-8") as f:
                sql_script = f.read()
                self.cursor.executescript(sql_script)
            # Disable foreign keys after schema creation (schema.sql enables them)
            self.conn.execute("PRAGMA foreign_keys = OFF;")
            self.conn.commit()
        else:
            # For file-based database, use existing logic
            if not db_path.exists():
                self.init_db(db_path)
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row  # access columns by name
            self.cursor = self.conn.cursor()

    def init_db(self, db_path: Path):
        """Initialize Endmember Thermodynamic Database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        sql_path = Path(__file__).parent.joinpath("schema.sql")
        if not sql_path.exists():
            raise FileNotFoundError(f"{sql_path} not found")

        with open(sql_path, "r", encoding="utf-8") as f:
            sql_script = f.read()
            cursor.executescript(sql_script)

        conn.commit()
        conn.close()

    def entries(self) -> list[str]:
        """Get list of table names in the database."""
        return list(self._table_meta.keys())

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __del__(self):
        self.close()

    def __exit__(self):
        self.__del__()

    def _create(self, table: str, data: AllData):
        """Create a new record in the specified table."""
        fields = self._table_meta[table]["fields"]
        if not set(data.keys()).issubset(set(fields)):
            raise ValueError(f"{data.keys()} not in {fields}")

        columns = ", ".join([v for v in data.keys()])
        holders = ", ".join(["?"] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({holders})"

        self.cursor.execute(sql, tuple([v for v in data.values()]))

    def _create_many(self, table: str, data_list: Sequence[AllData]):
        """Create many records in the specified table."""
        fields = self._table_meta[table]["fields"]
        if not set(data_list[0].keys()).issubset(set(fields)):
            raise ValueError(f"{data_list[0].keys()} not in {fields}")

        columns = ", ".join([v for v in data_list[0].keys()])
        holders = ", ".join(["?"] * len(data_list[0]))
        sql = f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({holders})"
        values = [tuple([v for v in d.values()]) for d in data_list]

        self.cursor.executemany(sql, values)

    def _read(self, table: str, queries: dict[str, Any], use_like: bool = False) -> list[dict]:
        """Read records from the specified table."""
        if not queries:
            values = ()
            sql = f"SELECT * FROM {table}"
        elif use_like:
            conds = [f"{k} LIKE ?" for k in queries.keys()]
            values = [f"%{v}%" for v in queries.values()]
            sql = f"SELECT * FROM {table} WHERE {' AND '.join(conds)}"
        else:
            conds = [f"{k} = ?" for k in queries.keys()]
            values = [v for v in queries.values()]
            sql = f"SELECT * FROM {table} WHERE {' AND '.join(conds)}"

        self.cursor.execute(sql, tuple(values))
        return [dict(row) for row in self.cursor.fetchall()]

    def _update(self, table: str, pk_data: dict[str, Any], data: dict[str, Any]):
        """Update a record in the specified table."""
        pk = self._table_meta[table]["primary_key"]
        pks = [pk] if isinstance(pk, str) else pk
        if set(pk_data.keys()) != set(pks):
            raise ValueError(f"{pk_data.keys()} not in {pks}")

        fields = self._table_meta[table]["fields"]
        if not set(data.keys()).issubset(set(fields)):
            raise ValueError(f"{data.keys()} not in {fields}")

        sets = ", ".join([f"{k} = ?" for k in data.keys()])
        conds = " AND ".join([f"{pk} = ?" for pk in pk_data.keys()])
        values = [v for v in data.values()]
        values.extend([v for v in pk_data.values()])
        sql = f"UPDATE {table} SET {sets} WHERE {conds}"

        self.cursor.execute(sql, tuple(values))
        self.conn.commit()

    def _delete(self, table: str, pk_data: dict[str, Any]):
        """Delete a record from the specified table."""
        pk = self._table_meta[table]["primary_key"]
        pks = [pk] if isinstance(pk, str) else pk
        if set(pk_data.keys()) != set(pks):
            raise ValueError(f"{pk_data.keys()} not in {pks}")

        conds = " AND ".join([f"{pk} = ?" for pk in pk_data.keys()])
        values = [v for v in pk_data.values()]
        sql = f"DELETE FROM {table} WHERE {conds}"

        self.cursor.execute(sql, tuple(values))
        self.conn.commit()

    # Elements CRUD
    def create_element(
        self,
        elem: str,
        ref_state: str,
        atomic_mass: float,
        h298_h0: float,
        s298: float,
    ):
        """Create a new element record.

        Args:
            elem: Element symbol.
            ref_state: Reference state.
            atomic_mass: Atomic mass.
            h298_h0: H298 - H0.
            s298: S298.
        """
        self._create(
            "elements",
            Elem(
                elem=elem,
                ref_state=ref_state,
                atomic_mass=atomic_mass,
                h298_h0=h298_h0,
                s298=s298,
            ),
        )

    def create_elements(self, data_list: list[Elem]):
        """Create multiple new element records.

        Args:
            data_list: List of element records.
        """
        self._create_many("elements", deepcopy(data_list))

    def read_element(self, elem: str = "", use_like: bool = False, **kwargs) -> list[Elem]:
        """Read element record.

        Args:
            elem: Element symbol.
            use_like: Whether to use LIKE operator.
        Returns:
            list: List of element records.
        """
        queries = {"elem": elem} if elem else {}
        return [Elem(**e) for e in self._read("elements", queries, use_like)]

    def update_element(
        self,
        elem: str,
        ref_state: str,
        atomic_mass: float,
        h298_h0: float,
        s298: float,
    ) -> list[Elem]:
        """Update element record.

        Args:
            elem: Element symbol.
            ref_state: Reference state.
            atomic_mass: Atomic mass.
            h298_h0: H298 - H0.
            s298: S298.
        Returns:
            List of updated element records.
        """
        self._update(
            "elements",
            {"elem": elem},
            {
                "ref_state": ref_state,
                "atomic_mass": atomic_mass,
                "h298_h0": h298_h0,
                "s298": s298,
            },
        )
        return self.read_element(elem)

    def delete_element(self, elem: str, cascade: bool = False, **kwargs):
        """Delete element record.

        Args:
            elem: Element symbol.
            cascade: Whether to delete all functions associated with the element.
        """
        if cascade:
            funcs = self.read_function(elem=elem)
            for func in funcs:
                self.delete_function(func["func"])
        self._delete("elements", {"elem": elem})

    # Functions CRUD
    def create_function(
        self,
        func: str,
        elem: str,
        temp_start: float,
        temp_end: float,
        expression: str,
        is_continued: str,
    ):
        """Create a new function record.

        Args:
            func: Function name.
            elem: Element symbol.
            temp_start: Start temperature.
            temp_end: End temperature.
            expression: Function expression.
            is_continued: Whether the function is continued.
        """

        self._create(
            "functions",
            Func(
                func=func,
                elem=elem,
                temp_start=temp_start,
                temp_end=temp_end,
                expression=expression,
                is_continued=is_continued,
            ),
        )

    def create_functions(self, data_list: list[Func]):
        """Create multiple new function records.

        Args:
            data_list: List of function records.
        """
        self._create_many("functions", deepcopy(data_list))

    def read_function(
        self,
        func: str = "",
        elem: str = "",
        use_like: bool = False,
        **kwargs,
    ) -> list[Func]:
        """Read function record.

        Args:
            func: Function name.
            elem: Element symbol.
            use_like: Whether to use LIKE operator.
        Return:
            list: List of function records.
        """
        if func:
            queries = {"func": func}
        elif elem:
            queries = {"elem": elem}
        else:
            queries = {}
        return [Func(**f) for f in self._read("functions", queries, use_like)]

    def update_function(
        self,
        func: str,
        elem: str,
        temp_start: float,
        temp_end: float,
        expression: str,
        is_continued: str,
    ) -> list[Func]:
        """Update function record.

        Args:
            func: Function name.
            elem: Element symbol.
            temp_start: Start temperature.
            temp_end: End temperature.
            expression: Function expression.
            is_continued: Whether the function is continued.
        Returns:
            list: Updated function record.
        """

        self._update(
            "functions",
            {"func": func},
            {
                "elem": elem,
                "temp_start": temp_start,
                "temp_end": temp_end,
                "expression": expression,
                "is_continued": is_continued,
            },
        )
        return self.read_function(func)

    def delete_function(self, func: str, **kwargs):
        """Delete function record.

        Args:
            func: Function name.
        """
        self._delete("functions", {"func": func})

    # TDB CRUD
    def create_tdb(self, tdb: str, description: str, version: str):
        """Create a new tdb record.

        Args:
            tdb: TDB name.
            description: TDB description.
            version: TDB version.
        """
        data = Tdb(tdb=tdb, description=description, version=version)
        self._create("tdbs", data)
        self.conn.commit()

    def read_tdb(self, tdb: str = "", use_like: bool = False, **kwargs) -> list[Tdb]:
        """Read tdb record.

        Args:
            tdb: TDB name.
            use_like: Whether to use LIKE operator.
        Returns:
            list: List of tdb records.
        """
        queries = {"tdb": tdb} if tdb else {}
        return [Tdb(**t) for t in self._read("tdbs", queries, use_like)]

    def update_tdb(self, tdb: str, description: str, version: str) -> list[Tdb]:
        """Update tdb record.

        Args:
            tdb: TDB name.
            description: TDB description.
            version: TDB version.
        Returns:
            list: Updated tdb record.
        """

        self._update("tdbs", {"tdb": tdb}, {"description": description, "version": version})
        return self.read_tdb(tdb)

    def delete_tdb(self, tdb: str, cascade: bool = False, **kwargs):
        """Delete tdb record.

        Args:
            tdb: TDB name.
            cascade: Whether to delete all parameters and phases associated with the tdb.
        """
        if cascade:
            params = self.read_parameter(tdb=tdb)
            for param in params:
                self.delete_parameter(param["param"], tdb)
            phases = self.read_phase(tdb=tdb)
            for phase in phases:
                self.delete_phase(phase["phase"], tdb)

        self._delete("tdbs", {"tdb": tdb})

    # Phases CRUD
    def create_phase(
        self,
        phase: str,
        sub_lattices: int,
        stoichiometry: str,
        components: str,
        tdb: str,
    ):
        """Create a new phase record.

        Args:
            phase: Phase name.
            sub_lattices: Number of sublattices.
            stoichiometry: Stoichiometry.
            components: Components.
            tdb: TDB name.
        """
        self._create(
            "phases",
            Phase(
                phase=phase,
                sub_lattices=sub_lattices,
                stoichiometry=stoichiometry,
                components=components,
                tdb=tdb,
            ),
        )

    def create_phases(self, data_list: list[Phase]):
        """Create multiple new phase records.

        Args:
            data_list: List of phase records.
        """
        self._create_many("phases", deepcopy(data_list))
        self.conn.commit()

    def read_phase(
        self,
        phase: str = "",
        tdb: str = "",
        use_like: bool = False,
        **kwargs,
    ) -> list[Phase]:
        """Read phase record.

        Args:
            phase: Phase name.
            tdb: TDB name.
            use_like: Whether to use LIKE operator.
        Returns:
            list: List of phase records.
        """
        queries = {}
        if tdb:
            queries["tdb"] = tdb
        if phase:
            queries["phase"] = phase
        return [Phase(**p) for p in self._read("phases", queries, use_like)]

    def update_phase(
        self,
        phase: str,
        sub_lattices: int,
        stoichiometry: str,
        components: str,
        tdb: str,
    ) -> list[Phase]:
        """Update phase record.

        Args:
            phase: Phase name.
            sub_lattices: Number of sublattices.
            stoichiometry: Stoichiometry.
            components: Components.
            tdb: TDB name.
        Returns:
            list: Updated phase record.
        """

        self._update(
            "phases",
            {"phase": phase, "tdb": tdb},
            {
                "sub_lattices": sub_lattices,
                "stoichiometry": stoichiometry,
                "components": components,
            },
        )
        return self.read_phase(phase)

    def delete_phase(self, phase: str, tdb: str, cascade: bool = False, **kwargs):
        """Delete phase record.

        Args:
            phase: Phase name.
            tdb: TDB name.
            cascade: Whether to delete all parameters associated with the phase.
        """
        if cascade:
            params = self.read_parameter(phase=phase, tdb=tdb)
            for param in params:
                self.delete_parameter(param["param"], tdb)
        self._delete("phases", {"phase": phase, "tdb": tdb})

    # Parameters CRUD
    def create_parameter(
        self,
        ptype: str,
        phase: str,
        components: str,
        order_num: int,
        temp_start: float,
        temp_end: float,
        expression: str,
        is_continued: str,
        tdb: str,
    ):
        """Create a new parameter record.

        Args:
            ptype: Parameter type.
            phase: Phase name.
            components: Components.
            order_num: Order number.
            temp_start: Start temperature.
            temp_end: End temperature.
            expression: Parameter expression.
            is_continued: Whether the parameter is continued.
            tdb: TDB name.
        """
        self._create(
            "parameters",
            Param(
                param=f"{ptype}({phase},{components};{order_num})",
                ptype=ptype,
                phase=phase,
                components=components,
                order_num=order_num,
                temp_start=temp_start,
                temp_end=temp_end,
                expression=expression,
                is_continued=is_continued,
                tdb=tdb,
            ),
        )

    def create_parameters(self, data_list: list[Param]):
        """Create multiple new parameter records.

        Args:
            data_list: List of parameter records.
        """
        data_list = deepcopy(data_list)
        for p in data_list:
            p["param"] = f"{p['ptype']}({p['phase']},{p['components']};{p['order_num']})"
        self._create_many("parameters", data_list)

    def read_parameter(
        self,
        param: str = "",
        phase: str = "",
        tdb: str = "",
        use_like: bool = False,
        **kwargs,
    ) -> list[Param]:
        """Read parameter record.

        Args:
            param: Parameter name.
            phase: Phase name.
            tdb: TDB name.
            use_like: Whether to use LIKE operator.
        Returns:
            list: List of parameter records.
        """
        queries = {"param": param, "phase": phase, "tdb": tdb}
        if not param:
            del queries["param"]
        if not phase:
            del queries["phase"]
        if not tdb:
            del queries["tdb"]
        return [Param(**p) for p in self._read("parameters", queries, use_like)]

    def update_parameter(
        self,
        ptype: str,
        phase: str,
        components: str,
        order_num: int,
        temp_start: float,
        temp_end: float,
        expression: str,
        is_continued: str,
        tdb: str,
    ):
        """Update parameter record.

        Args:
            ptype: Parameter type.
            phase: Phase name.
            components: Components.
            order_num: Order number.
            temp_start: Start temperature.
            temp_end: End temperature.
            expression: Parameter expression.
            is_continued: Whether the parameter is continued.
            tdb: TDB name.
        """
        param = f"{ptype}({phase},{components};{order_num})"

        self._update(
            "parameters",
            {"param": param, "tdb": tdb},
            {
                "ptype": ptype,
                "phase": phase,
                "components": components,
                "order_num": order_num,
                "temp_start": temp_start,
                "temp_end": temp_end,
                "expression": expression,
                "is_continued": is_continued,
            },
        )
        return self.read_parameter(param)

    def delete_parameter(self, param: str, tdb: str, **kwargs):
        """Delete parameter record.

        Args:
            param: Parameter name.
            tdb: TDB name.
        """
        self._delete("parameters", {"param": param, "tdb": tdb})
