import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TDBFunction:
    name: str
    temp_start: float
    temp_end: float
    expression: str
    is_continued: str


@dataclass
class TDBParameter:
    name: str
    temp_start: float
    temp_end: float
    expression: str
    is_continued: str


class TDBComparator:
    def __init__(self, tdb1_path: str, tdb2_path: str):
        self.tdb1_path = Path(tdb1_path)
        self.tdb2_path = Path(tdb2_path)
        self.tdb1_name = self.tdb1_path.stem
        self.tdb2_name = self.tdb2_path.stem
        
        self.funcs1 = {}
        self.funcs2 = {}
        self.params1 = {}
        self.params2 = {}
        
    def parse_expression(self, line: str) -> str:
        """Parse TDB expression to Python expression."""
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
                lambda m: f"{float(m.group(1)):+E}*T*np.log(T)",
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
                rf"{digital}\*([^#\+\-][A-Za-z]+#)",
                lambda m: f"{float(m.group(1))}*{m.group(2)}",
            ),
        ]
        new_line = ""
        for key, pattern, formatter in pattern_map:
            for match in re.finditer(pattern, line):
                new_line += formatter(match)
        return new_line

    def parse_tdb(self, tdb_path: Path) -> Tuple[Dict[str, TDBFunction], Dict[str, TDBParameter]]:
        """Parse TDB file and extract functions and parameters."""
        with open(tdb_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        text = "".join(s.split("$", 1)[0].strip() for s in lines)
        funcs = {}
        params = {}
        
        for line in [s.strip() for s in text.split("!")]:
            line += " !"
            if line.startswith("FUNCTION"):
                match = re.match(
                    r"FUNCTION\s+(\S+)\s+(\S+)\s+([^\;].+)\s*\;\s+(\S+)\s+([YN]?)\s*\!", line
                )
                if match:
                    func, temp_start, expression, temp_end, is_continued = match.groups()
                    funcs[func] = TDBFunction(
                        name=func,
                        temp_start=float(temp_start),
                        temp_end=float(temp_end),
                        expression=self.parse_expression(expression),
                        is_continued=is_continued
                    )
            elif line.startswith("PARAMETER"):
                pattern = r"PARAMETER\s+([GL])\((\S+)\,(\S+)\;(\d)\)\s+(\S+)\s+([^\;].+)\s*\;\s+(\S+)\s+([YN]?)\s*\!"
                match = re.match(pattern, line)
                if match:
                    ptype, phase, components, order_num, temp_start, expression, temp_end, is_continues = match.groups()
                    param = f"{ptype}({phase},{components};{order_num})"
                    params[param] = TDBParameter(
                        name=param,
                        temp_start=float(temp_start),
                        temp_end=float(temp_end),
                        expression=self.parse_expression(expression),
                        is_continued=is_continues
                    )
        
        return funcs, params

    def load_tdb_files(self):
        """Load and parse both TDB files."""
        self.funcs1, self.params1 = self.parse_tdb(self.tdb1_path)
        self.funcs2, self.params2 = self.parse_tdb(self.tdb2_path)
        
        print(f"TDB1 ({self.tdb1_name}): {len(self.funcs1)} functions, {len(self.params1)} parameters")
        print(f"TDB2 ({self.tdb2_name}): {len(self.funcs2)} functions, {len(self.params2)} parameters")

    def evaluate_expression(self, expression: str, temp_values: np.ndarray, 
                            func_dict: Dict[str, TDBFunction] = None) -> np.ndarray:
        """Evaluate expression at given temperatures."""
        expr = expression
        
        if func_dict:
            for func_name, func in func_dict.items():
                if func_name in expr:
                    func_expr = func.expression
                    func_values = self.evaluate_expression(func_expr, temp_values)
                    expr = expr.replace(func_name, f"func_vals_{func_name}")
                    globals()[f"func_vals_{func_name}"] = func_values
        
        T = temp_values
        try:
            result = eval(expr, {"T": T, "np": np, **{k: v for k, v in globals().items() if k.startswith("func_vals_")}})
            return np.array(result)
        except Exception as e:
            print(f"Error evaluating expression: {expr}")
            print(f"Error: {e}")
            return np.zeros_like(temp_values)

    def generate_temp_points(self, temp_start: float = 10.0, temp_end: float = 2990.0, num_points: int = 200) -> np.ndarray:
        """Generate evenly spaced temperature points."""
        return np.linspace(temp_start, temp_end, num_points)

    def plot_function_comparison(self, output_path: str = "function_comparison.png"):
        """Plot comparison of functions from both TDB files."""
        common_funcs = set(self.funcs1.keys()) & set(self.funcs2.keys())
        
        if not common_funcs:
            print("No common functions found between TDB files.")
            return
        
        num_funcs = len(common_funcs)
        cols = 2
        rows = (num_funcs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        if num_funcs == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, func_name in enumerate(sorted(common_funcs)):
            ax = axes[idx]
            func1 = self.funcs1[func_name]
            func2 = self.funcs2[func_name]
            
            temp_points = self.generate_temp_points()
            
            values1 = self.evaluate_expression(func1.expression, temp_points)
            values2 = self.evaluate_expression(func2.expression, temp_points)
            
            ax.plot(temp_points, values1, label=self.tdb1_name, linewidth=2, alpha=0.8)
            ax.plot(temp_points, values2, label=self.tdb2_name, linewidth=2, alpha=0.8, linestyle='--')
            
            ax.set_xlabel('Temperature (K)', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(f'FUNCTION: {func_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            diff = np.abs(values1 - values2)
            max_diff = np.max(diff)
            ax.text(0.02, 0.98, f'Max diff: {max_diff:.2e}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for idx in range(num_funcs, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Function comparison saved to {output_path}")
        plt.close()

    def plot_parameter_comparison(self, output_path: str = "parameter_comparison.png"):
        """Plot comparison of parameters from both TDB files."""
        common_params = set(self.params1.keys()) & set(self.params2.keys())
        
        if not common_params:
            print("No common parameters found between TDB files.")
            return
        
        num_params = len(common_params)
        cols = 2
        rows = (num_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        if num_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, param_name in enumerate(sorted(common_params)):
            ax = axes[idx]
            param1 = self.params1[param_name]
            param2 = self.params2[param_name]
            
            temp_points = self.generate_temp_points()
            
            values1 = self.evaluate_expression(param1.expression, temp_points, self.funcs1)
            values2 = self.evaluate_expression(param2.expression, temp_points, self.funcs2)
            
            ax.plot(temp_points, values1, label=self.tdb1_name, linewidth=2, alpha=0.8)
            ax.plot(temp_points, values2, label=self.tdb2_name, linewidth=2, alpha=0.8, linestyle='--')
            
            ax.set_xlabel('Temperature (K)', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(f'PARAMETER: {param_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            diff = np.abs(values1 - values2)
            max_diff = np.max(diff)
            ax.text(0.02, 0.98, f'Max diff: {max_diff:.2e}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for idx in range(num_params, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Parameter comparison saved to {output_path}")
        plt.close()

    def plot_all_comparisons(self):
        """Plot all comparisons (functions and parameters)."""
        self.plot_function_comparison()
        self.plot_parameter_comparison()


if __name__ == "__main__":
    tdb1_path = r"d:\Documents\Projects\SOFsTools\mpea-tdb-fit\BCC+FCC+HCP-FeMnNi.tdb"
    tdb2_path = r"d:\Documents\Projects\SOFsTools\mpea-tdb-fit\remote_data_fit.tdb"
    
    comparator = TDBComparator(tdb1_path, tdb2_path)
    comparator.load_tdb_files()
    comparator.plot_all_comparisons()
