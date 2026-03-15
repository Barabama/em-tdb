#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract magnetic moments from VASP r3f files and compare with reference values.
"""

import os
import re
from pathlib import Path


def extract_mag_from_r3f(r3f_path: str) -> float:
    """
    Extract the total magnetic moment from a VASP r3f file.
    
    Args:
        r3f_path: Path to the r3f file
        
    Returns:
        The magnetic moment value, or None if not found
    """
    try:
        with open(r3f_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Search for mag= pattern in the last few lines
        for line in reversed(lines[-10:]):  # Check last 10 lines
            match = re.search(r'mag=\s+([-\d.]+)', line)
            if match:
                return float(match.group(1))
        
        return None
    except Exception as e:
        print(f"Error reading {r3f_path}: {e}")
        return None


def parse_endmember_name(folder_name: str) -> list:
    """
    Parse the end-member folder name to get element composition.
    
    Example:
        "FCC-Co-Co-4" -> ["Co", "Co"]
        "FCC-Ni-Co" -> ["Ni", "Co"]
        
    Args:
        folder_name: Name of the end-member folder
        
    Returns:
        List of element symbols
    """
    # Remove common prefixes and split by hyphen
    name = folder_name.replace("FCC-", "").replace("L12-", "")
    parts = name.split("-")
    
    # Filter out numeric parts (like "4" in "Co-Co-4")
    elements = [p for p in parts if not p.isdigit()]
    
    return elements


def calculate_reference_moment(elements: list, default_moments: dict) -> float:
    """
    Calculate the reference magnetic moment based on default values.
    
    For L12-AB3 structure using magnetic entropy formula:
    Ref mag = R * (0.25 * ln(1+μ_A) + 0.75 * ln(1+μ_B))
    
    where R = 8.314 J/(mol·K), μ is the magnetic moment per element
    
    Args:
        elements: List of element symbols [A, B]
        default_moments: Dictionary of default magnetic moments per element
        
    Returns:
        Reference magnetic moment (magnetic entropy contribution)
    """
    if len(elements) < 2:
        return 0.0
    
    # Gas constant R in J/(mol·K)
    R = 8.314
    
    # For AB3 structure: 25% A + 75% B
    element_a = elements[0]
    element_b = elements[1]
    
    moment_a = default_moments.get(element_a, 0.0)
    moment_b = default_moments.get(element_b, 0.0)
    
    # Linear weighted formula: R * (0.25*μ_A + 0.75*μ_B)
    reference_moment = (0.25 * moment_a + 0.75 * moment_b)
    
    return reference_moment


def main():
    """Main function to process all end-member folders."""
    
    # Configuration
    base_dir = Path(r"d:\Documents\Projects\SOFsTools\mpea-tdb-fit\data\Co-Cu-Ni-L12-AB3-qha")
    
    # Default magnetic moments (μ_B per atom)
    default_moments = {
        'Co': 1.5,
        'Cu': 0.0,
        'Ni': 0.5,
        # 'Fe': 2.2,
        # 'Cr': 0.0,
        # 'Mn': 0.0,
        # 'V': 0.0,
    }
    
    # Results storage
    results = []
    
    print(f"Processing directory: {base_dir}\n")
    print(f"Default magnetic moments: {default_moments}\n")
    print("-" * 80)
    print(f"{'Folder Name':<30} {'DFT mag':>12} {'Ref mag':>12} {'Difference':>12}")
    print("-" * 80)
    
    # Iterate through all subdirectories
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        # Look for r3f file
        r3f_path = folder / "r3f"
        
        if not r3f_path.exists():
            continue
        
        # Extract magnetic moment from r3f
        dft_mag = extract_mag_from_r3f(r3f_path)
        
        if dft_mag is None:
            print(f"{folder.name:<30} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
            continue
        
        # Parse folder name to get elements
        elements = parse_endmember_name(folder.name)
        
        # Calculate reference magnetic moment
        ref_mag = calculate_reference_moment(elements, default_moments)
        
        # Calculate difference
        diff = dft_mag - ref_mag
        
        # Store result
        results.append({
            'folder': folder.name,
            'elements': elements,
            'dft_mag': dft_mag,
            'ref_mag': ref_mag,
            'diff': diff
        })
        
        # Print row
        print(f"{folder.name:<30} {dft_mag:>12.4f} {ref_mag:>12.4f} {diff:>12.4f}")
    
    print("-" * 80)
    print(f"\nTotal folders processed: {len(results)}")
    
    # Save results to CSV file
    output_file = base_dir.parent / "magnetic_moments_comparison.csv"
    
    try:
        import csv
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Folder', 'Element_A', 'Element_B', 'DFT_Mag', 'Ref_Mag', 'Difference']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'Folder': result['folder'],
                    'Element_A': result['elements'][0] if len(result['elements']) > 0 else '',
                    'Element_B': result['elements'][1] if len(result['elements']) > 1 else '',
                    'DFT_Mag': f"{result['dft_mag']:.4f}",
                    'Ref_Mag': f"{result['ref_mag']:.4f}",
                    'Difference': f"{result['diff']:.4f}"
                })
        
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving CSV: {e}")


if __name__ == "__main__":
    main()
