#!/usr/bin/env python3
"""
Extract DRT-0073 cell names from OpenROAD routing output.
Run: openroad -exit route.tcl 2>&1 | python3 extract_drt_errors.py

Or save the output to a file first:
  openroad -exit route.tcl > route.log 2>&1
  python3 extract_drt_errors.py route.log
"""

import sys
import re

def extract_problematic_cells(text):
    """Extract cell names from DRT-0073 error messages."""
    # Pattern: [ERROR DRT-0073] No access point for CELLNAME/PIN.
    pattern = r'\[ERROR DRT-0073\] No access point for ([^/]+)/\w+\.'
    
    cells = set()
    for match in re.finditer(pattern, text):
        cells.add(match.group(1))
    
    return sorted(cells)

def main():
    # Read from file or stdin
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    
    cells = extract_problematic_cells(text)
    
    if not cells:
        print("No DRT-0073 errors found!")
        return
    
    print(f"Found {len(cells)} problematic cells:\n")
    
    # Output in Python set format for easy copy-paste
    print("# Add these to problematic_cells set in make_def.py:")
    for cell in cells:
        print(f'            "{cell}",')
    
    print("\n# Or as a single line:")
    print(", ".join(f'"{c}"' for c in cells))

if __name__ == "__main__":
    main()

