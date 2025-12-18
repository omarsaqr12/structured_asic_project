#!/usr/bin/env python3
"""
Automated routing loop - iteratively finds and excludes problematic cells
until detailed routing succeeds.

Usage: python3 auto_route.py [design_name]
Default: python3 auto_route.py 6502
"""

import subprocess
import re
import sys
import os

def extract_problematic_cells(log_file):
    """Extract cell names from DRT-0073 errors in route.log"""
    pattern = r'\[ERROR DRT-0073\] No access point for ([^/]+)/\w+\.'
    cells = set()
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    cells.add(match.group(1))
    except FileNotFoundError:
        pass
    
    return cells

def update_make_def(new_cells, batch_num):
    """Add new cells to the problematic_cells sets in make_def.py"""
    with open('make_def.py', 'r') as f:
        content = f.read()
    
    # Create the new batch entries
    batch_comment = f"            # Batch {batch_num} (auto-added)\n"
    batch_entries = "\n".join(f'            "{cell}",' for cell in sorted(new_cells))
    new_batch = batch_comment + batch_entries + "\n"
    
    # Find the two locations and add the batch
    # Location 1: in get_io_ports_and_nets function
    marker1 = "        }\n        \n        # Find bits connected to problematic cells"
    if marker1 in content:
        content = content.replace(marker1, new_batch + "        }\n        \n        # Find bits connected to problematic cells")
    
    # Location 2: in generate_def_file function  
    marker2 = "    }\n    \n    def is_problematic_cell(cell_name):"
    if marker2 in content:
        content = content.replace(marker2, new_batch.replace("            ", "        ") + "    }\n    \n    def is_problematic_cell(cell_name):")
    
    with open('make_def.py', 'w') as f:
        f.write(content)
    
    print(f"  Updated make_def.py with batch {batch_num} ({len(new_cells)} cells)")

def run_iteration(design, iteration):
    """Run one iteration of DEF generation and routing"""
    print(f"\n{'='*50}")
    print(f"Iteration {iteration}")
    print(f"{'='*50}")
    
    # Generate DEF
    print("  Generating DEF...")
    result = subprocess.run(
        ['python3', 'make_def.py', '--design', design],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR in make_def.py: {result.stderr}")
        return None
    
    # Run OpenROAD
    print("  Running OpenROAD routing...")
    with open('route.log', 'w') as log:
        result = subprocess.run(
            ['openroad', '-exit', 'route.tcl'],
            stdout=log, stderr=subprocess.STDOUT
        )
    
    # Extract any new problematic cells
    new_cells = extract_problematic_cells('route.log')
    
    return new_cells

def main():
    design = sys.argv[1] if len(sys.argv) > 1 else '6502'
    max_iterations = 30
    all_problematic = set()
    
    print(f"Automated routing for design: {design}")
    print(f"Max iterations: {max_iterations}")
    
    for iteration in range(1, max_iterations + 1):
        new_cells = run_iteration(design, iteration)
        
        if new_cells is None:
            print("ERROR: DEF generation failed")
            sys.exit(1)
        
        if not new_cells:
            print(f"\n{'='*50}")
            print("SUCCESS! Detailed routing completed!")
            print(f"{'='*50}")
            print(f"Total iterations: {iteration}")
            print(f"Total problematic cells excluded: {len(all_problematic)}")
            
            # Save the final list
            with open('problematic_cells_final.txt', 'w') as f:
                for cell in sorted(all_problematic):
                    f.write(f"{cell}\n")
            print(f"Cell list saved to: problematic_cells_final.txt")
            sys.exit(0)
        
        # Check for truly new cells
        truly_new = new_cells - all_problematic
        if not truly_new:
            print(f"  No NEW cells found (same {len(new_cells)} cells as before)")
            print("  This might indicate a deeper issue. Check route.log")
            sys.exit(1)
        
        print(f"  Found {len(truly_new)} new problematic cells:")
        for cell in sorted(truly_new)[:5]:
            print(f"    - {cell}")
        if len(truly_new) > 5:
            print(f"    ... and {len(truly_new) - 5} more")
        
        # Update tracking
        all_problematic.update(truly_new)
        
        # Update make_def.py
        update_make_def(truly_new, iteration + 4)  # +4 because we already have batches 1-4
    
    print(f"\nReached max iterations ({max_iterations})")
    print(f"Total problematic cells found: {len(all_problematic)}")
    sys.exit(1)

if __name__ == "__main__":
    main()

