#!/usr/bin/env python3
"""
Validator: Compares logical_db (required cells) against fabric_db (available slots).
Exits with error code 1 if design cannot be built on the fabric.
Prints fabric utilization report.
"""

import sys
import argparse
from parse_fabric import parse_fabric_cells, parse_pins
from parse_design import parse_design


def validate_design(fabric_cells_path: str, design_path: str) -> bool:
    """
    Validate if a design can be built on the fabric.
    
    Returns:
        True if design is valid, False otherwise
    """
    # Parse fabric and design
    fabric_db = parse_fabric_cells(fabric_cells_path)
    logical_db, _ = parse_design(design_path)
    
    # Count available slots by type
    fabric_slots = {cell_type: len(slots) for cell_type, slots in fabric_db.items()}
    
    # Count required cells by type
    logical_cells = {cell_type: len(instances) for cell_type, instances in logical_db.items()}
    
    # Validate: check if we have enough slots for each cell type
    is_valid = True
    print("=" * 60)
    print("Fabric Utilization Report")
    print("=" * 60)
    
    # Get all unique cell types (union of fabric and logical)
    all_types = set(fabric_slots.keys()) | set(logical_cells.keys())
    
    for cell_type in sorted(all_types):
        available = fabric_slots.get(cell_type, 0)
        required = logical_cells.get(cell_type, 0)
        
        if required > 0:
            if available == 0:
                print(f"ERROR: {cell_type}: {required} required, 0 available")
                is_valid = False
            elif required > available:
                print(f"ERROR: {cell_type}: {required} required, {available} available")
                is_valid = False
            else:
                utilization = (required / available) * 100 if available > 0 else 0
                print(f"{cell_type}: {required}/{available} used ({utilization:.1f}%)")
        elif available > 0:
            # Available but not used
            print(f"{cell_type}: 0/{available} used (0.0%)")
    
    print("=" * 60)
    
    if not is_valid:
        print("\nVALIDATION FAILED: Design cannot be built on this fabric.")
        return False
    else:
        print("\nVALIDATION PASSED: Design can be built on this fabric.")
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate design against fabric')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--design', required=True,
                        help='Path to design mapped JSON file')
    
    args = parser.parse_args()
    
    is_valid = validate_design(args.fabric_cells, args.design)
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()

