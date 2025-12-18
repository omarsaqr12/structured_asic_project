#!/usr/bin/env python3
"""
Convert placement JSON file to .map file format.

Usage:
    python json_to_map.py <input_json> [output_map]
    
Example:
    python json_to_map.py build/6502/6502_placement.json build/6502/6502.map
"""

import sys
import json
import os
from typing import Dict, Any

def json_to_map(json_path: str, map_path: str = None) -> bool:
    """
    Convert placement JSON file to .map file.
    
    Args:
        json_path: Path to input JSON file
        map_path: Path to output .map file (default: same as JSON but with .map extension)
    
    Returns:
        True if successful, False otherwise
    """
    # Read JSON file
    try:
        with open(json_path, 'r') as f:
            placement = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found: {json_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON file: {e}")
        return False
    
    # Determine output path
    if map_path is None:
        # Replace .json with .map, or append .map if no .json extension
        if json_path.endswith('.json'):
            map_path = json_path[:-5] + '.map'
        elif json_path.endswith('_placement.json'):
            map_path = json_path[:-15] + '.map'
        else:
            map_path = json_path + '.map'
    
    # Create output directory if needed
    output_dir = os.path.dirname(map_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Validate placement structure
    if not isinstance(placement, dict):
        print("ERROR: JSON file does not contain a valid placement dictionary")
        return False
    
    # Write .map file
    try:
        with open(map_path, 'w') as f:
            for logical_name in sorted(placement.keys()):
                cell_data = placement[logical_name]
                if isinstance(cell_data, dict):
                    slot_name = cell_data.get('fabric_slot_name', '')
                    if slot_name:
                        f.write(f"{logical_name} {slot_name}\n")
                else:
                    print(f"WARNING: Invalid entry for {logical_name}, skipping")
        
        print(f"✓ Successfully converted JSON to .map file")
        print(f"  Input:  {json_path}")
        print(f"  Output: {map_path}")
        print(f"  Total instances: {len(placement)}")
        return True
        
    except IOError as e:
        print(f"ERROR: Failed to write .map file: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python json_to_map.py <input_json> [output_map]")
        print("\nExample:")
        print("  python json_to_map.py build/6502/6502_placement.json")
        print("  python json_to_map.py build/6502/6502_placement.json build/6502/6502.map")
        sys.exit(1)
    
    json_path = sys.argv[1]
    map_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = json_to_map(json_path, map_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

