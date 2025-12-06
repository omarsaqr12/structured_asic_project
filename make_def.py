#!/usr/bin/env python3
"""
Generate DEF file for OpenROAD routing.
Creates [design_name]_fixed.def with DIEAREA, PINS (+ FIXED), and COMPONENTS (+ FIXED).
"""

import argparse
import os
import json
import yaml
from typing import Dict, List, Tuple, Any, Set
from parse_fabric import parse_fabric_cells, parse_pins, extract_cell_type


def parse_placement_map(map_path: str) -> Dict[str, str]:
    """
    Parse placement.map file to get logical_name -> slot_name mapping.
    
    Format: logical_name slot_name
    """
    placement_map = {}
    
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Placement map not found: {map_path}")
    
    with open(map_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                logical_name = parts[0]
                slot_name = parts[1]
                placement_map[logical_name] = slot_name
    
    return placement_map


def get_io_ports(design_json_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract input and output ports from design JSON file.
    
    Returns:
        (input_ports, output_ports)
    """
    if not os.path.exists(design_json_path):
        raise FileNotFoundError(f"Design JSON not found: {design_json_path}")
    
    with open(design_json_path, 'r') as f:
        design_data = json.load(f)
    
    # Find the top module (usually the first module or one marked as top)
    modules = design_data.get('modules', {})
    top_module = None
    
    for module_name, module_data in modules.items():
        if module_data.get('attributes', {}).get('top') == '00000000000000000000000000000001':
            top_module = module_data
            break
    
    if top_module is None and modules:
        # Fallback: use first module
        top_module = list(modules.values())[0]
    
    if top_module is None:
        return [], []
    
    ports = top_module.get('ports', {})
    input_ports = []
    output_ports = []
    
    for port_name, port_data in ports.items():
        direction = port_data.get('direction', '')
        if direction == 'input':
            input_ports.append(port_name)
        elif direction == 'output':
            output_ports.append(port_name)
    
    return input_ports, output_ports


def get_diearea(pins_db: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Extract DIEAREA from pins database.
    
    Returns:
        (min_x, min_y, max_x, max_y) in microns
    """
    die = pins_db['die']
    width_um = die['width_um']
    height_um = die['height_um']
    
    # Die area: (0, 0) to (width, height)
    return 0.0, 0.0, width_um, height_um


def get_pin_coordinates(pin_name: str, pins_db: Dict[str, Any]) -> Tuple[float, float]:
    """
    Get pin coordinates from pins database.
    
    Returns:
        (x, y) in microns
    """
    for pin in pins_db['pins']:
        if pin['name'] == pin_name:
            return pin['x_um'], pin['y_um']
    
    # Pin not found in pins.yaml - this might happen if design uses different pin names
    # Return a default position (will need to be handled)
    raise ValueError(f"Pin '{pin_name}' not found in pins.yaml")


def build_fabric_slot_lookup(fabric_db: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup dictionary: slot_name -> {x, y, cell_type, orient}
    """
    slot_lookup = {}
    
    for cell_type, slots in fabric_db.items():
        for slot in slots:
            slot_name = slot['name']
            slot_lookup[slot_name] = {
                'x': slot['x'],
                'y': slot['y'],
                'cell_type': cell_type,
                'orient': slot.get('orient', 'N')
            }
    
    return slot_lookup


def generate_def_file(
    design_name: str,
    placement_map: Dict[str, str],
    fabric_db: Dict[str, List[Dict[str, Any]]],
    pins_db: Dict[str, Any],
    input_ports: List[str],
    output_ports: List[str],
    output_path: str
):
    """
    Generate DEF file with DIEAREA, PINS, and COMPONENTS.
    Uses streaming write to handle large component counts.
    """
    # Build slot lookup
    slot_lookup = build_fabric_slot_lookup(fabric_db)
    
    # Get used slot names
    used_slots = set(placement_map.values())
    
    # Get DIEAREA
    min_x, min_y, max_x, max_y = get_diearea(pins_db)
    
    # Count total components first
    num_used = sum(1 for slot_name in placement_map.values() if slot_name in slot_lookup)
    num_unused = sum(1 for slots in fabric_db.values() for slot in slots if slot['name'] not in used_slots)
    total_components = num_used + num_unused
    
    # Open output file with explicit buffering
    with open(output_path, 'w', buffering=1024*1024) as f:  # 1MB buffer
        # Write header
        f.write("VERSION 5.8 ;\n")
        f.write("DIVIDERCHAR \"/\" ;\n")
        f.write("BUSBITCHARS \"[]\" ;\n")
        f.write(f"DESIGN {design_name} ;\n")
        f.write("UNITS DISTANCE MICRONS 1000 ;\n")
        f.write("\n")
        
        # Write DIEAREA
        f.write(f"DIEAREA ( {int(min_x * 1000)} {int(min_y * 1000)} ) ( {int(max_x * 1000)} {int(max_y * 1000)} ) ;\n")
        f.write("\n")
        
        # Write PINS section
        all_ports = input_ports + output_ports
        f.write(f"PINS {len(all_ports)} ;\n")
        
        for port_name in all_ports:
            try:
                x_um, y_um = get_pin_coordinates(port_name, pins_db)
                x_db = int(x_um * 1000)  # Convert to database units
                y_db = int(y_um * 1000)
                
                # Determine direction
                direction = "INPUT" if port_name in input_ports else "OUTPUT"
                
                f.write(f"- {port_name}\n")
                f.write(f"  + DIRECTION {direction}\n")
                f.write("  + USE SIGNAL\n")
                f.write(f"  + FIXED ( {x_db} {y_db} ) N ;\n")
            except ValueError as e:
                # Pin not found in pins.yaml - skip or use default
                print(f"Warning: {e}. Skipping pin {port_name}.")
                continue
        
        f.write("END PINS\n")
        f.write("\n")
        
        # Write COMPONENTS section - stream directly without building list
        f.write(f"COMPONENTS {total_components} ;\n")
        
        component_count = 0
        
        # Write used components (from placement.map)
        for logical_name, slot_name in placement_map.items():
            if slot_name in slot_lookup:
                slot_info = slot_lookup[slot_name]
                x_db = int(slot_info['x'] * 1000)
                y_db = int(slot_info['y'] * 1000)
                orient = slot_info['orient']
                
                f.write(f"- {logical_name} {slot_info['cell_type']}\n")
                f.write(f"  + FIXED ( {x_db} {y_db} ) {orient} ;\n")
                
                component_count += 1
                if component_count % 10000 == 0:
                    f.flush()
        
        # Write unused components (from fabric_cells.yaml, not in placement.map)
        for cell_type, slots in fabric_db.items():
            for slot in slots:
                slot_name = slot['name']
                if slot_name not in used_slots:
                    x_db = int(slot['x'] * 1000)
                    y_db = int(slot['y'] * 1000)
                    orient = slot.get('orient', 'N')
                    
                    f.write(f"- {slot_name} {cell_type}\n")
                    f.write(f"  + FIXED ( {x_db} {y_db} ) {orient} ;\n")
                    
                    component_count += 1
                    if component_count % 10000 == 0:
                        f.flush()
        
        f.write("END COMPONENTS\n")
        f.write("\n")
        f.write("END DESIGN\n")
        f.flush()
    
    print(f"  Written {component_count} components")


def main():
    parser = argparse.ArgumentParser(description='Generate DEF file for OpenROAD routing')
    parser.add_argument('--design', required=True, help='Design name (e.g., 6502)')
    parser.add_argument('--build-dir', default='build', help='Build directory (default: build)')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml', 
                       help='Path to fabric_cells.yaml (default: fabric/fabric_cells.yaml)')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                       help='Path to pins.yaml (default: fabric/pins.yaml)')
    parser.add_argument('--designs-dir', default='designs',
                       help='Designs directory (default: designs)')
    
    args = parser.parse_args()
    
    # Build paths
    placement_map_path = os.path.join(args.build_dir, args.design, f"{args.design}.map")
    design_json_path = os.path.join(args.designs_dir, f"{args.design}_mapped.json")
    output_path = os.path.join(args.build_dir, args.design, f"{args.design}_fixed.def")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating DEF file for design: {args.design}")
    print(f"  Placement map: {placement_map_path}")
    print(f"  Design JSON: {design_json_path}")
    print(f"  Output: {output_path}")
    
    # Parse input files
    print("Parsing placement map...")
    placement_map = parse_placement_map(placement_map_path)
    print(f"  Found {len(placement_map)} placed cells")
    
    print("Parsing fabric cells...")
    fabric_db = parse_fabric_cells(args.fabric_cells)
    total_slots = sum(len(slots) for slots in fabric_db.values())
    print(f"  Found {total_slots} total fabric slots")
    
    print("Parsing pins...")
    pins_db = parse_pins(args.pins)
    print(f"  Die: {pins_db['die']['width_um']} x {pins_db['die']['height_um']} um")
    print(f"  Found {len(pins_db['pins'])} pins in pins.yaml")
    
    print("Extracting I/O ports...")
    input_ports, output_ports = get_io_ports(design_json_path)
    print(f"  Input ports: {len(input_ports)}")
    print(f"  Output ports: {len(output_ports)}")
    
    # Generate DEF file
    print("Generating DEF file...")
    generate_def_file(
        args.design,
        placement_map,
        fabric_db,
        pins_db,
        input_ports,
        output_ports,
        output_path
    )
    
    print(f"✓ DEF file generated: {output_path}")


if __name__ == '__main__':
    main()

