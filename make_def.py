#!/usr/bin/env python3
"""
Generate DEF file for OpenROAD routing.
Creates [design_name].def with DIEAREA, PINS, COMPONENTS, and NETS.

The NETS section is CRITICAL for routing - it defines which cell pins connect to each other.
"""

import argparse
import os
import json
import yaml
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
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


def get_io_ports_and_nets(design_json_path: str, placement_map: Dict[str, str]) -> Tuple[List[str], List[str], Dict[int, List[Tuple[str, str]]], str]:
    """
    Extract input/output ports AND net connectivity from design JSON file.
    
    The JSON uses bit numbers to represent connectivity:
    - Each port/pin has a "bits" array with bit numbers
    - Cells with same bit number on their connections are on the same net
    
    Note: The instance names in the returned nets are logical names from the JSON.
    These need to be mapped to physical slot names using placement_map before
    writing to DEF file (which is done in generate_def_file).
    
    Returns:
        (input_ports, output_ports, nets, top_module_name)
        - nets: Dict mapping bit_id -> list of (logical_instance_name, pin_name) tuples
        - top_module_name: Name of the top module (for DEF DESIGN statement)
    """
    if not os.path.exists(design_json_path):
        raise FileNotFoundError(f"Design JSON not found: {design_json_path}")
    
    try:
        with open(design_json_path, 'r') as f:
            design_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {design_json_path}: {e}")
    except Exception as e:
        raise IOError(f"Error reading {design_json_path}: {e}")
    
    if not isinstance(design_data, dict):
        raise ValueError(f"Design JSON must be a dictionary, got {type(design_data)}")
    
    # Find the top module (usually the first module or one marked as top)
    modules = design_data.get('modules', {})
    top_module = None
    top_module_name = None
    
    for module_name, module_data in modules.items():
        if module_data.get('attributes', {}).get('top') == '00000000000000000000000000000001':
            top_module = module_data
            top_module_name = module_name
            break
    
    if top_module is None and modules:
        # Fallback: use first non-blackbox module
        for module_name, module_data in modules.items():
            if not module_data.get('attributes', {}).get('blackbox'):
                top_module = module_data
                top_module_name = module_name
                break
    
    if top_module is None:
        return [], [], {}, ""
    
    # Build reverse lookup: original_cell_name -> DEF_instance_name
    # The placement_map has: logical_name -> slot_name
    # We need to map JSON cell names to the logical names used in placement
    
    # Extract ports
    ports = top_module.get('ports', {})
    input_ports = []
    output_ports = []
    port_bits = {}  # port_name -> bit_id
    
    for port_name, port_data in ports.items():
        direction = port_data.get('direction', '')
        bits = port_data.get('bits', [])
        
        if direction == 'input':
            input_ports.append(port_name)
        elif direction == 'output':
            output_ports.append(port_name)
        
        # Store port's bit numbers (handle multi-bit ports)
        # For multi-bit ports, we'll use the first bit for net connectivity
        # In practice, each bit should be treated as a separate net
        if bits and isinstance(bits, list):
            # Use first bit for now (can be extended to handle all bits)
            if len(bits) > 0 and isinstance(bits[0], int):
                port_bits[port_name] = bits[0]
            elif len(bits) > 0:
                # Handle string bits like "0" or "1" (constants)
                try:
                    port_bits[port_name] = int(bits[0])
                except (ValueError, TypeError):
                    pass  # Skip constant bits
    
    # Build nets from cell connections
    # nets[bit_id] = [(instance_name, pin_name), ...]
    nets = defaultdict(list)
    
    # Add I/O port connections
    for port_name, bit_id in port_bits.items():
        if isinstance(bit_id, int):  # Skip constant bits like "0" or "1"
            nets[bit_id].append(("PIN", port_name))
    
    # Process cells - track which cells are placed vs unplaced
    cells = top_module.get('cells', {})
    if not cells:
        print("  WARNING: No cells found in design")
        return input_ports, output_ports, {}, top_module_name if top_module_name else ""
    
    placed_cells = set(placement_map.keys())
    unplaced_cells = set()
    
    # First pass: identify all cells and their connections
    all_cell_connections = {}  # bit_id -> set of unplaced cell names
    for cell_name, cell_data in cells.items():
        connections = cell_data.get('connections', {})
        if not connections:
            continue
            
        # Get cell type for validation
        cell_type = cell_data.get('type', 'UNKNOWN')
        
        # Check if cell is in placement map
        is_placed = cell_name in placed_cells
        
        # If cell is placed, it's valid (regardless of name)
        # Only filter out unplaced invalid cells (Yosys internal or non-Sky130)
        if not is_placed and not is_valid_cell(cell_name, cell_type):
            # Mark as unplaced so their nets are excluded
            unplaced_cells.add(cell_name)
            for pin_name, bit_list in connections.items():
                if not isinstance(bit_list, list):
                    continue
                for bit_id in bit_list:
                    if isinstance(bit_id, int):
                        # Mark this bit as connected to an invalid cell
                        if bit_id not in all_cell_connections:
                            all_cell_connections[bit_id] = set()
                        all_cell_connections[bit_id].add(cell_name)
            continue
        
        if is_placed:
            # Add connections from placed cells to nets
            for pin_name, bit_list in connections.items():
                if not isinstance(bit_list, list):
                    continue
                for bit_id in bit_list:
                    if isinstance(bit_id, int):  # Skip constant bits
                        nets[bit_id].append((cell_name, pin_name))
        else:
            # Track unplaced cells and their bit connections
            unplaced_cells.add(cell_name)
            for pin_name, bit_list in connections.items():
                if not isinstance(bit_list, list):
                    continue
                for bit_id in bit_list:
                    if isinstance(bit_id, int):
                        # Mark this bit as connected to an unplaced cell
                        if bit_id not in all_cell_connections:
                            all_cell_connections[bit_id] = set()
                        all_cell_connections[bit_id].add(cell_name)
    
    # Find bits that connect to unplaced cells (these nets are incomplete)
    bits_with_unplaced = set()
    excluded_net_details = []  # Store details about excluded nets for reporting
    
    for bit_id, unplaced_set in all_cell_connections.items():
        if unplaced_set:  # If any unplaced cell connects to this bit
            bits_with_unplaced.add(bit_id)
            
            # Collect details about this excluded net
            if bit_id in nets:
                net_connections = nets[bit_id]
                placed_conns = [conn for conn in net_connections if conn[0] != "PIN" and conn[0] in placed_cells]
                unplaced_conns = list(unplaced_set)
                
                excluded_net_details.append({
                    'bit_id': bit_id,
                    'total_connections': len(net_connections),
                    'placed_connections': len(placed_conns),
                    'unplaced_cells': list(unplaced_set)[:5],  # First 5 unplaced cells
                    'example_placed': placed_conns[:3] if placed_conns else [],  # First 3 placed connections
                })
    
    if bits_with_unplaced:
        print(f"  WARNING: {len(bits_with_unplaced)} nets connect to unplaced cells - excluding them")
        print(f"           Unplaced cells: {len(unplaced_cells)} (e.g., buffer cells from synthesis)")
        
        # Show detailed information about excluded nets
        if excluded_net_details:
            print(f"\n  Excluded net details (showing first 10):")
            for i, detail in enumerate(excluded_net_details[:10], 1):
                print(f"    Net {detail['bit_id']}:")
                print(f"      Total connections: {detail['total_connections']}")
                print(f"      Placed connections: {detail['placed_connections']}")
                print(f"      Unplaced cells: {len(detail['unplaced_cells'])}")
                if detail['unplaced_cells']:
                    print(f"        Examples: {', '.join(detail['unplaced_cells'][:3])}")
                if detail['example_placed']:
                    print(f"      Placed connections: {', '.join([f'{c[0]}.{c[1]}' for c in detail['example_placed']])}")
                if i < len(excluded_net_details[:10]):
                    print()
        
        bits_to_exclude = bits_with_unplaced
    else:
        bits_to_exclude = set()
    
    # Filter nets:
    # 1. Must have 2+ connections (routable)
    # 2. Must NOT connect to any unplaced or problematic cells
    routable_nets = {
        bit_id: conns 
        for bit_id, conns in nets.items() 
        if len(conns) >= 2 and bit_id not in bits_to_exclude
    }
    
    return input_ports, output_ports, routable_nets, top_module_name if top_module_name else ""


def get_io_ports(design_json_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract input and output ports from design JSON file.
    (Legacy function for backward compatibility)
    
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


def calculate_num_tracks(die_size_um: float, start_um: float, step_um: float) -> int:
    """
    Calculate number of tracks that fit in the die dimension.
    
    Args:
        die_size_um: Die dimension (width or height) in microns
        start_um: Starting offset for tracks in microns
        step_um: Track pitch in microns
    
    Returns:
        Number of tracks
    """
    # Available space after start offset
    available_space = die_size_um - start_um
    if available_space < 0:
        return 0
    
    # Number of tracks = floor(available_space / step) + 1
    num_tracks = int(available_space / step_um) + 1
    return num_tracks


def get_track_info(pins_db: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract track information from pins database.
    
    Returns:
        Dictionary mapping layer name to {start_um, step_um}
    """
    tracks_info = pins_db.get('tracks', {})
    return tracks_info


def get_pin_info(pin_name: str, pins_db: Dict[str, Any]) -> Tuple[float, float, str]:
    """
    Get pin coordinates and layer from pins database.
    
    Returns:
        (x, y, layer) in microns
    """
    for pin in pins_db['pins']:
        if pin['name'] == pin_name:
            return pin['x_um'], pin['y_um'], pin.get('layer', 'met2')
    
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


def is_valid_cell(cell_name: str, cell_type: str, is_placed: bool = False) -> bool:
    """
    Check if cell should be included in DEF.
    
    Filters out:
    - Unplaced Yosys internal cells ($abc$, $auto$) - these don't have LEF pins and cause DRT-0073 errors
    - Cells not in Sky130 library
    
    Note: If a cell is in the placement map (is_placed=True), it's considered valid
    regardless of name, because it was successfully placed and must have a valid LEF definition.
    
    Args:
        cell_name: Logical cell name (e.g., "$abc$9276$auto$blifparse.cc:396:parse_blif$11059")
        cell_type: Sky130 cell type (e.g., "sky130_fd_sc_hd__clkinv_2")
        is_placed: Whether the cell is in the placement map (default: False)
    
    Returns:
        True if cell should be included, False otherwise
    """
    # If cell is placed, it's valid (it was successfully placed, so it must have LEF pins)
    if is_placed:
        # Still check cell type for unplaced cells, but placed cells are always valid
        return True
    
    # Filter out unplaced Yosys internal cells
    # These are generated by Yosys during synthesis and don't have corresponding LEF pins
    # They cause "No access point" errors (DRT-0073) during detailed routing
    if cell_name.startswith('$abc$') or cell_name.startswith('$auto$'):
        return False
    
    # Filter out cells not in Sky130 library
    if not cell_type.startswith('sky130_fd_sc_hd__'):
        return False
    
    return True


def sanitize_identifier(name: str) -> str:
    """
    Sanitize identifier by replacing invalid characters.
    This ensures names match between Verilog and DEF files.
    
    Verilog identifiers can contain letters, digits, underscores, and $.
    However, to ensure compatibility and avoid issues, we replace
    all special characters ($, :, ., etc.) with underscores.
    
    Args:
        name: Identifier name (may contain special characters)
        
    Returns:
        Sanitized identifier with special characters replaced by underscores
    """
    if not name:
        return name
    # Replace all special characters with underscore
    sanitized = name
    # Replace $, :, . with underscore
    sanitized = sanitized.replace('$', '_')
    sanitized = sanitized.replace(':', '_')
    sanitized = sanitized.replace('.', '_')
    # Ensure it doesn't start with a digit (Verilog requirement)
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized


def map_pin_name_to_lef(pin_name: str, cell_type: str) -> str:
    """
    Map JSON pin names to LEF pin names based on cell type.
    
    The JSON file may use different pin names than the LEF file.
    Based on Sky130 LEF files:
    - Buffer cells (clkbuf_4): JSON uses "Y", LEF uses "X" for output
    - Inverter cells (clkinv_2): Both JSON and LEF use "Y" for output
    - NAND/OR gates: Both use "Y" for output
    - DFF cells: Use specific pin names (Q, Q_N, D, CLK, etc.)
    
    Args:
        pin_name: Pin name from JSON file
        cell_type: Sky130 cell type (e.g., 'sky130_fd_sc_hd__clkbuf_4')
    
    Returns:
        LEF pin name
    """
    # Buffer cells: JSON uses "Y", LEF uses "X" for output
    if cell_type == 'sky130_fd_sc_hd__clkbuf_4':
        if pin_name == 'Y':
            return 'X'  # Output pin: JSON "Y" -> LEF "X"
        elif pin_name == 'A':
            return 'A'  # Input pin (same in both)
    
    # Inverter cells: Both JSON and LEF use "Y" for output
    # If JSON has "X", it might be a mistake - keep it as is or map to "Y"
    if cell_type == 'sky130_fd_sc_hd__clkinv_2':
        if pin_name == 'Y':
            return 'Y'  # Output pin (same in both)
        elif pin_name == 'X':
            # JSON might have "X" but LEF uses "Y" - map it
            return 'Y'
        elif pin_name == 'A':
            return 'A'  # Input pin (same in both)
    
    # NAND, OR gates: Both use "Y" for output, "A"/"B" for inputs
    if cell_type in ['sky130_fd_sc_hd__nand2_2', 'sky130_fd_sc_hd__or2_2']:
        # Pin names should match - no mapping needed
        return pin_name
    
    # For other cells, pin names usually match
    # Add more mappings as needed
    return pin_name


def generate_def_file(
    design_name: str,
    placement_map: Dict[str, str],
    fabric_db: Dict[str, List[Dict[str, Any]]],
    pins_db: Dict[str, Any],
    input_ports: List[str],
    output_ports: List[str],
    output_path: str,
    nets: Dict[int, List[Tuple[str, str]]] = None,
    top_module_name: str = None
):
    """
    Generate DEF file with DIEAREA, PINS, COMPONENTS, and NETS.
    Uses streaming write to handle large component counts.
    
    Args:
        nets: Optional dict mapping bit_id -> list of (instance_name, pin_name) tuples
              If None, no NETS section is written (router will have nothing to route!)
    """
    # Build slot lookup
    slot_lookup = build_fabric_slot_lookup(fabric_db)
    
    # Get used slot names
    used_slots = set(placement_map.values())
    
    # Get DIEAREA
    try:
        min_x, min_y, max_x, max_y = get_diearea(pins_db)
        
        # Validate DIEAREA
        if max_x <= min_x or max_y <= min_y:
            raise ValueError(f"Invalid DIEAREA: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        
        if max_x <= 0 or max_y <= 0:
            raise ValueError(f"Invalid DIEAREA dimensions: {max_x} x {max_y}")
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Error getting DIEAREA from pins database: {e}")
    
    # Count total components first
    num_used = sum(1 for slot_name in placement_map.values() if slot_name in slot_lookup)
    num_unused = sum(1 for slots in fabric_db.values() for slot in slots if slot.get('name') not in used_slots)
    total_components = num_used + num_unused
    
    # Open output file with explicit buffering
    with open(output_path, 'w', buffering=1024*1024) as f:  # 1MB buffer
        # Write header
        f.write("VERSION 5.8 ;\n")
        f.write("DIVIDERCHAR \"/\" ;\n")
        f.write("BUSBITCHARS \"[]\" ;\n")
        # Use top_module_name from JSON if available, otherwise fall back to design_name
        # This ensures DEF DESIGN name matches the Verilog module name
        if top_module_name:
            def_design_name = top_module_name
        else:
            # Fallback: Use 'top_<design_name>' for DESIGN statement to match Verilog module name
            # Verilog identifiers cannot start with digits, so we prepend 'top_'
            def_design_name = f"top_{design_name}" if design_name and design_name[0].isdigit() else design_name
        f.write(f"DESIGN {def_design_name} ;\n")
        f.write("UNITS DISTANCE MICRONS 1000 ;\n")
        f.write("\n")
        
        # Write DIEAREA
        f.write(f"DIEAREA ( {round(min_x * 1000)} {round(min_y * 1000)} ) ( {round(max_x * 1000)} {round(max_y * 1000)} ) ;\n")
        f.write("\n")
        
        # NOTE: TRACKS should NOT be in DEF files when using OpenROAD.
        # OpenROAD will load tracks from the tech LEF file using the make_tracks command
        # in route.tcl after reading the DEF file. This avoids layer validation errors.
        # The tech LEF file (tech/sky130_fd_sc_hd.tlef) already contains all TRACK definitions.
        
        # Write PINS section with proper PORT geometry for routing grid alignment
        all_ports = input_ports + output_ports
        f.write(f"PINS {len(all_ports)} ;\n")
        
        # Pin geometry size depends on layer
        # Much larger pins provide ample room for via placement and avoid DRC violations
        # Requirements:
        # - mcon via size: 0.17um × 0.17um
        # - met1 enclosure: 0.06um × 0.03um (or 0.03um × 0.06um)
        # - Minimum pin size for via: 0.17 + 2*0.06 = 0.29um (use much larger for safety)
        # - met1/met2 minimum width: 0.14um
        # - met3 minimum width: 0.3um
        # Using multiplier-based sizes for configurable pin placement
        # Layer pitches (in microns): met1=0.34, met2=0.46, met3=0.68, met4=0.92, met5=3.40
        pin_size_multiplier = 2  # Default multiplier (2x pitch for normal pin placement)
        
        # Layer pitches in microns (used as base dimensions)
        layer_pitches = {
            'met1': 0.34,
            'met2': 0.46,
            'met3': 0.68,
            'met4': 0.92,
            'met5': 3.40,
        }
        
        # Calculate pin sizes based on layer pitch * multiplier (convert to DB units)
        # Manufacturing grid: 5 DB units - ensure pin sizes are grid-aligned
        grid = 5
        pin_sizes = {}
        for layer, pitch_um in layer_pitches.items():
            size_um = pitch_um * pin_size_multiplier
            size_db = round(size_um * 1000)  # Convert microns to database units
            # Snap to manufacturing grid
            size_db = round(size_db / grid) * grid
            pin_sizes[layer] = (size_db, size_db)
        
        # Default to met2 size if layer not specified
        default_pin_size = pin_sizes.get('met2', (920, 920))
        
        # Get die dimensions in DB units
        die_width_db = round(max_x * 1000)
        die_height_db = round(max_y * 1000)
        
        for port_name in all_ports:
            try:
                x_um, y_um, layer = get_pin_info(port_name, pins_db)
                
                # Validate coordinates
                if not isinstance(x_um, (int, float)) or not isinstance(y_um, (int, float)):
                    print(f"Warning: Invalid coordinates for pin {port_name}: ({x_um}, {y_um}). Skipping.")
                    continue
                
                if x_um < 0 or y_um < 0:
                    print(f"Warning: Negative coordinates for pin {port_name}: ({x_um}, {y_um}). Skipping.")
                    continue
                
                x_db = round(x_um * 1000)  # Convert to database units
                y_db = round(y_um * 1000)
                
                # Determine direction
                direction = "INPUT" if port_name in input_ports else "OUTPUT"
                
                # Get layer-appropriate pin size
                pin_width, pin_height = pin_sizes.get(layer, default_pin_size)
                half_width = pin_width // 2
                half_height = pin_height // 2
                
                # Manufacturing grid: 5 DB units (0.005 microns)
                # All coordinates must be multiples of 5
                grid = 5
                
                # Snap pin coordinates to manufacturing grid
                pin_x = round(x_db / grid) * grid
                pin_y = round(y_db / grid) * grid
                
                # Ensure pin geometry doesn't extend outside die area
                # Calculate absolute bounds of pin geometry
                pin_x_min = pin_x - half_width
                pin_x_max = pin_x + half_width
                pin_y_min = pin_y - half_height
                pin_y_max = pin_y + half_height
                
                # Clamp pin position to keep geometry within die bounds
                if pin_x_min < 0:
                    pin_x = half_width
                    pin_x = round(pin_x / grid) * grid  # Re-snap to grid
                elif pin_x_max > die_width_db:
                    pin_x = die_width_db - half_width
                    pin_x = round(pin_x / grid) * grid  # Re-snap to grid
                
                if pin_y_min < 0:
                    pin_y = half_height
                    pin_y = round(pin_y / grid) * grid  # Re-snap to grid
                elif pin_y_max > die_height_db:
                    pin_y = die_height_db - half_height
                    pin_y = round(pin_y / grid) * grid  # Re-snap to grid
                
                # Calculate pin geometry as RELATIVE offsets from pin center
                # DEF PIN geometry coordinates are relative to FIXED location, not absolute!
                # Use symmetric offsets centered on (0,0)
                # Snap geometry coordinates to grid as well
                half_width_snapped = round(half_width / grid) * grid
                half_height_snapped = round(half_height / grid) * grid
                
                # Relative coordinates (offsets from pin center)
                x1 = -half_width_snapped
                y1 = -half_height_snapped
                x2 = half_width_snapped
                y2 = half_height_snapped
                
                # Write pin with PORT geometry so it's on the routing grid
                # Use FIXED status - pins are pre-placed and should not be moved
                # Format: - pin_name + NET net_name + DIRECTION + USE SIGNAL
                #         + PORT + LAYER layer ( x1 y1 ) ( x2 y2 )
                #         + FIXED ( x y ) orientation
                f.write(f"- {port_name} + NET {port_name} + DIRECTION {direction} + USE SIGNAL\n")
                f.write(f"  + PORT\n")
                f.write(f"    + LAYER {layer} ( {x1} {y1} ) ( {x2} {y2} )\n")
                f.write(f"  + FIXED ( {pin_x} {pin_y} ) N ;\n")
            except ValueError as e:
                # Pin not found in pins.yaml - skip or use default
                print(f"Warning: {e}. Skipping pin {port_name}.")
                continue
        
        f.write("END PINS\n")
        f.write("\n")
        
        # Count components
        num_used = sum(1 for ln in placement_map.keys() if placement_map[ln] in slot_lookup)
        num_unused = sum(1 for slots in fabric_db.values() for slot in slots if slot['name'] not in used_slots)
        total_components = num_used + num_unused
        
        # Write COMPONENTS section - stream directly without building list
        f.write(f"COMPONENTS {total_components} ;\n")
        
        component_count = 0
        # Track which cells are actually written to COMPONENTS (for net filtering)
        written_components = set()
        
        # Write used components (from placement.map)
        skipped_invalid = 0
        skipped_yosys = 0
        skipped_non_sky130 = 0
        skipped_examples = []  # Store first few examples for reporting
        
        for logical_name, slot_name in placement_map.items():
            if slot_name in slot_lookup:
                try:
                    slot_info = slot_lookup[slot_name]
                    cell_type = slot_info.get('cell_type', 'UNKNOWN')
                    
                    # If cell is in placement map, it's valid (was successfully placed)
                    # Only filter out non-Sky130 cell types
                    if not cell_type.startswith('sky130_fd_sc_hd__'):
                        skipped_invalid += 1
                        skipped_non_sky130 += 1
                        if len(skipped_examples) < 5:
                            skipped_examples.append(f"Non-Sky130 ({cell_type}): {logical_name[:50]}...")
                        continue
                    
                    # Validate coordinates
                    x = slot_info.get('x', 0)
                    y = slot_info.get('y', 0)
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        print(f"Warning: Invalid coordinates for slot {slot_name}. Skipping component {logical_name}.")
                        continue
                    
                    x_db = round(x * 1000)
                    y_db = round(y * 1000)
                    orient = slot_info.get('orient', 'N')
                    
                    # Validate orientation
                    valid_orientations = ['N', 'S', 'E', 'W', 'FN', 'FS', 'FE', 'FW']
                    if orient not in valid_orientations:
                        print(f"Warning: Invalid orientation '{orient}' for slot {slot_name}. Using 'N'.")
                        orient = 'N'
                    
                    # Use physical slot name in DEF (matches renamed Verilog)
                    # Sanitize slot name to ensure no special characters (matches rename.py)
                    sanitized_slot_name = sanitize_identifier(slot_name)
                    f.write(f"- {sanitized_slot_name} {cell_type}\n")
                    f.write(f"  + FIXED ( {x_db} {y_db} ) {orient} ;\n")
                    
                    # Track that this component was successfully written (using sanitized slot_name)
                    written_components.add(sanitized_slot_name)
                    
                    component_count += 1
                    if component_count % 10000 == 0:
                        f.flush()
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Error processing component {logical_name} in slot {slot_name}: {e}. Skipping.")
                    continue
            else:
                print(f"Warning: Slot {slot_name} not found in fabric database. Skipping component {logical_name}.")
        
        if skipped_invalid > 0:
            print(f"  Skipped {skipped_invalid} invalid cells:")
            if skipped_yosys > 0:
                print(f"    - {skipped_yosys} Yosys internal cells ($abc$, $auto$)")
            if skipped_non_sky130 > 0:
                print(f"    - {skipped_non_sky130} non-Sky130 cells")
            if skipped_examples:
                print(f"    Examples:")
                for example in skipped_examples[:3]:  # Show first 3 examples
                    print(f"      {example}")
        
        # Write unused components (from fabric_cells.yaml, not in placement.map)
        for cell_type, slots in fabric_db.items():
            for slot in slots:
                slot_name = slot.get('name', 'UNKNOWN')
                if slot_name not in used_slots:
                    try:
                        x = slot.get('x', 0)
                        y = slot.get('y', 0)
                        
                        # Validate coordinates
                        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                            print(f"Warning: Invalid coordinates for slot {slot_name}. Skipping.")
                            continue
                        
                        x_db = round(x * 1000)
                        y_db = round(y * 1000)
                        orient = slot.get('orient', 'N')
                        
                        # Validate orientation
                        valid_orientations = ['N', 'S', 'E', 'W', 'FN', 'FS', 'FE', 'FW']
                        if orient not in valid_orientations:
                            orient = 'N'
                        
                        # Sanitize slot name to ensure no special characters (matches rename.py)
                        sanitized_slot_name = sanitize_identifier(slot_name)
                        f.write(f"- {sanitized_slot_name} {cell_type}\n")
                        f.write(f"  + FIXED ( {x_db} {y_db} ) {orient} ;\n")
                        
                        component_count += 1
                        if component_count % 10000 == 0:
                            f.flush()
                    except (KeyError, ValueError, TypeError) as e:
                        print(f"Warning: Error processing unused slot {slot_name}: {e}. Skipping.")
                        continue
        
        f.write("END COMPONENTS\n")
        f.write("\n")
        
        # Write NETS section - CRITICAL for routing!
        if nets:
            # Use the set of cells that were actually written to COMPONENTS
            # This ensures we only reference cells that exist in the DEF file
            valid_cells = written_components
            
            # Build reverse mapping: logical_name -> sanitized_slot_name (for translating JSON cell names)
            # Note: We sanitize slot names to match what rename.py does
            logical_to_slot = {}
            for logical_name, slot_name in placement_map.items():
                sanitized_slot_name = sanitize_identifier(slot_name)
                logical_to_slot[logical_name] = sanitized_slot_name
            
            # Build mapping: sanitized_slot_name -> cell_type (for pin name mapping)
            instance_to_cell_type = {}
            for logical_name, slot_name in placement_map.items():
                sanitized_slot_name = sanitize_identifier(slot_name)
                if slot_name in slot_lookup and sanitized_slot_name in written_components:
                    instance_to_cell_type[sanitized_slot_name] = slot_lookup[slot_name]['cell_type']
            
            # Also build set of valid pin names (from input/output ports)
            valid_pins = set(input_ports + output_ports)
            
            f.write(f"NETS {len(nets)} ;\n")
            
            net_count = 0
            skipped_nets = 0
            invalid_connections = 0
            for bit_id, connections in nets.items():
                try:
                    # Validate connections
                    if not connections or len(connections) < 2:
                        skipped_nets += 1
                        continue
                    
                    # Generate net name (ensure valid DEF identifier)
                    # DEF net names cannot start with digits, so prefix if needed
                    if isinstance(bit_id, int) and bit_id >= 0:
                        net_name = f"net_{bit_id}"
                    else:
                        net_name = f"net_{abs(hash(str(bit_id)))}"
                    
                    # Build connection list - only include valid connections
                    conn_strs = []
                    for inst_name, pin_name in connections:
                        # Validate instance and pin names
                        if not inst_name or not pin_name:
                            continue
                        
                        # Check if this is a PIN connection (always valid)
                        if inst_name == "PIN":
                            if pin_name in valid_pins:
                                inst_name_escaped = "PIN"
                                pin_name_escaped = str(pin_name).replace(' ', '_')
                                conn_strs.append(f"( {inst_name_escaped} {pin_name_escaped} )")
                            else:
                                invalid_connections += 1
                        # Check if this is a cell connection
                        # inst_name from JSON is logical_name, need to map to sanitized_slot_name
                        elif inst_name in logical_to_slot:
                            # Map logical_name (from JSON) to sanitized_slot_name (physical name in DEF)
                            sanitized_slot_name = logical_to_slot[inst_name]
                            
                            # Check if sanitized_slot_name is in valid_cells (was written to COMPONENTS)
                            if sanitized_slot_name in valid_cells:
                                # Map pin name from JSON to LEF pin name based on cell type
                                cell_type = instance_to_cell_type.get(sanitized_slot_name, '')
                                lef_pin_name = map_pin_name_to_lef(pin_name, cell_type)
                                
                                # Escape special characters if needed (DEF format)
                                inst_name_escaped = str(sanitized_slot_name).replace(' ', '_')
                                pin_name_escaped = str(lef_pin_name).replace(' ', '_')
                                conn_strs.append(f"( {inst_name_escaped} {pin_name_escaped} )")
                            else:
                                # Slot name not in COMPONENTS - skip this connection
                                invalid_connections += 1
                                continue
                        else:
                            # Logical name not in placement map - skip this connection
                            invalid_connections += 1
                            continue
                    
                    # Only write net if it has at least 2 valid connections
                    if len(conn_strs) >= 2:
                        f.write(f"- {net_name}\n")
                        f.write(f"  {' '.join(conn_strs)} ;\n")
                        
                        net_count += 1
                        if net_count % 1000 == 0:
                            f.flush()
                    else:
                        skipped_nets += 1
                except Exception as e:
                    print(f"Warning: Error writing net for bit_id {bit_id}: {e}. Skipping.")
                    skipped_nets += 1
                    continue
            
            if invalid_connections > 0:
                print(f"  Filtered out {invalid_connections} invalid connections (cells not in COMPONENTS)")
            
            f.write("END NETS\n")
            f.write("\n")
            print(f"  Written {net_count} nets")
            if skipped_nets > 0:
                print(f"  Skipped {skipped_nets} nets (invalid or incomplete connections)")
        else:
            print("  WARNING: No nets provided - router will have nothing to route!")
        
        f.write("END DESIGN\n")
        f.flush()
    
    print(f"  Written {component_count} components")


def main():
    parser = argparse.ArgumentParser(description='Generate DEF file for OpenROAD routing')
    parser.add_argument('--design', default='6502', help='Design name (e.g., 6502)')
    parser.add_argument('--build-dir', default='build', help='Build directory (default: build)')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml', 
                       help='Path to fabric_cells.yaml (default: fabric/fabric_cells.yaml)')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                       help='Path to pins.yaml (default: fabric/pins.yaml)')
    parser.add_argument('--designs-dir', default='designs',
                       help='Designs directory (default: designs)')
    parser.add_argument('--placement-map', '--map', default=None,
                       help='Path to placement map file (default: build/{design}/{design}_eco.map)')
    parser.add_argument('--no-nets', action='store_true',
                       help='Skip NETS section (not recommended - router needs nets!)')
    
    args = parser.parse_args()
    
    # Build paths
    if args.placement_map:
        placement_map_path = args.placement_map
    else:
        placement_map_path = os.path.join(args.build_dir, args.design, f"{args.design}_eco.map")
    design_json_path = os.path.join(args.designs_dir, f"{args.design}_mapped.json")
    output_path = os.path.join(args.build_dir, args.design, f"{args.design}.def")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating DEF file for design: {args.design}")
    print(f"  Placement map: {placement_map_path}")
    print(f"  Design JSON: {design_json_path}")
    print(f"  Output: {output_path}")
    
    # Parse input files
    print("Parsing placement map...")
    try:
        placement_map = parse_placement_map(placement_map_path)
        print(f"  Found {len(placement_map)} placed cells")
        if len(placement_map) == 0:
            print("  WARNING: No cells in placement map!")
    except Exception as e:
        raise RuntimeError(f"Error parsing placement map: {e}")
    
    print("Parsing fabric cells...")
    try:
        fabric_db = parse_fabric_cells(args.fabric_cells)
        total_slots = sum(len(slots) for slots in fabric_db.values())
        print(f"  Found {total_slots} total fabric slots")
        if total_slots == 0:
            raise ValueError("No fabric slots found in fabric_cells.yaml")
    except Exception as e:
        raise RuntimeError(f"Error parsing fabric cells: {e}")
    
    print("Parsing pins...")
    try:
        pins_db = parse_pins(args.pins)
        if 'die' not in pins_db:
            raise ValueError("Missing 'die' section in pins.yaml")
        if 'pins' not in pins_db:
            raise ValueError("Missing 'pins' section in pins.yaml")
        
        die = pins_db['die']
        if 'width_um' not in die or 'height_um' not in die:
            raise ValueError("Missing width_um or height_um in die section")
        
        print(f"  Die: {die['width_um']} x {die['height_um']} um")
        print(f"  Found {len(pins_db['pins'])} pins in pins.yaml")
    except Exception as e:
        raise RuntimeError(f"Error parsing pins: {e}")
    
    print("Extracting I/O ports and net connectivity...")
    input_ports, output_ports, nets, top_module_name = get_io_ports_and_nets(design_json_path, placement_map)
    print(f"  Input ports: {len(input_ports)}")
    print(f"  Output ports: {len(output_ports)}")
    print(f"  Routable nets (2+ terminals): {len(nets)}")
    if top_module_name:
        print(f"  Top module name: {top_module_name}")
    print("  Note: Yosys internal cells ($abc$, $auto$) are automatically filtered out")
    print("        to prevent DRT-0073 'No access point' errors")
    
    if len(nets) == 0:
        print("  WARNING: No routable nets found! Check that:")
        print("    1. Design JSON has cells with connections")
        print("    2. Cell names in JSON match placement.map logical names")
    
    # Generate DEF file
    print("Generating DEF file...")
    generate_def_file(
        args.design,
        placement_map,
        fabric_db,
        pins_db,
        input_ports,
        output_ports,
        output_path,
        nets=None if args.no_nets else nets,
        top_module_name=top_module_name
    )
    
    print(f"✓ DEF file generated: {output_path}")
    
    if not args.no_nets and len(nets) > 0:
        print(f"\nNet statistics:")
        # Count terminals per net
        term_counts = [len(conns) for conns in nets.values()]
        print(f"  Total nets: {len(nets)}")
        print(f"  Min terminals/net: {min(term_counts)}")
        print(f"  Max terminals/net: {max(term_counts)}")
        print(f"  Avg terminals/net: {sum(term_counts)/len(term_counts):.1f}")


if __name__ == '__main__':
    main()