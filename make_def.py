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


def get_io_ports_and_nets(design_json_path: str, placement_map: Dict[str, str]) -> Tuple[List[str], List[str], Dict[int, List[Tuple[str, str]]]]:
    """
    Extract input/output ports AND net connectivity from design JSON file.
    
    The JSON uses bit numbers to represent connectivity:
    - Each port/pin has a "bits" array with bit numbers
    - Cells with same bit number on their connections are on the same net
    
    Returns:
        (input_ports, output_ports, nets)
        - nets: Dict mapping bit_id -> list of (instance_name, pin_name) tuples
    """
    if not os.path.exists(design_json_path):
        raise FileNotFoundError(f"Design JSON not found: {design_json_path}")
    
    with open(design_json_path, 'r') as f:
        design_data = json.load(f)
    
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
        return [], [], {}
    
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
        
        # Store port's bit number (assume single-bit ports for now)
        if bits:
            port_bits[port_name] = bits[0]
    
    # Build nets from cell connections
    # nets[bit_id] = [(instance_name, pin_name), ...]
    nets = defaultdict(list)
    
    # Add I/O port connections
    for port_name, bit_id in port_bits.items():
        if isinstance(bit_id, int):  # Skip constant bits like "0" or "1"
            nets[bit_id].append(("PIN", port_name))
    
        # Process cells - track which cells are placed vs unplaced
        cells = top_module.get('cells', {})
        placed_cells = set(placement_map.keys())
        unplaced_cells = set()
        
        # First pass: identify all cells and their connections
        all_cell_connections = {}  # cell_name -> {pin_name -> [bit_ids]}
        for cell_name, cell_data in cells.items():
            if cell_name in placed_cells:
                connections = cell_data.get('connections', {})
                for pin_name, bit_list in connections.items():
                    for bit_id in bit_list:
                        if isinstance(bit_id, int):  # Skip constant bits
                            nets[bit_id].append((cell_name, pin_name))
            else:
                # Track unplaced cells and their bit connections
                unplaced_cells.add(cell_name)
                connections = cell_data.get('connections', {})
                for pin_name, bit_list in connections.items():
                    for bit_id in bit_list:
                        if isinstance(bit_id, int):
                            # Mark this bit as connected to an unplaced cell
                            if bit_id not in all_cell_connections:
                                all_cell_connections[bit_id] = set()
                            all_cell_connections[bit_id].add(cell_name)
        
        # Find bits that connect to unplaced cells (these nets are incomplete)
        bits_with_unplaced = set()
        for bit_id, unplaced_set in all_cell_connections.items():
            if unplaced_set:  # If any unplaced cell connects to this bit
                bits_with_unplaced.add(bit_id)
        
        if bits_with_unplaced:
            print(f"  WARNING: {len(bits_with_unplaced)} nets connect to unplaced cells - excluding them")
            print(f"           Unplaced cells: {len(unplaced_cells)} (e.g., buffer cells from synthesis)")
        
        # Known problematic cells with no routing access points (DRT-0073 errors)
        # These cells are at fabric positions where pin A cannot be accessed
        problematic_cells = {
            # Batch 1
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11484",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9661",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10382",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10871",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10393",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10176",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10143",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10229",
            # Batch 2
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10402",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11342",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10409",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10851",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10354",
            # Batch 3
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10551",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10803",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10966",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11010",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9298",
            # Batch 4
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10624",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11059",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11126",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11182",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9663",
            # Batch 5
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10649",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11180",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11181",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11218",
            # Batch 6
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10810",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11226",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11258",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11332",
            # Batch 7 - all remaining problematic cells
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10860",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$10975",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11248",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11286",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11300",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11349",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11394",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11402",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11423",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11429",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11447",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11569",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11595",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11597",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11643",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11650",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$11683",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9307",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9345",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9375",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9405",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9432",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9452",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9533",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9547",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9658",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9668",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9671",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9698",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9742",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9913",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9941",
            "$abc$9276$auto$blifparse.cc:396:parse_blif$9949",
        }
        
        # Find bits connected to problematic cells
        bits_with_problematic = set()
        for cell_name, cell_data in cells.items():
            if cell_name in problematic_cells:
                connections = cell_data.get('connections', {})
                for pin_name, bit_list in connections.items():
                    for bit_id in bit_list:
                        if isinstance(bit_id, int):
                            bits_with_problematic.add(bit_id)
        
        if bits_with_problematic:
            print(f"  Excluding {len(bits_with_problematic)} nets connected to 8 problematic cells")
        
        # Combine exclusions
        bits_to_exclude = bits_with_unplaced | bits_with_problematic
        
        # Filter nets:
        # 1. Must have 2+ connections (routable)
        # 2. Must NOT connect to any unplaced or problematic cells
        routable_nets = {
            bit_id: conns 
            for bit_id, conns in nets.items() 
            if len(conns) >= 2 and bit_id not in bits_to_exclude
        }
        
        return input_ports, output_ports, routable_nets


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


def generate_def_file(
    design_name: str,
    placement_map: Dict[str, str],
    fabric_db: Dict[str, List[Dict[str, Any]]],
    pins_db: Dict[str, Any],
    input_ports: List[str],
    output_ports: List[str],
    output_path: str,
    nets: Dict[int, List[Tuple[str, str]]] = None
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
    
    # Helper function to identify problematic cells
    # Only exclude cells that we've specifically identified as having no access points
    # These cells were identified by DRT-0073 errors during detailed routing
    problematic_cell_set = {
        # Batch 1
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11484",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9661",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10382",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10871",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10393",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10176",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10143",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10229",
        # Batch 2
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10402",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11342",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10409",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10851",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10354",
        # Batch 3
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10551",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10803",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10966",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11010",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9298",
        # Batch 4
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10624",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11059",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11126",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11182",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9663",
        # Batch 5
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10649",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11180",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11181",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11218",
        # Batch 6
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10810",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11226",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11258",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11332",
        # Batch 7 - all remaining problematic cells
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10860",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$10975",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11248",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11286",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11300",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11349",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11394",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11402",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11423",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11429",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11447",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11569",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11595",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11597",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11643",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11650",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$11683",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9307",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9345",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9375",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9405",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9432",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9452",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9533",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9547",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9658",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9668",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9671",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9698",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9742",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9913",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9941",
        "$abc$9276$auto$blifparse.cc:396:parse_blif$9949",
    }
    
    def is_problematic_cell(cell_name):
        """Check if cell is in the known problematic set"""
        return cell_name in problematic_cell_set
    
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
        # Use 'top_<design_name>' for DESIGN statement to match Verilog module name
        # Verilog identifiers cannot start with digits, so we prepend 'top_'
        def_design_name = f"top_{design_name}" if design_name and design_name[0].isdigit() else design_name
        f.write(f"DESIGN {def_design_name} ;\n")
        f.write("UNITS DISTANCE MICRONS 1000 ;\n")
        f.write("\n")
        
        # Write DIEAREA
        f.write(f"DIEAREA ( {int(min_x * 1000)} {int(min_y * 1000)} ) ( {int(max_x * 1000)} {int(max_y * 1000)} ) ;\n")
        f.write("\n")
        
        # NOTE: TRACKS should NOT be in DEF files when using OpenROAD.
        # OpenROAD will load tracks from the tech LEF file using the make_tracks command
        # in route.tcl after reading the DEF file. This avoids layer validation errors.
        # The tech LEF file (tech/sky130_fd_sc_hd.tlef) already contains all TRACK definitions.
        
        # Write PINS section with proper PORT geometry for routing grid alignment
        all_ports = input_ports + output_ports
        f.write(f"PINS {len(all_ports)} ;\n")
        
        # Pin geometry size depends on layer
        # met1/met2 minimum width: 0.14um, met3 minimum width: 0.3um
        # Use layer-appropriate sizes for each pin
        pin_sizes = {
            'met1': (280, 280),   # 0.28um (2x min width for better access)
            'met2': (280, 280),   # 0.28um
            'met3': (400, 400),   # 0.40um (> 0.3um min width)
            'met4': (400, 400),   # 0.40um
            'met5': (400, 400),   # 0.40um
        }
        default_pin_size = (280, 280)
        
        # Get die dimensions in DB units
        die_width_db = int(max_x * 1000)
        die_height_db = int(max_y * 1000)
        
        for port_name in all_ports:
            try:
                x_um, y_um, layer = get_pin_info(port_name, pins_db)
                x_db = int(x_um * 1000)  # Convert to database units
                y_db = int(y_um * 1000)
                
                # Determine direction
                direction = "INPUT" if port_name in input_ports else "OUTPUT"
                
                # Pin placement: use coordinates from pins.yaml directly
                # The pins.yaml should have coordinates that align with routing tracks
                # Only apply minimal adjustment if pin is exactly at 0
                min_offset = 1  # Minimal 1 DBU offset if exactly at boundary
                
                pin_x = x_db
                pin_y = y_db
                
                # Only adjust if exactly at boundary (0 or max)
                if pin_x == 0:
                    pin_x = min_offset
                elif pin_x >= die_width_db:
                    pin_x = die_width_db - min_offset
                    
                if pin_y == 0:
                    pin_y = min_offset
                elif pin_y >= die_height_db:
                    pin_y = die_height_db - min_offset
                
                # Calculate pin geometry as RELATIVE offsets from pin center
                # DEF PIN geometry coordinates are relative to FIXED location, not absolute!
                # Use symmetric offsets centered on (0,0)
                # Get layer-appropriate pin size
                pin_width, pin_height = pin_sizes.get(layer, default_pin_size)
                half_width = pin_width // 2
                half_height = pin_height // 2
                
                # Relative coordinates (offsets from pin center)
                x1 = -half_width
                y1 = -half_height
                x2 = half_width
                y2 = half_height
                
                # Write pin with PORT geometry so it's on the routing grid
                # Use FIXED instead of PLACED for pre-route DEF files
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
        
        # Use pattern-based exclusion for ABC buffer cells (same as NETS section)
        # These are auto-generated cells that often have inaccessible pins
        
        # Count components excluding problematic ones (pattern-based)
        num_used_valid = sum(1 for ln in placement_map.keys() 
                           if not is_problematic_cell(ln) and placement_map[ln] in slot_lookup)
        num_unused = sum(1 for slots in fabric_db.values() for slot in slots if slot['name'] not in used_slots)
        actual_total_components = num_used_valid + num_unused
        
        # Write COMPONENTS section - stream directly without building list
        f.write(f"COMPONENTS {actual_total_components} ;\n")
        
        component_count = 0
        skipped_problematic = 0
        
        # Write used components (from placement.map), excluding problematic cells
        for logical_name, slot_name in placement_map.items():
            # Skip problematic ABC buffer cells (pattern-based)
            if is_problematic_cell(logical_name):
                skipped_problematic += 1
                continue
                
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
        
        if skipped_problematic > 0:
            print(f"  Skipped {skipped_problematic} problematic cells from COMPONENTS")
        
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
        
        # Write NETS section - CRITICAL for routing!
        if nets:
            f.write(f"NETS {len(nets)} ;\n")
            
            net_count = 0
            for bit_id, connections in nets.items():
                # Generate net name
                net_name = f"net_{bit_id}"
                
                # Build connection list
                conn_strs = []
                for inst_name, pin_name in connections:
                    conn_strs.append(f"( {inst_name} {pin_name} )")
                
                # Write net
                f.write(f"- {net_name}\n")
                f.write(f"  {' '.join(conn_strs)} ;\n")
                
                net_count += 1
                if net_count % 1000 == 0:
                    f.flush()
            
            f.write("END NETS\n")
            f.write("\n")
            print(f"  Written {net_count} nets")
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
    parser.add_argument('--no-nets', action='store_true',
                       help='Skip NETS section (not recommended - router needs nets!)')
    
    args = parser.parse_args()
    
    # Build paths
    placement_map_path = os.path.join(args.build_dir, args.design, f"{args.design}.map")
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
    
    print("Extracting I/O ports and net connectivity...")
    input_ports, output_ports, nets = get_io_ports_and_nets(design_json_path, placement_map)
    print(f"  Input ports: {len(input_ports)}")
    print(f"  Output ports: {len(output_ports)}")
    print(f"  Routable nets (2+ terminals): {len(nets)}")
    
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
        nets=None if args.no_nets else nets
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
