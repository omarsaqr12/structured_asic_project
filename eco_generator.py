#!/usr/bin/env python3
"""
ECO Generator: Tie unused logic cells to conb_1 LO output

After placement + CTS, identifies all unused logic cells in the fabric
and ties their inputs to a conb_1 tie-low cell's LO output.

Integrates with CTS API to properly account for CTS buffers when identifying unused cells.
"""

import json
import argparse
import sys
import os
import glob
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from parse_fabric import parse_fabric_cells
from parse_design import parse_design
from cts_api import generate_cts_tree
from buffer_manager import parse_placement_map


# Logic cell types that need to be tied low if unused
# Maps cell type to list of input pin names
LOGIC_CELL_TYPES = {
    'sky130_fd_sc_hd__nand2_2': ['A', 'B'],      # NAND2: inputs A, B
    'sky130_fd_sc_hd__or2_2': ['A', 'B'],        # OR2: inputs A, B
    'sky130_fd_sc_hd__and2_2': ['A', 'B'],       # AND2: inputs A, B
    'sky130_fd_sc_hd__clkinv_2': ['A'],          # Inverter: input A
    'sky130_fd_sc_hd__clkbuf_4': ['A'],          # Buffer: input A
}

# Non-logic cell types that should NOT be tied (infrastructure cells)
NON_LOGIC_CELL_TYPES = {
    'sky130_fd_sc_hd__conb_1',           # Tie-high/tie-low cell
    'sky130_fd_sc_hd__tapvpwrvgnd_1',    # Tap cell
    'sky130_fd_sc_hd__decap_3',          # Decap cell
    'sky130_fd_sc_hd__decap_4',          # Decap cell
    'sky130_fd_sc_hd__dfbbp_1',          # DFF cell (sequential, not combinational logic)
    'sky130_fd_sc_hd__fill_1',           # Fill cell
}

CONB_CELL_TYPE = 'sky130_fd_sc_hd__conb_1'
CONB_LO_PIN = 'LO'


def get_used_slots(placement_path: str, 
                   placement_map_path: Optional[str] = None,
                   cts_placement_map: Optional[Dict[str, str]] = None,
                   design_path: Optional[str] = None) -> Set[str]:
    """
    Get all used fabric slots from placement JSON and placement map.
    Only includes slots whose logical cell names are actually in the design netlist.
    
    Args:
        placement_path: Path to placement JSON file
        placement_map_path: Optional path to placement.map file (for CTS buffers)
        cts_placement_map: Optional CTS-generated placement map dict (includes CTS buffers)
        design_path: Optional path to design JSON file to verify cells exist in netlist
    
    Returns:
        Set of used fabric slot names (only for cells that exist in design netlist)
    """
    used_slots = set()
    
    # Get set of logical cell names that actually exist in the design netlist
    # This ensures we only mark slots as "used" if the cell is really in the design
    design_cell_names = None  # Start as None - will be set to a set() if we successfully load design
    if design_path and os.path.exists(design_path):
        try:
            with open(design_path, 'r') as f:
                design_data = json.load(f)
            
            # Find top module
            modules = design_data.get('modules', {})
            top_module = None
            for mod_name, mod_data in modules.items():
                if mod_data.get('attributes', {}).get('top') == '00000000000000000000000000000001':
                    top_module = mod_data
                    break
            
            if not top_module:
                # Fallback: use first module with cells
                for mod_name, mod_data in modules.items():
                    if 'cells' in mod_data and len(mod_data['cells']) > 0:
                        top_module = mod_data
                        break
            
            if top_module:
                cells = top_module.get('cells', {})
                design_cell_names = set(cells.keys())
                print(f"  Loaded design netlist: {len(design_cell_names)} cells")
            else:
                print(f"  WARNING: Could not find top module in design")
                design_cell_names = None  # If we can't find top module, don't filter
        except Exception as e:
            print(f"  WARNING: Could not load design to verify cells: {e}")
            design_cell_names = None  # If we can't load, don't filter
    
    # Load placement JSON
    with open(placement_path, 'r') as f:
        placement = json.load(f)
    
    # Extract used slots from placement JSON
    # Only include if the cell is in the design netlist (if we loaded it)
    filtered_from_placement_json = 0
    for cell_name, pos_data in placement.items():
        if 'fabric_slot_name' in pos_data:
            slot_name = pos_data['fabric_slot_name']
            # If we have design cell names, only include if cell is in design
            if design_cell_names is None:
                # Can't verify - include all (conservative approach)
                used_slots.add(slot_name)
            elif cell_name in design_cell_names:
                # Cell exists in design - mark slot as used
                used_slots.add(slot_name)
            else:
                # Cell is in placement JSON but NOT in design netlist - don't mark as used
                filtered_from_placement_json += 1
    
    if filtered_from_placement_json > 0:
        print(f"  Filtered out {filtered_from_placement_json} slots from placement JSON because their logical cells are not in the design netlist")
    
    # Use CTS placement map if provided (includes CTS buffers)
    if cts_placement_map:
        # Add all slots from CTS placement map
        # Only include if the logical cell name is in the design netlist
        for logical_name, slot_name in cts_placement_map.items():
            # CTS buffers and special cells should always be included
            # Only filter out eco_unused_ cells that aren't in the design
            if design_cell_names is None or logical_name in design_cell_names or logical_name.startswith('cts_buffer_'):
                used_slots.add(slot_name)
    elif placement_map_path:
        # Fallback: parse placement map file
        try:
            placement_map = parse_placement_map(placement_map_path)
            for logical_name, slot_name in placement_map.items():
                # Only include if the logical cell name is in the design netlist
                # Special cells (cts_buffer_, tie_low_cell) should always be included
                if design_cell_names is None or logical_name in design_cell_names or logical_name.startswith('cts_buffer_') or logical_name.startswith('tie_low_cell'):
                    used_slots.add(slot_name)
        except FileNotFoundError:
            pass  # Placement map might not exist
    
    return used_slots


def find_unused_logic_cells(fabric_db: Dict[str, List[Dict[str, Any]]],
                            used_slots: Set[str]) -> List[Dict[str, Any]]:
    """
    Find all unused logic cells in the fabric.
    
    This function iterates through ALL cell types in the fabric database and identifies
    unused logic cells (excluding non-logic cells like CONB, TAP, DECAP, DFBBP, FILL).
    
    Args:
        fabric_db: Fabric database from parse_fabric_cells
        used_slots: Set of used fabric slot names
    
    Returns:
        List of unused logic cell dictionaries with 'name', 'type', 'x', 'y', 'orient'
    """
    unused_cells = []
    
    # Iterate through ALL cell types in the fabric database
    for cell_type, slots in fabric_db.items():
        # Skip non-logic cell types (infrastructure cells)
        if cell_type in NON_LOGIC_CELL_TYPES:
            continue
        
        # Check all slots of this cell type
        for slot in slots:
            if slot['name'] not in used_slots:
                unused_cells.append({
                    'name': slot['name'],
                    'type': cell_type,
                    'x': slot['x'],
                    'y': slot['y'],
                    'orient': slot.get('orient', 'N')
                })
    
    return unused_cells


def count_used_logic_cells(fabric_db: Dict[str, List[Dict[str, Any]]],
                           used_slots: Set[str]) -> int:
    """
    Count all used logic cells in the fabric.
    
    Args:
        fabric_db: Fabric database from parse_fabric_cells
        used_slots: Set of used fabric slot names
    
    Returns:
        Number of used logic cells
    """
    used_count = 0
    
    # Iterate through ALL cell types in the fabric database
    for cell_type, slots in fabric_db.items():
        # Skip non-logic cell types (infrastructure cells)
        if cell_type in NON_LOGIC_CELL_TYPES:
            continue
        
        # Count used slots of this cell type
        for slot in slots:
            if slot['name'] in used_slots:
                used_count += 1
    
    return used_count


def count_total_fabric_cells(fabric_db: Dict[str, List[Dict[str, Any]]]) -> int:
    """
    Count total number of cells in the fabric (all types).
    
    Args:
        fabric_db: Fabric database from parse_fabric_cells
    
    Returns:
        Total number of fabric cells
    """
    total = 0
    for cell_type, slots in fabric_db.items():
        total += len(slots)
    return total


def count_unused_non_logic_cells(fabric_db: Dict[str, List[Dict[str, Any]]],
                                 used_slots: Set[str]) -> int:
    """
    Count unused cells that are NOT logic cells (e.g., DFFs, decaps, taps, etc.).
    
    Args:
        fabric_db: Fabric database from parse_fabric_cells
        used_slots: Set of used fabric slot names
    
    Returns:
        Number of unused non-logic cells
    """
    unused_count = 0
    
    for cell_type, slots in fabric_db.items():
        # Skip logic cell types (they're handled separately)
        if cell_type in LOGIC_CELL_TYPES:
            continue
        
        for slot in slots:
            if slot['name'] not in used_slots:
                unused_count += 1
    
    return unused_count


def find_available_conb(fabric_db: Dict[str, List[Dict[str, Any]]],
                        used_slots: Set[str]) -> Optional[Dict[str, Any]]:
    """
    Find one available conb_1 cell in the fabric.
    
    Args:
        fabric_db: Fabric database from parse_fabric_cells
        used_slots: Set of used fabric slot names
    
    Returns:
        First available conb_1 slot dict or None if none available
    """
    if CONB_CELL_TYPE not in fabric_db:
        return None
    
    for slot in fabric_db[CONB_CELL_TYPE]:
        if slot['name'] not in used_slots:
            return {
                'name': slot['name'],
                'type': CONB_CELL_TYPE,
                'x': slot['x'],
                'y': slot['y'],
                'orient': slot.get('orient', 'N')
            }
    
    return None


def get_max_net_id(design_data: Dict[str, Any]) -> int:
    """
    Find the maximum net ID in the design to assign new net IDs.
    
    Args:
        design_data: Full design JSON data
    
    Returns:
        Maximum net ID found in the design
    """
    max_net_id = 0
    
    modules = design_data.get('modules', {})
    for mod_name, mod_data in modules.items():
        cells = mod_data.get('cells', {})
        for cell_name, cell_data in cells.items():
            connections = cell_data.get('connections', {})
            for port_name, net_ids in connections.items():
                if isinstance(net_ids, list):
                    for net_id in net_ids:
                        if isinstance(net_id, int) and net_id > max_net_id:
                            max_net_id = net_id
    
    return max_net_id


def find_top_module(design_data: Dict[str, Any]) -> Optional[str]:
    """
    Find the top module in the design.
    
    Args:
        design_data: Full design JSON data
    
    Returns:
        Top module name or None
    """
    modules = design_data.get('modules', {})
    
    # First try to find by top attribute
    for mod_name, mod_data in modules.items():
        if mod_data.get('attributes', {}).get('top') == '00000000000000000000000000000001':
            return mod_name
    
    # Fallback: use first module with cells
    for mod_name, mod_data in modules.items():
        if 'cells' in mod_data and len(mod_data['cells']) > 0:
            return mod_name
    
    return None


def extract_design_name(design_path: str) -> str:
    """
    Extract design name from design file path.
    
    Examples:
        designs/6502_mapped.json -> 6502
        designs/aes_128_mapped.json -> aes_128
        designs/arith_mapped.json -> arith
    
    Args:
        design_path: Path to design mapped JSON file
    
    Returns:
        Design name (without _mapped.json suffix)
    """
    basename = os.path.basename(design_path)
    # Remove _mapped.json or .json suffix
    if basename.endswith('_mapped.json'):
        return basename[:-12]  # Remove '_mapped.json'
    elif basename.endswith('.json'):
        return basename[:-5]  # Remove '.json'
    else:
        return basename


def find_best_directory(build_dir: str, design_name: str) -> Optional[str]:
    """
    Find the Best* directory in build/{design_name}/.
    Matches any directory that starts with "Best" regardless of what comes after.
    
    Examples:
        build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/ -> found
        build/arith/Best_greedy/ -> found
        build/aes_128/Best_any_name_here/ -> found
    
    Args:
        build_dir: Build directory (e.g., 'build')
        design_name: Design name (e.g., '6502')
    
    Returns:
        Path to Best* directory if found, None otherwise.
        If multiple Best* directories exist, returns the first one found.
    """
    design_build_dir = os.path.join(build_dir, design_name)
    if not os.path.exists(design_build_dir):
        return None
    
    # Find all directories starting with "Best" (Best* matches any suffix)
    best_dirs = [d for d in glob.glob(os.path.join(design_build_dir, "Best*")) 
                 if os.path.isdir(d)]
    
    if not best_dirs:
        return None
    
    # If multiple Best* directories exist, return the first one
    # (You could sort by modification time to get the most recent)
    return best_dirs[0]


def generate_eco(placement_path: str,
                design_path: str,
                fabric_cells_path: str,
                placement_map_path: Optional[str] = None,
                output_json_path: Optional[str] = None,
                output_verilog_path: Optional[str] = None,
                enable_cts: bool = False,
                cts_tree_type: str = 'h',
                cts_threshold: int = 4) -> Dict[str, Any]:
    """
    Generate ECO to tie unused logic cells to conb_1 LO output.
    
    Args:
        placement_path: Path to placement JSON file (after placement)
        design_path: Path to design JSON file
        fabric_cells_path: Path to fabric_cells.yaml
        placement_map_path: Path to placement.map file (required for CTS)
        output_json_path: Optional path to save ECO JSON
        output_verilog_path: Optional path to save final Verilog
        enable_cts: If True, generate CTS tree using CTS API (default: False)
        cts_tree_type: CTS tree type: 'h' for H-Tree, 'x' for X-Tree (default: 'h')
        cts_threshold: CTS threshold for sinks per leaf node (default: 4)
    
    Returns:
        ECO data structure as dictionary
    """
    print("=" * 60)
    print("ECO Generator: Tie Unused Logic Cells to conb_1 LO")
    print("=" * 60)
    
    # Generate CTS if requested
    cts_tree = None
    cts_placement_map = None
    
    if enable_cts:
        if not placement_map_path:
            raise ValueError("placement_map_path is required when enable_cts=True")
        
        print("\n" + "=" * 60)
        print("Generating CTS tree using CTS API...")
        print("=" * 60)
        
        try:
            cts_tree, cts_placement_map = generate_cts_tree(
                placement_map_path=placement_map_path,
                fabric_cells_path=fabric_cells_path,
                design_path=design_path,
                tree_type=cts_tree_type,
                threshold=cts_threshold
            )
            
            print(f"\n✓ CTS tree generated successfully!")
            print(f"  Tree type: {cts_tree['_metadata']['tree_type']}")
            print(f"  Buffers used: {cts_tree['_metadata']['num_buffers']}")
            print(f"  Sinks: {cts_tree['_metadata']['num_sinks']}")
            print(f"  Updated placement map includes {len(cts_placement_map)} entries")
            
        except Exception as e:
            print(f"\nWARNING: CTS generation failed: {e}")
            print("Continuing without CTS - using original placement map only")
            if placement_map_path:
                cts_placement_map = parse_placement_map(placement_map_path)
    elif placement_map_path:
        # Load existing placement map if CTS not enabled
        print("\nLoading placement map...")
        cts_placement_map = parse_placement_map(placement_map_path)
        print(f"Loaded {len(cts_placement_map)} entries from placement map")
    
    # Load fabric database
    print("\nLoading fabric database...")
    fabric_db = parse_fabric_cells(fabric_cells_path)
    
    # Get used slots (including CTS buffers if CTS was generated)
    # Pass design_path to verify cells exist in netlist
    print("\nIdentifying used fabric slots...")
    print("  (Only marking slots as used if cells exist in design netlist)")
    used_slots = get_used_slots(placement_path, placement_map_path, cts_placement_map, design_path)
    
    # Count how many slots were in placement map but not in design
    if placement_map_path and os.path.exists(placement_map_path):
        try:
            placement_map = parse_placement_map(placement_map_path)
            map_slots = set(placement_map.values())
            slots_not_in_design = map_slots - used_slots
            if slots_not_in_design:
                print(f"  Found {len(slots_not_in_design)} slots in placement map but not in design netlist")
                print(f"  These will be identified as unused and tied to conb_1 LO")
        except:
            pass
    
    print(f"Found {len(used_slots)} used fabric slots (verified against design netlist)")
    
    # Calculate statistics
    total_fabric_cells = count_total_fabric_cells(fabric_db)
    used_cells_count = len(used_slots)
    unused_cells_count = total_fabric_cells - used_cells_count
    
    # Find unused logic cells
    print("\nFinding unused logic cells...")
    unused_logic_cells = find_unused_logic_cells(fabric_db, used_slots)
    used_logic_cells_count = count_used_logic_cells(fabric_db, used_slots)
    unused_non_logic_cells_count = count_unused_non_logic_cells(fabric_db, used_slots)
    print(f"Found {len(unused_logic_cells)} unused logic cells")
    print(f"Found {used_logic_cells_count} used logic cells")
    
    if not unused_logic_cells:
        print("No unused logic cells found. Nothing to do.")
        print("\n" + "=" * 60)
        print("ECO Generation Summary")
        print("=" * 60)
        print(f"Total Fabric cells: {total_fabric_cells}")
        print(f"Used cells (from .map): {used_cells_count}")
        print(f"Unused cells: {unused_cells_count}")
        print(f"Unused logic cells (added to netlist): 0")
        print(f"Unused cells not added to netlist: {unused_cells_count}")
        print("=" * 60)
        return {
            'conb_cell': None,
            'conb_lo_net': None,
            'unused_cells': [],
            'eco_applied': False
        }
    
    # Find available conb_1 cell
    print("\nFinding available conb_1 cell...")
    conb_cell = find_available_conb(fabric_db, used_slots)
    
    if not conb_cell:
        print("ERROR: No available conb_1 cell found!")
        print("Cannot proceed with ECO.")
        return {
            'conb_cell': None,
            'conb_lo_net': None,
            'unused_cells': unused_logic_cells,
            'eco_applied': False
        }
    
    print(f"Found available conb_1: {conb_cell['name']}")
    
    # Load design
    print("\nLoading design netlist...")
    with open(design_path, 'r') as f:
        design_data = json.load(f)
    
    # Find top module
    top_module = find_top_module(design_data)
    if not top_module:
        raise ValueError("Could not find top module in design")
    
    print(f"Top module: {top_module}")
    
    # Get max net ID for new net assignment
    max_net_id = get_max_net_id(design_data)
    conb_lo_net_id = max_net_id + 1
    print(f"Assigned conb_1 LO net ID: {conb_lo_net_id}")
    
    # Get module cells
    modules = design_data.get('modules', {})
    module_data = modules[top_module]
    cells = module_data.get('cells', {})
    
    # Add conb_1 cell to netlist
    conb_logical_name = 'eco_conb_1'
    # Ensure uniqueness
    counter = 0
    while conb_logical_name in cells:
        conb_logical_name = f'eco_conb_1_{counter}'
        counter += 1
    
    cells[conb_logical_name] = {
        'type': CONB_CELL_TYPE,
        'connections': {
            CONB_LO_PIN: [conb_lo_net_id]
            # Note: HI pin is not connected (left floating)
        }
    }
    print(f"Added conb_1 cell: {conb_logical_name}")
    
    # Add unused logic cells to netlist and tie inputs to conb_1 LO
    tied_cells = []
    for unused_cell in unused_logic_cells:
        cell_type = unused_cell['type']
        # Use physical slot name directly as logical name (no prefix)
        # The slot name is already unique and will be renamed to itself in rename.py
        logical_name = unused_cell['name']  # e.g., "T0Y0__R0_NAND_0"
        
        # Ensure uniqueness (shouldn't happen, but safety check)
        original_logical_name = logical_name
        counter = 0
        while logical_name in cells:
            logical_name = f"{original_logical_name}_{counter}"
            counter += 1
        
        # Get input pins for this cell type
        input_pins = LOGIC_CELL_TYPES.get(cell_type, [])
        
        if not input_pins:
            # Try to infer input pins for common logic cell patterns
            # This handles logic cells that might exist in fabric but aren't in LOGIC_CELL_TYPES
            if 'nand' in cell_type.lower() or 'and' in cell_type.lower():
                input_pins = ['A', 'B']  # Most 2-input gates have A, B
            elif 'or' in cell_type.lower() or 'nor' in cell_type.lower():
                input_pins = ['A', 'B']  # Most 2-input gates have A, B
            elif 'inv' in cell_type.lower() or 'buf' in cell_type.lower():
                input_pins = ['A']  # Inverters and buffers typically have A input
            elif 'xor' in cell_type.lower() or 'xnor' in cell_type.lower():
                input_pins = ['A', 'B']  # XOR gates have A, B
            else:
                print(f"WARNING: Unknown logic cell type {cell_type}, cannot determine input pins. Skipping.")
                continue
        
        # Create connections: all inputs tied to conb_1 LO net
        connections = {}
        for pin in input_pins:
            connections[pin] = [conb_lo_net_id]
        
        # Add cell to netlist
        cells[logical_name] = {
            'type': cell_type,
            'connections': connections
        }
        
        tied_cells.append({
            'logical_name': logical_name,
            'fabric_slot': unused_cell['name'],
            'cell_type': cell_type,
            'x': unused_cell['x'],
            'y': unused_cell['y'],
            'orient': unused_cell['orient']
        })
    
    print(f"Added {len(tied_cells)} unused logic cells to netlist")
    print(f"All inputs tied to conb_1 LO net ({conb_lo_net_id})")
    
    # Update placement JSON to mark cells as claimed
    print("\nUpdating placement JSON to mark cells as claimed...")
    with open(placement_path, 'r') as f:
        placement = json.load(f)
    
    # Add conb_1 to placement
    placement[conb_logical_name] = {
        'fabric_slot_name': conb_cell['name'],
        'x': conb_cell['x'],
        'y': conb_cell['y'],
        'orient': conb_cell['orient']
    }
    
    # Add unused logic cells to placement
    for tied_cell in tied_cells:
        placement[tied_cell['logical_name']] = {
            'fabric_slot_name': tied_cell['fabric_slot'],
            'x': tied_cell['x'],
            'y': tied_cell['y'],
            'orient': tied_cell['orient']
        }
    
    # Save updated placement JSON
    placement_output_path = placement_path.replace('.json', '_eco.json')
    print(f"Saving updated placement JSON to: {placement_output_path}")
    with open(placement_output_path, 'w') as f:
        json.dump(placement, f, indent=2)
    print("✓ Placement JSON updated")
    
    # Save updated placement map (including CTS buffers if CTS was generated)
    if cts_placement_map:
        # Extract design name and construct output path: build/{Design_Name}/{Design_Name}_eco.map
        design_name = extract_design_name(design_path)
        build_dir = 'build'
        design_build_dir = os.path.join(build_dir, design_name)
        os.makedirs(design_build_dir, exist_ok=True)
        placement_map_output_path = os.path.join(design_build_dir, f"{design_name}_eco.map")
        
        print(f"\nSaving updated placement map to: {placement_map_output_path}")
        with open(placement_map_output_path, 'w') as f:
            # Write original placement + CTS buffers + ECO cells
            for logical_name in sorted(cts_placement_map.keys()):
                f.write(f"{logical_name} {cts_placement_map[logical_name]}\n")
            
            # Add ECO cells
            f.write(f"{conb_logical_name} {conb_cell['name']}\n")
            for tied_cell in tied_cells:
                f.write(f"{tied_cell['logical_name']} {tied_cell['fabric_slot']}\n")
        
        print("✓ Placement map updated")
    
    # Build ECO data structure
    eco_data = {
        'conb_cell': {
            'logical_name': conb_logical_name,
            'fabric_slot': conb_cell['name'],
            'cell_type': CONB_CELL_TYPE,
            'x': conb_cell['x'],
            'y': conb_cell['y'],
            'orient': conb_cell['orient'],
            'lo_net_id': conb_lo_net_id
        },
        'unused_cells': tied_cells,
        'eco_applied': True,
        'statistics': {
            'total_unused_logic_cells': len(unused_logic_cells),
            'cells_tied_to_conb_lo': len(tied_cells),
            'conb_lo_net_id': conb_lo_net_id
        },
        'placement_updated': True,
        'placement_output': placement_output_path,
        'cts_info': {
            'cts_generated': enable_cts and cts_tree is not None,
            'cts_tree_type': cts_tree['_metadata']['tree_type'] if (enable_cts and cts_tree) else None,
            'cts_buffers': cts_tree['_metadata']['num_buffers'] if (enable_cts and cts_tree) else 0,
            'cts_sinks': cts_tree['_metadata']['num_sinks'] if (enable_cts and cts_tree) else 0
        }
    }
    
    # Save updated design JSON if requested
    if output_json_path:
        print(f"\nSaving updated design JSON to: {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(design_data, f, indent=2)
        print("✓ Design JSON saved")
    
    # Save ECO JSON
    eco_json_path = output_json_path.replace('.json', '_eco.json') if output_json_path else 'eco_data.json'
    print(f"\nSaving ECO data to: {eco_json_path}")
    with open(eco_json_path, 'w') as f:
        json.dump(eco_data, f, indent=2)
    print("✓ ECO JSON saved")
    
    # Write Verilog if requested
    if output_verilog_path:
        print(f"\nWriting Verilog to: {output_verilog_path}")
        write_verilog(design_data, top_module, output_verilog_path)
        print("✓ Verilog saved")
    
    # Calculate final statistics
    unused_logic_added = len(tied_cells)
    unused_logic_not_added = len(unused_logic_cells) - unused_logic_added
    unused_cells_not_added = unused_non_logic_cells_count + unused_logic_not_added
    
    print("\n" + "=" * 60)
    print("ECO Generation Summary")
    print("=" * 60)
    if enable_cts and cts_tree:
        print(f"CTS: {cts_tree['_metadata']['tree_type']} "
              f"({cts_tree['_metadata']['num_buffers']} buffers, "
              f"{cts_tree['_metadata']['num_sinks']} sinks)")
    print(f"conb_1 cell: {conb_logical_name} ({conb_cell['name']})")
    print(f"conb_1 LO net ID: {conb_lo_net_id}")
    print(f"Total Fabric cells: {total_fabric_cells}")
    print(f"Used cells (from .map): {used_cells_count}")
    print(f"Unused cells: {unused_cells_count}")
    print(f"Unused logic cells (added to netlist): {unused_logic_added}")
    print(f"Unused cells not added to netlist: {unused_cells_not_added}")
    print("=" * 60)
    
    return eco_data


def write_verilog(design_data: Dict[str, Any], top_module: str, output_path: str):
    """
    Write design to Verilog format.
    
    Args:
        design_data: Full design JSON data
        design_data: Top module name
        output_path: Path to save Verilog file
    """
    modules = design_data.get('modules', {})
    module_data = modules[top_module]
    
    ports = module_data.get('ports', {})
    cells = module_data.get('cells', {})
    
    with open(output_path, 'w') as f:
        # Write header
        f.write(f"// Generated ECO netlist\n")
        f.write(f"// Top module: {top_module}\n")
        f.write(f"// Includes conb_1 tie-low and unused logic cells\n\n")
        
        # Write module declaration
        f.write(f"module {top_module} (\n")
        
        # Write ports
        port_list = []
        for port_name, port_data in ports.items():
            direction = port_data.get('direction', 'input')
            bits = port_data.get('bits', [])
            if len(bits) == 1:
                port_list.append(f"    {direction} {port_name}")
            else:
                port_list.append(f"    {direction} [{len(bits)-1}:0] {port_name}")
        
        f.write(",\n".join(port_list))
        f.write("\n);\n\n")
        
        # Write wire declarations for nets
        all_nets = set()
        for cell_name, cell_data in cells.items():
            connections = cell_data.get('connections', {})
            for port_name, net_ids in connections.items():
                if isinstance(net_ids, list):
                    for net_id in net_ids:
                        if isinstance(net_id, int):
                            all_nets.add(net_id)
        
        if all_nets:
            f.write("    // Wire declarations\n")
            for net_id in sorted(all_nets):
                f.write(f"    wire n{net_id};\n")
            f.write("\n")
        
        # Write cell instances
        f.write("    // Cell instances\n")
        for cell_name, cell_data in cells.items():
            cell_type = cell_data.get('type', 'unknown')
            connections = cell_data.get('connections', {})
            
            # Format connections
            conn_list = []
            for port_name, net_ids in connections.items():
                if isinstance(net_ids, list) and len(net_ids) == 1:
                    conn_list.append(f".{port_name}(n{net_ids[0]})")
                elif isinstance(net_ids, list) and len(net_ids) > 1:
                    # Multi-bit port
                    net_str = ", ".join([f"n{nid}" for nid in net_ids])
                    conn_list.append(f".{port_name}({{{net_str}}})")
            
            if not conn_list:
                # Skip cells with no connections
                continue
            
            conn_str = ",\n        ".join(conn_list)
            
            f.write(f"    {cell_type} {cell_name} (\n")
            f.write(f"        {conn_str}\n")
            f.write(f"    );\n\n")
        
        f.write("endmodule\n")


def main():
    parser = argparse.ArgumentParser(
        description='ECO Generator: Tie unused logic cells to conb_1 LO output'
    )
    parser.add_argument('--placement', 
                        default=None,
                        help='Path to placement JSON file (after placement, before ECO) '
                            #  '(default: build/{Design_Name}/Best*/{Design_Name}_placement.json)')
                            '(default: build/{Design_Name}/{Design_Name}_placement.json)')

    parser.add_argument('--design', 
                        default='6502',
                        help='Design name (e.g., "6502", "arith"). Path will be auto-constructed as designs/{design}_mapped.json (default: 6502)')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml (default: fabric/fabric_cells.yaml)')
    parser.add_argument('--placement-map', 
                        default=None,
                        help='Path to placement.map file (required for CTS) '
                            #  '(default: build/{Design_Name}/Best*/{Design_Name}.map)')
                            '(default: build/{Design_Name}/{Design_Name}_placement.map)')

    parser.add_argument('--enable-cts', action='store_true', default=True,
                        help='Generate CTS tree using CTS API before ECO (default: True). Use --disable-cts to disable.')
    parser.add_argument('--disable-cts', dest='enable_cts', action='store_false',
                        help='Disable CTS tree generation (overrides --enable-cts)')
    parser.add_argument('--cts-tree-type', choices=['h', 'x'], default='h',
                        help='CTS tree type: h for H-Tree, x for X-Tree (default: h)')
    parser.add_argument('--cts-threshold', type=int, default=4,
                        help='CTS threshold: max sinks per leaf node (default: 4)')
    parser.add_argument('--output-json', default=None,
                        help='Path to save updated design JSON (default: build/{Design_Name}/{Design_Name}_eco.json)')
    parser.add_argument('--output-verilog', default=None,
                        help='Path to save final Verilog (default: build/{Design_Name}/{Design_Name}_final.v)')
    
    args = parser.parse_args()
    
    # Construct design path from design name if it's just a name (not a full path)
    if not args.design.endswith('.json') and not os.path.sep in args.design:
        # It's just a design name, construct the full path
        args.design = os.path.join('designs', f"{args.design}_mapped.json")
    
    # Extract design name from design path
    design_name = extract_design_name(args.design)
    
    # Set default placement path if not provided
    if args.placement is None:
        build_dir = 'build'
        best_dir = find_best_directory(build_dir, design_name)
        if best_dir:
            args.placement = os.path.join(best_dir, f"{design_name}_placement.json")
        else:
            # Fallback to old default if Best* directory not found
            args.placement = f'build/{design_name}/Best_sa_alpha0.99_moves1000_Tfinal0.001/{design_name}_placement.json'
            print(f"Warning: Best* directory not found, using fallback path: {args.placement}")
    
    # Set default placement map path if not provided
    if args.placement_map is None:
        build_dir = 'build'
        best_dir = find_best_directory(build_dir, design_name)
        if best_dir:
            args.placement_map = os.path.join(best_dir, f"{design_name}.map")
        else:
            # Fallback to old default if Best* directory not found
            args.placement_map = f'build/{design_name}/Best_sa_alpha0.99_moves1000_Tfinal0.001/{design_name}.map'
            print(f"Warning: Best* directory not found, using fallback path: {args.placement_map}")
    
    # Set default output paths
    if args.output_json is None:
        # Create build directory if it doesn't exist
        build_dir = 'build'
        design_build_dir = os.path.join(build_dir, design_name)
        os.makedirs(design_build_dir, exist_ok=True)
        args.output_json = os.path.join(design_build_dir, f"{design_name}_eco.json")
    
    if args.output_verilog is None:
        # Create build directory if it doesn't exist
        build_dir = 'build'
        design_build_dir = os.path.join(build_dir, design_name)
        os.makedirs(design_build_dir, exist_ok=True)
        args.output_verilog = os.path.join(design_build_dir, f"{design_name}_final.v")
    
    try:
        eco_data = generate_eco(
            args.placement,
            args.design,
            args.fabric_cells,
            args.placement_map,
            args.output_json,
            args.output_verilog,
            enable_cts=args.enable_cts,
            cts_tree_type=args.cts_tree_type,
            cts_threshold=args.cts_threshold
        )
        
        if not eco_data.get('eco_applied', False):
            print("\nWARNING: ECO was not applied. Check for errors above.")
            sys.exit(1)
        
        print("\n✓ ECO generation completed successfully!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

