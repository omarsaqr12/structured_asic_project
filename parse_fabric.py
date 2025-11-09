#!/usr/bin/env python3
"""
Parser for fabric platform files (fabric_cells.yaml and pins.yaml).
Creates a master fabric_db containing all avaiable physical cell slots.
"""

import yaml
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def extract_cell_type(cell_name: str) -> str:
    """
    Extract cell type from fabric cell name and map to sky130 cell type.
    Examples:
        T0Y0__R0_NAND_0 -> sky130_fd_sc_hd__nand2_2
        T0Y0__R1_INV_0 -> sky130_fd_sc_hd__clkinv_2
        T0Y0__R1_DFBBP_0 -> sky130_fd_sc_hd__dfbbp_1
    """
    # Pattern: T0Y0__R0_NAND_0 -> extract the type part (NAND, INV, etc.)
    match = re.search(r'__R\d+_([A-Z]+)', cell_name)
    if match:
        cell_type_prefix = match.group(1).upper()
        
        # Map fabric naming to full sky130 cell types
        type_mapping = {
            'NAND': 'sky130_fd_sc_hd__nand2_2',
            'INV': 'sky130_fd_sc_hd__clkinv_2',
            'BUF': 'sky130_fd_sc_hd__clkbuf_4',
            'OR': 'sky130_fd_sc_hd__or2_2',
            'DFBBP': 'sky130_fd_sc_hd__dfbbp_1',
            'CONB': 'sky130_fd_sc_hd__conb_1',
            'TAP': 'sky130_fd_sc_hd__tapvpwrvgnd_1',
        }
        
        # Handle DECAP - need to determine if decap_3 or decap_4
        if cell_type_prefix == 'DECAP':
            # Check row number to determine decap type
            row_match = re.search(r'__R(\d+)_', cell_name)
            if row_match:
                row_num = int(row_match.group(1))
                # R2 and R3 use decap_3, others use decap_4
                if row_num in [2, 3]:
                    return 'sky130_fd_sc_hd__decap_3'
            return 'sky130_fd_sc_hd__decap_4'
        
        return type_mapping.get(cell_type_prefix, f'sky130_fd_sc_hd__{cell_type_prefix.lower()}_1')
    
    return 'unknown'


def parse_fabric_cells(fabric_cells_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse fabric_cells.yaml and return a dictionary mapping cell types to lists of slots.
    
    Returns:
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
    """
    with open(fabric_cells_path, 'r') as f:
        data = yaml.safe_load(f)
    
    fabric_db = defaultdict(list)
    
    # Parse the fabric_cells_by_tile structure
    if 'fabric_cells_by_tile' in data:
        tiles = data['fabric_cells_by_tile'].get('tiles', {})
        
        for tile_name, tile_data in tiles.items():
            cells = tile_data.get('cells', [])
            
            for cell in cells:
                cell_name = cell['name']
                cell_type = extract_cell_type(cell_name)
                
                fabric_db[cell_type].append({
                    'name': cell_name,
                    'x': cell['x'],
                    'y': cell['y'],
                    'orient': cell.get('orient', 'N')
                })
    
    return dict(fabric_db)


def parse_pins(pins_path: str) -> Dict[str, Any]:
    """
    Parse pins.yaml and return die/core dimensions and pin information.
    
    Returns:
        pins_db: Dict containing die, core, and pins information
    """
    with open(pins_path, 'r') as f:
        data = yaml.safe_load(f)
    
    pin_placement = data.get('pin_placement', {})
    
    pins_db = {
        'die': {
            'width_um': pin_placement['die']['width_um'],
            'height_um': pin_placement['die']['height_um'],
            'core_margin_um': pin_placement['die']['core_margin_um'],
            'corner_keepout_um': pin_placement['die']['corner_keepout_um']
        },
        'core': {
            'width_um': pin_placement['core']['width_um'],
            'height_um': pin_placement['core']['height_um']
        },
        'pins': []
    }
    
    for pin in pin_placement.get('pins', []):
        pins_db['pins'].append({
            'name': pin['name'],
            'x_um': pin['x_um'],
            'y_um': pin['y_um'],
            'side': pin['side'],
            'layer': pin['layer'],
            'direction': pin['direction'],
            'status': pin['status']
        })
    
    return pins_db


if __name__ == '__main__':
    # Test the parsers
    fabric_db = parse_fabric_cells('fabric/fabric_cells.yaml')
    pins_db = parse_pins('fabric/pins.yaml')
    
    print("Fabric DB Summary:")
    for cell_type, slots in sorted(fabric_db.items()):
        print(f"  {cell_type}: {len(slots)} slots")
    
    print(f"\nDie: {pins_db['die']['width_um']} x {pins_db['die']['height_um']} um")
    print(f"Core: {pins_db['core']['width_um']} x {pins_db['core']['height_um']} um")
    print(f"Pins: {len(pins_db['pins'])} total")

