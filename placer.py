#!/usr/bin/env python3
"""
Optimized Placer: Maps logical cells to physical fabric slots using a greedy algorithm
to minimize total Half-Perimeter Wirelength (HPWL).

Performance Optimizations:
- Pre-built net-to-cells index for O(1) neighbor lookups
- Incremental score updates (only affected cells)
- KD-tree spatial indexing for fast nearest slot searches
- Efficient I/O cell placement with barycenter of connected pins
"""

import sys
import argparse
import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from validator import validate_design
from parse_fabric import parse_fabric_cells, parse_pins
from parse_design import parse_design
from tqdm import tqdm

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Falling back to linear search for slots.")
    
    
HAS_SCIPY = False


# ============================================================================
# HPWL Calculation Utilities (Issue #1)
# ============================================================================

def extract_nets(netlist_graph: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Extract all unique nets from netlist_graph.
    
    A net is identified by a net ID (integer) that connects multiple cells.
    In Yosys JSON format, connections are stored as lists of net IDs.
    
    Args:
        netlist_graph: Dict mapping instance_name -> {type, connections}
    
    Returns:
        nets_dict: Dict mapping net_id -> list of cell instances connected to this net
    """
    nets_dict = defaultdict(list)
    
    for cell_name, cell_data in netlist_graph.items():
        connections = cell_data.get('connections', {})
        for port_name, net_ids in connections.items():
            # net_ids is a list of integers (net IDs)
            for net_id in net_ids:
                if cell_name not in nets_dict[net_id]:
                    nets_dict[net_id].append(cell_name)
    
    return dict(nets_dict)


def calculate_hpwl(positions: List[Tuple[float, float]]) -> float:
    """
    Calculate Half-Perimeter Wirelength (HPWL) for a single net.
    
    Formula: HPWL = (max_x - min_x) + (max_y - min_y)
    
    Args:
        positions: List of (x, y) coordinates in microns
    
    Returns:
        HPWL in microns (float)
    """
    if len(positions) == 0 or len(positions) == 1:
        return 0.0
    
    if len(positions) == 2:
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        return abs(x2 - x1) + abs(y2 - y1)
    
    # For 3+ cells, find bounding box
    x_coords = [x for x, y in positions]
    y_coords = [y for x, y in positions]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    return (max_x - min_x) + (max_y - min_y)


def calculate_total_hpwl(placement: Dict[str, Dict[str, Any]],
                         nets_dict: Dict[int, List[str]],
                         fabric_db: Dict[str, List[Dict[str, Any]]],
                         pins_db: Optional[Dict[str, Any]] = None,
                         port_to_nets: Optional[Dict[str, List[int]]] = None) -> float:
    """
    Calculate total HPWL for all nets in the design.
    
    Args:
        placement: Dict mapping logical_instance_name -> {fabric_slot_name, x, y, orient}
        nets_dict: Dict mapping net_id -> list of cell instances (from extract_nets)
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        pins_db: Optional pins database for I/O pin positions
        port_to_nets: Optional port-to-net mapping for I/O nets
    
    Returns:
        Total HPWL in microns (float)
    """
    total_hpwl = 0.0
    
    # Build a reverse lookup: slot_name -> slot_dict for fast access
    slot_lookup = {}
    for slots in fabric_db.values():
        for slot in slots:
            slot_lookup[slot['name']] = slot
    
    # Process each net
    for net_id, cell_list in nets_dict.items():
        positions = []
        
        # Get positions of all cells on this net
        for cell_name in cell_list:
            if cell_name in placement:
                cell_placement = placement[cell_name]
                slot_name = cell_placement.get('fabric_slot_name')
                
                if slot_name and slot_name in slot_lookup:
                    slot = slot_lookup[slot_name]
                    positions.append((slot['x'], slot['y']))
        
        # Add I/O pin positions if this net is connected to pins
        if pins_db and port_to_nets:
            for pin in pins_db.get('pins', []):
                if pin.get('status') == 'FIXED':
                    pin_name = pin['name']
                    if pin_name in port_to_nets:
                        pin_nets = port_to_nets[pin_name]
                        if net_id in pin_nets:
                            positions.append((pin['x_um'], pin['y_um']))
        
        # Calculate HPWL for this net
        net_hpwl = calculate_hpwl(positions)
        total_hpwl += net_hpwl
    
    return total_hpwl


# ============================================================================
# End of HPWL Utilities
# ============================================================================


def get_port_to_net_mapping(design_path: str) -> Dict[str, List[int]]:
    """
    Parse design JSON to map port names to net IDs.
    
    Args:
        design_path: Path to design mapped JSON file
    
    Returns:
        port_to_nets: Dict mapping port_name -> list of net IDs
    """
    with open(design_path, 'r') as f:
        data = json.load(f)
    
    port_to_nets = {}
    modules = data.get('modules', {})
    
    # Find the top module
    top_module = None
    for mod_name, mod_data in modules.items():
        if mod_data.get('attributes', {}).get('top') == '00000000000000000000000000000001':
            top_module = mod_name
            break
    
    if not top_module:
        # If no top attribute, use the first module with ports
        for mod_name, mod_data in modules.items():
            if 'ports' in mod_data and len(mod_data['ports']) > 0:
                top_module = mod_name
                break
    
    if top_module:
        module_data = modules[top_module]
        ports = module_data.get('ports', {})
        
        for port_name, port_data in ports.items():
            bits = port_data.get('bits', [])
            port_to_nets[port_name] = bits
    
    return port_to_nets


def build_net_index(netlist_graph: Dict[str, Dict[str, Any]]) -> Dict[int, Set[str]]:
    """
    Build index mapping net_id -> set of cell names connected to it.
    This enables O(1) lookup of cells on a net instead of O(n) scan.
    
    Args:
        netlist_graph: Dict mapping instance_name -> {type, connections}
    
    Returns:
        net_to_cells: Dict mapping net_id -> set of cell names
    """
    net_to_cells = defaultdict(set)
    
    for cell_name, cell_data in netlist_graph.items():
        connections = cell_data.get('connections', {})
        for port_nets in connections.values():
            for net_id in port_nets:
                net_to_cells[net_id].add(cell_name)
    
    return dict(net_to_cells)


def precompute_cell_nets(netlist_graph: Dict[str, Dict[str, Any]]) -> Dict[str, Set[int]]:
    """
    Pre-compute which nets each cell is connected to.
    
    Args:
        netlist_graph: Dict mapping instance_name -> {type, connections}
    
    Returns:
        cell_to_nets: Dict mapping cell_name -> set of net IDs
    """
    cell_to_nets = {}
    
    for cell_name, cell_data in netlist_graph.items():
        cell_nets = set()
        connections = cell_data.get('connections', {})
        for port_nets in connections.values():
            cell_nets.update(port_nets)
        cell_to_nets[cell_name] = cell_nets
    
    return cell_to_nets


def build_slot_spatial_index(fabric_db: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict, Dict]:
    """
    Build KD-trees for fast spatial queries on fabric slots.
    
    Args:
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
    
    Returns:
        kdtrees: Dict mapping cell_type -> KDTree
        slot_lists: Dict mapping cell_type -> List of slots (in same order as KDTree)
    """
    kdtrees = {}
    slot_lists = {}
    
    if not HAS_SCIPY:
        return {}, fabric_db
    
    for cell_type, slots in fabric_db.items():
        if not slots:
            continue
        
        coords = [(slot['x'], slot['y']) for slot in slots]
        kdtrees[cell_type] = KDTree(coords)
        slot_lists[cell_type] = slots
    
    return kdtrees, slot_lists


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_nearest_slot_kdtree(target_x: float,
                              target_y: float,
                              cell_type: str,
                              kdtrees: Dict,
                              slot_lists: Dict,
                              used_slots: Set[str],
                              max_candidates: int = 50) -> Optional[Dict[str, Any]]:
    """
    Find nearest available slot using KD-tree for fast spatial search.
    
    Args:
        target_x: Target X coordinate in microns
        target_y: Target Y coordinate in microns
        cell_type: Type of cell to place
        kdtrees: Dict of KD-trees per cell type
        slot_lists: Dict of slot lists per cell type
        used_slots: Set of already used slot names
        max_candidates: Number of nearest candidates to check
    
    Returns:
        Best available slot dict or None
    """
    if cell_type not in kdtrees or cell_type not in slot_lists:
        return None
    
    kdtree = kdtrees[cell_type]
    slots = slot_lists[cell_type]
    
    # Query k nearest neighbors
    k = min(max_candidates, len(slots))
    distances, indices = kdtree.query([target_x, target_y], k=k)
    
    # Handle single result
    if not isinstance(indices, (list, tuple)):
        indices = [indices]
    
    # Find first available slot
    for idx in indices:
        slot = slots[idx]
        if slot['name'] not in used_slots:
            return slot
    
    return None


def find_nearest_slot_linear(target_x: float,
                              target_y: float,
                              available_slots: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Linear search fallback for finding nearest slot.
    
    Args:
        target_x: Target X coordinate in microns
        target_y: Target Y coordinate in microns
        available_slots: List of available slot dicts
    
    Returns:
        Best slot dict or None
    """
    if not available_slots:
        return None
    
    nearest_slot = None
    min_distance = float('inf')
    
    for slot in available_slots:
        distance = calculate_distance(target_x, target_y, slot['x'], slot['y'])
        if distance < min_distance:
            min_distance = distance
            nearest_slot = slot
    
    return nearest_slot


def place_io_connected_cells_optimized(fabric_db: Dict[str, List[Dict[str, Any]]],
                                        pins_db: Dict[str, Any],
                                        netlist_graph: Dict[str, Dict[str, Any]],
                                        port_to_nets: Dict[str, List[int]],
                                        cell_to_nets: Dict[str, Set[int]],
                                        placement: Dict[str, Dict[str, Any]],
                                        used_slots: Set[str],
                                        kdtrees: Dict,
                                        slot_lists: Dict) -> int:
    """
    Optimized placement of cells connected to fixed I/O pins.
    Places each cell at the barycenter of all its connected I/O pins.
    
    Args:
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        pins_db: Dict containing die, core, and pins information
        netlist_graph: Dict mapping instance_name -> {type, connections}
        port_to_nets: Dict mapping port_name -> list of net IDs
        cell_to_nets: Pre-computed dict mapping cell_name -> set of net IDs
        placement: Current placement dict (will be updated)
        used_slots: Set of already used fabric slot names (will be updated)
        kdtrees: KD-trees for spatial indexing
        slot_lists: Slot lists corresponding to KD-trees
    
    Returns:
        Number of cells placed
    """
    # Step 1: Build mapping of net_id -> pin locations
    net_to_pin_locations = defaultdict(list)
    io_net_ids = set()
    
    for pin in pins_db.get('pins', []):
        if pin.get('status') == 'FIXED':
            pin_name = pin['name']
            if pin_name in port_to_nets:
                net_ids = port_to_nets[pin_name]
                pin_loc = (pin['x_um'], pin['y_um'])
                
                for net_id in net_ids:
                    net_to_pin_locations[net_id].append(pin_loc)
                    io_net_ids.add(net_id)
    
    if not io_net_ids:
        print("No fixed I/O pins found.")
        return 0
    
    # Step 2: Find all unique cells connected to I/O nets
    io_cells = set()
    for cell_name, cell_nets in cell_to_nets.items():
        if cell_nets & io_net_ids:  # Cell has at least one I/O net
            io_cells.add(cell_name)
    
    if not io_cells:
        print("No cells found connected to I/O pins.")
        return 0
    
    # Step 3: Place each I/O cell at barycenter of its connected pins
    cells_placed = 0
    
    for cell_name in tqdm(io_cells, desc="Placing I/O cells"):
        if cell_name in placement:
            continue
        
        cell_type = netlist_graph[cell_name]['type']
        cell_nets = cell_to_nets[cell_name]
        
        # Collect all pin locations connected to this cell
        pin_locations = []
        for net_id in cell_nets:
            if net_id in net_to_pin_locations:
                pin_locations.extend(net_to_pin_locations[net_id])
        
        if not pin_locations:
            continue
        
        # Calculate barycenter of all connected pins
        target_x = sum(x for x, y in pin_locations) / len(pin_locations)
        target_y = sum(y for x, y in pin_locations) / len(pin_locations)
        
        # Find nearest available slot
        if HAS_SCIPY and cell_type in kdtrees:
            best_slot = find_nearest_slot_kdtree(
                target_x, target_y, cell_type, kdtrees, slot_lists, used_slots
            )
        else:
            available_slots = [slot for slot in fabric_db.get(cell_type, [])
                             if slot['name'] not in used_slots]
            best_slot = find_nearest_slot_linear(target_x, target_y, available_slots)
        
        if best_slot:
            placement[cell_name] = {
                'fabric_slot_name': best_slot['name'],
                'x': best_slot['x'],
                'y': best_slot['y'],
                'orient': best_slot['orient']
            }
            used_slots.add(best_slot['name'])
            cells_placed += 1
        else:
            print(f"WARNING: No available slots for cell {cell_name} (type {cell_type})")
    
    return cells_placed


def find_placed_neighbors_fast(cell_name: str,
                                net_index: Dict[int, Set[str]],
                                cell_nets: Set[int],
                                placement: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Fast neighbor lookup using pre-built net index.
    
    Args:
        cell_name: Name of the cell
        net_index: Pre-built dict mapping net_id -> set of cell names
        cell_nets: Set of net IDs this cell is connected to
        placement: Current placement dict
    
    Returns:
        List of (neighbor_name, position_dict) tuples
    """
    neighbors = []
    seen = set()
    
    for net_id in cell_nets:
        for other_cell in net_index.get(net_id, set()):
            if other_cell != cell_name and other_cell in placement and other_cell not in seen:
                seen.add(other_cell)
                neighbors.append((other_cell, placement[other_cell]))
    
    return neighbors


def add_io_pin_neighbors(cell_nets: Set[int],
                          pins_db: Dict[str, Any],
                          port_to_nets: Dict[str, List[int]],
                          neighbors: List[Tuple[str, Dict[str, Any]]]) -> None:
    """
    Add I/O pins as neighbors if they share nets with the cell.
    Modifies neighbors list in place.
    
    Args:
        cell_nets: Set of net IDs the cell is connected to
        pins_db: Pins database
        port_to_nets: Port to net mapping
        neighbors: List to append pin neighbors to
    """
    for pin in pins_db.get('pins', []):
        if pin.get('status') == 'FIXED':
            pin_name = pin['name']
            if pin_name in port_to_nets:
                pin_nets = set(port_to_nets[pin_name])
                if cell_nets & pin_nets:
                    neighbors.append((
                        f"pin_{pin_name}",
                        {'x': pin['x_um'], 'y': pin['y_um']}
                    ))


def calculate_barycenter(neighbors: List[Tuple[str, Dict[str, Any]]]) -> Tuple[float, float]:
    """
    Calculate barycenter (center of gravity) of neighbors.
    
    Args:
        neighbors: List of (name, position_dict) tuples
    
    Returns:
        (barycenter_x, barycenter_y) in microns
    """
    if not neighbors:
        return (0.0, 0.0)
    
    total_x = sum(pos['x'] for _, pos in neighbors)
    total_y = sum(pos['y'] for _, pos in neighbors)
    count = len(neighbors)
    
    return (total_x / count, total_y / count)


def get_fallback_position(pins_db: Optional[Dict[str, Any]] = None,
                          fabric_db: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Tuple[float, float]:
    """
    Get a fallback position when a cell has no placed neighbors.
    
    Args:
        pins_db: Optional pins database
        fabric_db: Optional fabric database
    
    Returns:
        (x, y) position in microns
    """
    if fabric_db:
        all_x = []
        all_y = []
        for slots in fabric_db.values():
            for slot in slots:
                all_x.append(slot['x'])
                all_y.append(slot['y'])
        
        if all_x and all_y:
            center_x = (min(all_x) + max(all_x)) / 2
            center_y = (min(all_y) + max(all_y)) / 2
            return (center_x, center_y)
    
    if pins_db:
        die_width = pins_db.get('die', {}).get('width_um', 0)
        die_height = pins_db.get('die', {}).get('height_um', 0)
        if die_width > 0 and die_height > 0:
            return (die_width / 2, die_height / 2)
    
    return (0.0, 0.0)


def place_greedy_barycenter_optimized(fabric_db: Dict[str, List[Dict[str, Any]]],
                                       netlist_graph: Dict[str, Dict[str, Any]],
                                       placement: Dict[str, Dict[str, Any]],
                                       used_slots: Set[str],
                                       net_index: Dict[int, Set[str]],
                                       cell_to_nets: Dict[str, Set[int]],
                                       kdtrees: Dict,
                                       slot_lists: Dict,
                                       pins_db: Optional[Dict[str, Any]] = None,
                                       port_to_nets: Optional[Dict[str, List[int]]] = None) -> int:
    """
    Optimized greedy barycenter placement with incremental score updates.
    
    Args:
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        netlist_graph: Dict mapping instance_name -> {type, connections}
        placement: Current placement dict (will be updated)
        used_slots: Set of already used fabric slot names (will be updated)
        net_index: Pre-built net-to-cells index
        cell_to_nets: Pre-computed cell-to-nets mapping
        kdtrees: KD-trees for spatial indexing
        slot_lists: Slot lists corresponding to KD-trees
        pins_db: Optional pins database
        port_to_nets: Optional port-to-net mapping
    
    Returns:
        Number of cells placed
    """
    unplaced_cells = {cell for cell in netlist_graph.keys() if cell not in placement}
    
    if not unplaced_cells:
        return 0
    
    # Initialize scores for all unplaced cells
    cell_scores = {}
    
    print("Computing initial cell scores...")
    for cell_name in tqdm(unplaced_cells, desc="Initial scoring"):
        neighbors = find_placed_neighbors_fast(
            cell_name, net_index, cell_to_nets[cell_name], placement
        )
        
        if pins_db and port_to_nets:
            add_io_pin_neighbors(cell_to_nets[cell_name], pins_db, port_to_nets, neighbors)
        
        # Score: (num_placed_neighbors, total_connections)
        cell_scores[cell_name] = (len(neighbors), len(cell_to_nets[cell_name]))
    
    cells_placed = 0
    fallback_pos = get_fallback_position(pins_db, fabric_db)
    
    print("Placing cells with incremental updates...")
    with tqdm(total=len(unplaced_cells), desc="Placing remaining cells") as pbar:
        while unplaced_cells:
            # Find cell with highest score
            best_cell = max(unplaced_cells, key=lambda c: cell_scores[c])
            cell_type = netlist_graph[best_cell]['type']
            
            # Get neighbors and calculate barycenter
            neighbors = find_placed_neighbors_fast(
                best_cell, net_index, cell_to_nets[best_cell], placement
            )
            
            if pins_db and port_to_nets:
                add_io_pin_neighbors(cell_to_nets[best_cell], pins_db, port_to_nets, neighbors)
            
            if neighbors:
                target_x, target_y = calculate_barycenter(neighbors)
            else:
                target_x, target_y = fallback_pos
            
            # Find nearest available slot
            if HAS_SCIPY and cell_type in kdtrees:
                best_slot = find_nearest_slot_kdtree(
                    target_x, target_y, cell_type, kdtrees, slot_lists, used_slots
                )
            else:
                available_slots = [slot for slot in fabric_db.get(cell_type, [])
                                 if slot['name'] not in used_slots]
                best_slot = find_nearest_slot_linear(target_x, target_y, available_slots)
            
            if not best_slot:
                print(f"WARNING: No slots available for cell {best_cell} (type {cell_type})")
                unplaced_cells.remove(best_cell)
                del cell_scores[best_cell]
                pbar.update(1)
                continue
            
            # Place the cell
            placement[best_cell] = {
                'fabric_slot_name': best_slot['name'],
                'x': best_slot['x'],
                'y': best_slot['y'],
                'orient': best_slot['orient']
            }
            used_slots.add(best_slot['name'])
            unplaced_cells.remove(best_cell)
            del cell_scores[best_cell]
            cells_placed += 1
            
            # Incremental update: only update scores for affected cells
            placed_nets = cell_to_nets[best_cell]
            affected_cells = set()
            for net_id in placed_nets:
                affected_cells.update(net_index.get(net_id, set()))
            
            # Update scores only for unplaced affected cells
            for affected_cell in affected_cells:
                if affected_cell in unplaced_cells:
                    neighbors = find_placed_neighbors_fast(
                        affected_cell, net_index, cell_to_nets[affected_cell], placement
                    )
                    
                    if pins_db and port_to_nets:
                        add_io_pin_neighbors(cell_to_nets[affected_cell], pins_db, 
                                           port_to_nets, neighbors)
                    
                    cell_scores[affected_cell] = (len(neighbors), len(cell_to_nets[affected_cell]))
            
            pbar.update(1)
    
    return cells_placed


def place_design(fabric_cells_path: str, 
                 design_path: str, 
                 pins_path: str = 'fabric/pins.yaml') -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Place logical cells onto physical fabric slots using optimized greedy algorithm.
    
    Args:
        fabric_cells_path: Path to fabric_cells.yaml
        design_path: Path to design mapped JSON file
        pins_path: Path to pins.yaml
    
    Returns:
        placement: Dict mapping logical_instance_name -> {fabric_slot_name, x, y, orient}
                  Returns None if validation fails
    """
    # Validate design
    print("Validating design against fabric...")
    if not validate_design(fabric_cells_path, design_path):
        print("\nERROR: Design validation failed. Cannot proceed with placement.")
        return None
    
    print("\nValidation passed. Proceeding with placement...\n")
    
    # Parse fabric and design data
    print("Loading fabric and design data...")
    fabric_db = parse_fabric_cells(fabric_cells_path)
    logical_db, netlist_graph = parse_design(design_path)
    pins_db = parse_pins(pins_path)
    port_to_nets = get_port_to_net_mapping(design_path)
    
    # Build optimization data structures
    print("Building optimization indices...")
    net_index = build_net_index(netlist_graph)
    cell_to_nets = precompute_cell_nets(netlist_graph)
    kdtrees, slot_lists = build_slot_spatial_index(fabric_db)
    
    if HAS_SCIPY:
        print(f"Built KD-trees for {len(kdtrees)} cell types")
    else:
        print("Using linear search (install scipy for faster placement)")
    
    # Initialize placement
    placement = {}
    used_slots = set()
    
    # Place I/O connected cells first
    print("\nPlacing cells connected to I/O pins...")
    io_cells_placed = place_io_connected_cells_optimized(
        fabric_db, pins_db, netlist_graph, port_to_nets, 
        cell_to_nets, placement, used_slots, kdtrees, slot_lists
    )
    print(f"Placed {io_cells_placed} cells connected to I/O pins.\n")
    
    # Place remaining cells
    print("Placing remaining cells using optimized greedy barycenter algorithm...")
    remaining_cells_placed = place_greedy_barycenter_optimized(
        fabric_db, netlist_graph, placement, used_slots,
        net_index, cell_to_nets, kdtrees, slot_lists, pins_db, port_to_nets
    )
    print(f"\nPlaced {remaining_cells_placed} remaining cells.")
    
    total_cells = len(netlist_graph)
    total_placed = len(placement)
    print(f"\nPlacement Summary: {total_placed}/{total_cells} cells placed "
          f"({100*total_placed/total_cells:.1f}%)")
    
    return placement


def place_design_with_sa(fabric_cells_path: str, 
                         design_path: str, 
                         pins_path: str = 'fabric/pins.yaml',
                         enable_sa: bool = True,
                         sa_alpha: float = 0.95,
                         sa_moves_per_temp: int = 100,
                         sa_T_final: float = 0.1,
                         generate_move_func=None) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Place logical cells onto physical fabric slots using greedy + SA algorithm.
    
    Args:
        fabric_cells_path: Path to fabric_cells.yaml
        design_path: Path to design mapped JSON file
        pins_path: Path to pins.yaml
        enable_sa: Whether to run simulated annealing after greedy placement
        sa_alpha: SA cooling rate (default: 0.95)
        sa_moves_per_temp: SA moves per temperature step (default: 100)
        sa_T_final: SA final temperature (default: 0.1)
        generate_move_func: Optional move generation function (for Issue #5)
    
    Returns:
        placement: Dict mapping logical_instance_name -> {fabric_slot_name, x, y, orient}
                  Returns None if validation fails
    """
    # Run greedy placement first
    greedy_placement = place_design(fabric_cells_path, design_path, pins_path)
    
    if greedy_placement is None:
        return None
    
    if not enable_sa:
        return greedy_placement
    
    # Prepare data for SA
    fabric_db = parse_fabric_cells(fabric_cells_path)
    _, netlist_graph = parse_design(design_path)
    pins_db = parse_pins(pins_path)
    port_to_nets = get_port_to_net_mapping(design_path)
    nets_dict = extract_nets(netlist_graph)
    
    # Run simulated annealing
    sa_placement = simulated_annealing(
        greedy_placement,
        fabric_db,
        netlist_graph,
        nets_dict,
        pins_db,
        port_to_nets,
        T_initial=None,  # Auto-calculate
        alpha=sa_alpha,
        T_final=sa_T_final,
        moves_per_temp=sa_moves_per_temp,
        generate_move_func=generate_move_func
    )
    
    return sa_placement


# ============================================================================
# Simulated Annealing Core Algorithm (Issue #4)
# ============================================================================

def calculate_initial_temperature(initial_hpwl: float, multiplier: float = 10000.0) -> float:
    """
    Calculate initial temperature for simulated annealing.
    
    Args:
        initial_hpwl: Initial HPWL value
        multiplier: Multiplier for temperature scaling (default: 10000.0)
    
    Returns:
        Initial temperature
    """
    return multiplier * initial_hpwl


def cool_temperature(temperature: float, alpha: float) -> float:
    """
    Cool temperature according to annealing schedule.
    
    Args:
        temperature: Current temperature
        alpha: Cooling rate (typically 0.85-0.99)
    
    Returns:
        New temperature
    """
    return temperature * alpha


def should_accept_move(delta_cost: float, temperature: float) -> bool:
    """
    Determine if a move should be accepted using Metropolis criterion.
    
    If delta_cost < 0 (improvement): always accept
    If delta_cost >= 0 (worse): accept with probability exp(-delta_cost / T)
    
    Args:
        delta_cost: Change in cost (new_cost - old_cost)
        temperature: Current temperature
    
    Returns:
        True if move should be accepted, False otherwise
    """
    if delta_cost < 0:
        return True  # Always accept improvements
    
    if temperature <= 0:
        return False  # At zero temperature, reject all non-improving moves
    
    # Metropolis criterion: P(accept) = exp(-delta_cost / T)
    probability = math.exp(-delta_cost / temperature)
    return random.random() < probability


def simulated_annealing(initial_placement: Dict[str, Dict[str, Any]],
                       fabric_db: Dict[str, List[Dict[str, Any]]],
                       netlist_graph: Dict[str, Dict[str, Any]],
                       nets_dict: Dict[int, List[str]],
                       pins_db: Optional[Dict[str, Any]] = None,
                       port_to_nets: Optional[Dict[str, List[int]]] = None,
                       T_initial: Optional[float] = None,
                       alpha: float = 0.95,
                       T_final: float = 0.1,
                       moves_per_temp: int = 100,
                       generate_move_func=None) -> Dict[str, Dict[str, Any]]:
    """
    Simulated Annealing optimization algorithm.
    
    Takes the greedy placement as initial state and optimizes it to reduce HPWL.
    
    Args:
        initial_placement: Initial placement from greedy algorithm
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        netlist_graph: Dict mapping instance_name -> {type, connections}
        nets_dict: Dict mapping net_id -> list of cell instances (from extract_nets)
        pins_db: Optional pins database
        port_to_nets: Optional port-to-net mapping
        T_initial: Initial temperature (auto-calculated if None)
        alpha: Cooling rate (default: 0.95)
        T_final: Final temperature (default: 0.1)
        moves_per_temp: Number of moves attempted per temperature step (default: 100)
        generate_move_func: Function to generate moves (placeholder for Issue #5)
                          Signature: (placement, ...) -> (new_placement, delta_cost, move_type)
    
    Returns:
        Best placement found
    """
    # Initialize state
    current_placement = {k: v.copy() for k, v in initial_placement.items()}
    current_hpwl = calculate_total_hpwl(current_placement, nets_dict, fabric_db, pins_db, port_to_nets)
    
    best_placement = {k: v.copy() for k, v in current_placement.items()}
    best_hpwl = current_hpwl
    
    # Calculate initial temperature if not provided
    if T_initial is None:
        T_initial = calculate_initial_temperature(current_hpwl)
    
    T = T_initial
    iteration = 0
    
    print(f"\n{'='*60}")
    print("Simulated Annealing Optimization")
    print(f"{'='*60}")
    print(f"Initial HPWL: {current_hpwl:.2f} um")
    print(f"Initial Temperature: {T:.2f}")
    print(f"Cooling Rate (alpha): {alpha}")
    print(f"Moves per Temperature: {moves_per_temp}")
    print(f"Final Temperature: {T_final}")
    print(f"{'='*60}\n")
    
    # Main SA loop
    while T > T_final:
        acceptance_count = 0
        improvement_count = 0
        
        # Attempt N moves at this temperature
        for move_idx in range(moves_per_temp):
            # Generate move (placeholder - will be implemented in Issue #5)
            if generate_move_func is None:
                # Dummy move: swap two random cells of the same type (for testing)
                cell_names = list(current_placement.keys())
                if len(cell_names) < 2:
                    break
                
                cell1 = random.choice(cell_names)
                cell2 = random.choice(cell_names)
                
                if cell1 == cell2:
                    continue
                
                # Get cell types
                cell1_type = netlist_graph[cell1]['type']
                cell2_type = netlist_graph[cell2]['type']
                
                # Only swap if same type
                if cell1_type != cell2_type:
                    continue
                
                # Create new placement with swap
                new_placement = {k: v.copy() for k, v in current_placement.items()}
                slot1 = new_placement[cell1]['fabric_slot_name']
                slot2 = new_placement[cell2]['fabric_slot_name']
                
                # Swap slots
                new_placement[cell1]['fabric_slot_name'] = slot2
                new_placement[cell2]['fabric_slot_name'] = slot1
                
                # Update coordinates
                slot_lookup = {}
                for slots in fabric_db.values():
                    for slot in slots:
                        slot_lookup[slot['name']] = slot
                
                if slot2 in slot_lookup:
                    new_placement[cell1]['x'] = slot_lookup[slot2]['x']
                    new_placement[cell1]['y'] = slot_lookup[slot2]['y']
                
                if slot1 in slot_lookup:
                    new_placement[cell2]['x'] = slot_lookup[slot1]['x']
                    new_placement[cell2]['y'] = slot_lookup[slot1]['y']
                
                # Calculate new HPWL
                new_hpwl = calculate_total_hpwl(new_placement, nets_dict, fabric_db, pins_db, port_to_nets)
                delta_cost = new_hpwl - current_hpwl
                move_type = 'dummy_swap'
            else:
                # Use provided move generation function
                new_placement, delta_cost, move_type = generate_move_func(
                    current_placement, T, fabric_db, netlist_graph, nets_dict, 
                    pins_db, port_to_nets
                )
                new_hpwl = current_hpwl + delta_cost
            
            # Accept or reject move
            if should_accept_move(delta_cost, T):
                current_placement = new_placement
                current_hpwl = new_hpwl
                acceptance_count += 1
                
                if delta_cost < 0:
                    improvement_count += 1
                
                # Update best if needed
                if current_hpwl < best_hpwl:
                    best_placement = {k: v.copy() for k, v in current_placement.items()}
                    best_hpwl = current_hpwl
        
        # Calculate acceptance rate
        acceptance_rate = acceptance_count / moves_per_temp if moves_per_temp > 0 else 0.0
        
        # Print progress
        if iteration % 10 == 0 or T <= T_final * 2:  # Print more frequently near end
            print(f"T={T:.2f} | Current HPWL={current_hpwl:.2f} um | "
                  f"Best HPWL={best_hpwl:.2f} um | "
                  f"Accept Rate={acceptance_rate:.1%} | "
                  f"Improvements={improvement_count}")
        
        # Cool temperature
        T = cool_temperature(T, alpha)
        iteration += 1
    
    # Final statistics
    print(f"\n{'='*60}")
    print("SA Optimization Complete")
    print(f"{'='*60}")
    print(f"Initial HPWL: {calculate_total_hpwl(initial_placement, nets_dict, fabric_db, pins_db, port_to_nets):.2f} um")
    print(f"Final HPWL: {best_hpwl:.2f} um")
    improvement = calculate_total_hpwl(initial_placement, nets_dict, fabric_db, pins_db, port_to_nets) - best_hpwl
    improvement_pct = (improvement / calculate_total_hpwl(initial_placement, nets_dict, fabric_db, pins_db, port_to_nets)) * 100
    print(f"Improvement: {improvement:.2f} um ({improvement_pct:.2f}%)")
    print(f"Total Iterations: {iteration}")
    print(f"{'='*60}\n")
    
    return best_placement


# ============================================================================
# End of Simulated Annealing Core
# ============================================================================


def save_placement(placement: Dict[str, Dict[str, Any]], output_path: str):
    """Save placement results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(placement, f, indent=2)
    print(f"\nPlacement saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Optimized placer: Place logical cells onto physical fabric slots'
    )
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--design', default='designs/6502_mapped.json',
                        help='Path to design mapped JSON file')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                        help='Path to pins.yaml')
    parser.add_argument('--output', default='placement.json',
                        help='Output file for placement results')
    parser.add_argument('--no-sa', action='store_true',
                        help='Disable simulated annealing (greedy only)')
    parser.add_argument('--sa-alpha', type=float, default=0.95,
                        help='SA cooling rate (default: 0.95)')
    parser.add_argument('--sa-moves', type=int, default=100,
                        help='SA moves per temperature step (default: 100)')
    parser.add_argument('--sa-T-final', type=float, default=0.1,
                        help='SA final temperature (default: 0.1)')
    
    args = parser.parse_args()
    
    placement = place_design_with_sa(
        args.fabric_cells, 
        args.design, 
        args.pins,
        enable_sa=not args.no_sa,
        sa_alpha=args.sa_alpha,
        sa_moves_per_temp=args.sa_moves,
        sa_T_final=args.sa_T_final
    )
    
    if placement is None:
        sys.exit(1)
    
    save_placement(placement, args.output)
    print("\nPlacement completed successfully!")


if __name__ == '__main__':
    main()