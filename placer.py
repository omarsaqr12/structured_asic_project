#!/usr/bin/env python3
"""
Placer: Maps logical cells to physical fabric slots using a greedy algorithm
to minimize total Half-Perimeter Wirelength (HPWL).
"""

import sys
import argparse
import json
import math
from typing import Dict, List, Tuple, Any, Optional, Set
from validator import validate_design
from parse_fabric import parse_fabric_cells, parse_pins
from parse_design import parse_design
from tqdm import tqdm


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


def find_io_connected_cells(netlist_graph: Dict[str, Dict[str, Any]], 
                            io_net_ids: Set[int]) -> Dict[int, List[str]]:
    """
    Find all cells directly connected to I/O pin nets.
    
    Args:
        netlist_graph: Dict mapping instance_name -> {type, connections}
        io_net_ids: Set of net IDs that are connected to I/O pins
    
    Returns:
        io_net_to_cells: Dict mapping net_id -> list of cell instance names
    """
    io_net_to_cells = {}
    
    for cell_name, cell_data in netlist_graph.items():
        connections = cell_data.get('connections', {})
        
        # Check all ports of this cell
        for port_name, net_list in connections.items():
            for net_id in net_list:
                if net_id in io_net_ids:
                    if net_id not in io_net_to_cells:
                        io_net_to_cells[net_id] = []
                    if cell_name not in io_net_to_cells[net_id]:
                        io_net_to_cells[net_id].append(cell_name)
    
    return io_net_to_cells


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def place_io_connected_cells(fabric_db: Dict[str, List[Dict[str, Any]]],
                               pins_db: Dict[str, Any],
                               netlist_graph: Dict[str, Dict[str, Any]],
                               port_to_nets: Dict[str, List[int]],
                               placement: Dict[str, Dict[str, Any]],
                               used_slots: Set[str]) -> int:
    """
    Place all cells connected directly to fixed I/O pins at the nearest valid fabric slot.
    
    Args:
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        pins_db: Dict containing die, core, and pins information
        netlist_graph: Dict mapping instance_name -> {type, connections}
        port_to_nets: Dict mapping port_name -> list of net IDs
        placement: Current placement dict (will be updated)
        used_slots: Set of already used fabric slot names (will be updated)
    
    Returns:
        Number of cells placed
    """
    # Step 1: Get fixed I/O pins and map to net IDs
    fixed_pins = {}
    io_net_ids = set()
    
    for pin in pins_db.get('pins', []):
        if pin.get('status') == 'FIXED':
            pin_name = pin['name']
            # Find corresponding net IDs from port mapping
            if pin_name in port_to_nets:
                net_ids = port_to_nets[pin_name]
                fixed_pins[pin_name] = {
                    'x_um': pin['x_um'],
                    'y_um': pin['y_um'],
                    'net_ids': net_ids
                }
                io_net_ids.update(net_ids)
    
    if not io_net_ids:
        print("No fixed I/O pins found.")
        return 0
    
    # Step 2: Find cells connected to I/O pins
    io_net_to_cells = find_io_connected_cells(netlist_graph, io_net_ids)
    
    if not io_net_to_cells:
        print("No cells found connected to I/O pins.")
        return 0
    
    # Step 3: For each I/O net, place connected cells
    cells_placed = 0
    
    for net_id, cell_names in io_net_to_cells.items():
        # Find the pin(s) connected to this net
        pin_locations = []
        for pin_name, pin_info in fixed_pins.items():
            if net_id in pin_info['net_ids']:
                pin_locations.append((pin_info['x_um'], pin_info['y_um']))
        
        if not pin_locations:
            continue
        
        # Use the first pin location (or could average multiple pins)
        pin_x, pin_y = pin_locations[0]
        
        
        all_io_cells = [c for cells in io_net_to_cells.values() for c in cells]
        # Place each cell connected to this I/O net
        for cell_name in tqdm(all_io_cells, desc="Placing I/O cells"):
            if cell_name in placement:
                continue  # Already placed
            
            cell_type = netlist_graph[cell_name]['type']
            
            # Get available slots of this type
            available_slots = fabric_db.get(cell_type, [])
            available_slots = [slot for slot in available_slots 
                             if slot['name'] not in used_slots]
            
            if not available_slots:
                print(f"WARNING: No available slots for cell {cell_name} (type {cell_type})")
                continue
            
            # Find nearest slot
            nearest_slot = None
            min_distance = float('inf')
            
            for slot in available_slots:
                distance = calculate_distance(pin_x, pin_y, slot['x'], slot['y'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_slot = slot
            
            if nearest_slot:
                # Place the cell
                placement[cell_name] = {
                    'fabric_slot_name': nearest_slot['name'],
                    'x': nearest_slot['x'],
                    'y': nearest_slot['y'],
                    'orient': nearest_slot['orient']
                }
                used_slots.add(nearest_slot['name'])
                cells_placed += 1
    
    return cells_placed


def find_placed_neighbors(cell_name: str,
                          netlist_graph: Dict[str, Dict[str, Any]],
                          placement: Dict[str, Dict[str, Any]],
                          include_io_pins: bool = False,
                          pins_db: Optional[Dict[str, Any]] = None,
                          port_to_nets: Optional[Dict[str, List[int]]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Find all already-placed cells (and optionally I/O pins) that share nets with this cell.
    
    Args:
        cell_name: Name of the cell to find neighbors for
        netlist_graph: Dict mapping instance_name -> {type, connections}
        placement: Current placement dict
        include_io_pins: If True, include I/O pin positions as neighbors
        pins_db: Optional pins database for I/O pin positions
        port_to_nets: Optional port-to-net mapping for I/O pins
    
    Returns:
        List of (neighbor_name, position_dict) tuples where position_dict has 'x' and 'y'
    """
    neighbors = []
    cell_data = netlist_graph.get(cell_name)
    if not cell_data:
        return neighbors
    
    cell_connections = cell_data.get('connections', {})
    
    # Collect all nets this cell is connected to
    cell_nets = set()
    for port_nets in cell_connections.values():
        cell_nets.update(port_nets)
    
    # Find other placed cells on these nets
    for other_cell, other_data in netlist_graph.items():
        if other_cell == cell_name or other_cell not in placement:
            continue
        
        other_connections = other_data.get('connections', {})
        other_nets = set()
        for port_nets in other_connections.values():
            other_nets.update(port_nets)
        
        # If they share any net, they're neighbors
        if cell_nets & other_nets:  # Set intersection
            neighbors.append((other_cell, placement[other_cell]))
    
    # Optionally include I/O pins as neighbors
    if include_io_pins and pins_db and port_to_nets:
        for pin in pins_db.get('pins', []):
            if pin.get('status') == 'FIXED':
                pin_name = pin['name']
                if pin_name in port_to_nets:
                    pin_nets = set(port_to_nets[pin_name])
                    # If cell shares any net with this pin
                    if cell_nets & pin_nets:
                        neighbors.append((
                            f"pin_{pin_name}",
                            {'x': pin['x_um'], 'y': pin['y_um']}
                        ))
    
    return neighbors


def calculate_barycenter(neighbors: List[Tuple[str, Dict[str, Any]]]) -> Optional[Tuple[float, float]]:
    """
    Calculate barycenter (center of gravity) of neighbors.
    
    Args:
        neighbors: List of (name, position_dict) tuples where position_dict has 'x' and 'y'
    
    Returns:
        (barycenter_x, barycenter_y) in microns, or None if no neighbors
    """
    if not neighbors:
        return None
    
    total_x = sum(pos['x'] for _, pos in neighbors)
    total_y = sum(pos['y'] for _, pos in neighbors)
    count = len(neighbors)
    
    return (total_x / count, total_y / count)


def get_fallback_position(pins_db: Optional[Dict[str, Any]] = None,
                          fabric_db: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Tuple[float, float]:
    """
    Get a fallback position when a cell has no placed neighbors.
    Uses center of fabric or nearest I/O pin.
    
    Args:
        pins_db: Optional pins database
        fabric_db: Optional fabric database
    
    Returns:
        (x, y) position in microns
    """
    # Try to use center of fabric if available
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
    
    # Fallback to center of die if pins_db available
    if pins_db:
        die_width = pins_db.get('die', {}).get('width_um', 0)
        die_height = pins_db.get('die', {}).get('height_um', 0)
        if die_width > 0 and die_height > 0:
            return (die_width / 2, die_height / 2)
    
    # Last resort: return origin
    return (0.0, 0.0)


def find_nearest_slot_to_point(target_x: float,
                                target_y: float,
                                available_slots: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the nearest available slot to a target point.
    
    Args:
        target_x: Target X coordinate in microns
        target_y: Target Y coordinate in microns
        available_slots: List of available slot dicts with 'x', 'y', 'name', 'orient'
    
    Returns:
        Best slot dict or None if no slots available
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


def place_greedy_barycenter(fabric_db: Dict[str, List[Dict[str, Any]]],
                             netlist_graph: Dict[str, Dict[str, Any]],
                             placement: Dict[str, Dict[str, Any]],
                             used_slots: Set[str],
                             pins_db: Optional[Dict[str, Any]] = None,
                             port_to_nets: Optional[Dict[str, List[int]]] = None) -> int:
    """
    Iteratively place most-connected unplaced cell at barycenter of its already-placed neighbors.
    
    Args:
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        netlist_graph: Dict mapping instance_name -> {type, connections}
        placement: Current placement dict (will be updated)
        used_slots: Set of already used fabric slot names (will be updated)
        pins_db: Optional pins database for I/O pin positions
        port_to_nets: Optional port-to-net mapping for I/O pins
    
    Returns:
        Number of cells placed
    """
    # Get all unplaced cells
    unplaced_cells = [cell for cell in netlist_graph.keys() 
                      if cell not in placement]
    
    if not unplaced_cells:
        return 0
    
    cells_placed = 0
    
    for _ in tqdm(range(len(unplaced_cells)), desc="Placing remaining cells"):
        
        if not unplaced_cells:
            break

        # Step 1: Score each unplaced cell by number of placed neighbors
        cell_scores = []
        for cell_name in unplaced_cells:
            neighbors = find_placed_neighbors(
                cell_name, netlist_graph, placement,
                include_io_pins=True, pins_db=pins_db, port_to_nets=port_to_nets
            )
            # Also count total connections as tie-breaker
            cell_data = netlist_graph[cell_name]
            connections = cell_data.get('connections', {})
            total_connections = sum(len(nets) for nets in connections.values())
            
            cell_scores.append((
                cell_name,
                len(neighbors),  # Primary: number of placed neighbors
                total_connections  # Secondary: total connection count
            ))
        
        # Step 2: Sort by neighbor count (most neighbors first), then by total connections
        cell_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Step 3: Place the top cell
        cell_to_place, neighbor_count, _ = cell_scores[0]
        cell_type = netlist_graph[cell_to_place]['type']
        
        # Get placed neighbors
        neighbors = find_placed_neighbors(
            cell_to_place, netlist_graph, placement,
            include_io_pins=True, pins_db=pins_db, port_to_nets=port_to_nets
        )
        
        # Calculate barycenter
        if neighbors:
            barycenter_x, barycenter_y = calculate_barycenter(neighbors)
        else:
            # Fallback: use center of fabric or nearest I/O pin
            barycenter_x, barycenter_y = get_fallback_position(pins_db, fabric_db)
        
        # Find available slots
        available_slots = [slot for slot in fabric_db.get(cell_type, [])
                           if slot['name'] not in used_slots]
        
        if not available_slots:
            print(f"WARNING: No slots available for cell {cell_to_place} (type {cell_type})")
            unplaced_cells.remove(cell_to_place)
            continue
        
        # Find nearest slot to barycenter
        best_slot = find_nearest_slot_to_point(
            barycenter_x, barycenter_y, available_slots
        )
        
        if best_slot:
            # Place the cell
            placement[cell_to_place] = {
                'fabric_slot_name': best_slot['name'],
                'x': best_slot['x'],
                'y': best_slot['y'],
                'orient': best_slot['orient']
            }
            used_slots.add(best_slot['name'])
            unplaced_cells.remove(cell_to_place)
            cells_placed += 1
        else:
            # Should not happen, but handle gracefully
            print(f"ERROR: Could not find slot for {cell_to_place}")
            unplaced_cells.remove(cell_to_place)
    
    return cells_placed


def place_design(fabric_cells_path: str, design_path: str, pins_path: str = 'fabric/pins.yaml') -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Place logical cells onto physical fabric slots using a greedy algorithm
    to minimize total HPWL.
    
    Args:
        fabric_cells_path: Path to fabric_cells.yaml
        design_path: Path to design mapped JSON file
        pins_path: Path to pins.yaml (default: fabric/pins.yaml)
    
    Returns:
        placement: Dict mapping logical_instance_name -> {fabric_slot_name, x, y, orient}
                  Returns None if validation fails
    """
    # Step 1: Validate design against fabric
    print("Validating design against fabric...")
    if not validate_design(fabric_cells_path, design_path):
        print("\nERROR: Design validation failed. Cannot proceed with placement.")
        return None
    
    print("\nValidation passed. Proceeding with placement...\n")
    
    # Parse fabric and design data
    fabric_db = parse_fabric_cells(fabric_cells_path)
    logical_db, netlist_graph = parse_design(design_path)
    pins_db = parse_pins(pins_path)
    
    # Get port-to-net mapping
    port_to_nets = get_port_to_net_mapping(design_path)
    
    # Initialize placement and tracking
    placement = {}
    used_slots = set()
    
    # Step 2: Place cells connected to I/O pins first
    print("Placing cells connected to I/O pins...")
    io_cells_placed = place_io_connected_cells(
        fabric_db, pins_db, netlist_graph, port_to_nets, placement, used_slots
    )
    print(f"Placed {io_cells_placed} cells connected to I/O pins.\n")
    
    # Step 3: Place remaining cells using greedy barycenter algorithm
    print("Placing remaining cells using greedy barycenter algorithm...")
    remaining_cells_placed = place_greedy_barycenter(
        fabric_db, netlist_graph, placement, used_slots, pins_db, port_to_nets
    )
    print(f"Placed {remaining_cells_placed} remaining cells.\n")
    
    return placement


def calculate_hpwl(placement: Dict[str, Dict[str, Any]], 
                   netlist_graph: Dict[str, Dict[str, Any]]) -> float:
    """
    Calculate total Half-Perimeter Wirelength (HPWL) for the placement.
    
    Args:
        placement: Dict mapping logical_instance_name -> {fabric_slot_name, x, y, orient}
        netlist_graph: Dict mapping instance_name -> {type, connections}
    
    Returns:
        total_hpwl: Total HPWL across all nets
    """
    # TODO: Implement HPWL calculation
    # - For each net in the netlist:
    #   - Find all cells connected to the net
    #   - Get their (x, y) positions from placement
    #   - Calculate HPWL = (max_x - min_x) + (max_y - min_y)
    # - Sum HPWL across all nets
    
    total_hpwl = 0.0
    return total_hpwl


def main():
    parser = argparse.ArgumentParser(
        description='Place logical cells onto physical fabric slots using greedy HPWL optimization'
    )
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--design', default='designs/6502_mapped.json',
                        help='Path to design mapped JSON file')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                        help='Path to pins.yaml (default: fabric/pins.yaml)')
    parser.add_argument('--output', default='placement.json',
                        help='Output file for placement results (default: placement.json)')
    
    args = parser.parse_args()
    
    placement = place_design(args.fabric_cells, args.design, args.pins)
    
    if placement is None:
        sys.exit(1)
    
    # TODO: Save placement results to output file
    # For now, just print success message
    print(f"Placement completed. {len(placement)} cells placed.")
    print(f"Output will be saved to {args.output} (not yet implemented)")


if __name__ == '__main__':
    main()

