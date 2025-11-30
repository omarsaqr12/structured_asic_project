#!/usr/bin/env python3
"""
H-Tree Clock Tree Synthesis (CTS) Algorithm

Implements a hierarchical H-Tree algorithm that distributes clock signals to all DFFs
(clock sinks) while minimizing clock skew. The H-Tree uses alternating horizontal
and vertical partitioning to create a balanced clock distribution network.
"""

import sys
import argparse
import json
import math
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from parse_fabric import parse_fabric_cells
from parse_design import parse_design

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Falling back to linear search for buffers.")


class HTreeNode:
    """Represents a node in the H-Tree structure."""
    def __init__(self, node_type: str, buffer_slot: Optional[str] = None,
                 x: float = 0.0, y: float = 0.0, sinks: List[Dict] = None):
        self.type = node_type  # 'buffer' or 'sink'
        self.buffer_slot = buffer_slot
        self.x = x
        self.y = y
        self.sinks = sinks or []  # List of DFF sinks for leaf nodes
        self.children = []  # List of HTreeNode children
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization."""
        result = {
            'type': self.type,
            'x': self.x,
            'y': self.y
        }
        
        if self.type == 'buffer':
            result['buffer_slot'] = self.buffer_slot
            result['children'] = [child.to_dict() for child in self.children]
        else:  # sink (leaf)
            result['sinks'] = self.sinks
        
        return result


def calculate_geometric_center(positions: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate the geometric center (centroid) of a list of positions."""
    if not positions:
        return (0.0, 0.0)
    
    total_x = sum(x for x, y in positions)
    total_y = sum(y for x, y in positions)
    count = len(positions)
    
    return (total_x / count, total_y / count)


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_nearest_buffer(target_x: float, target_y: float,
                       available_buffers: List[Dict[str, Any]],
                       kdtree: Optional[Any] = None,
                       buffer_coords: Optional[List[Tuple[float, float]]] = None) -> Optional[Dict[str, Any]]:
    """Find the nearest available buffer to the target position."""
    if not available_buffers:
        return None
    
    if HAS_SCIPY and kdtree is not None and buffer_coords is not None:
        # Use KD-tree for fast search
        k = min(10, len(available_buffers))
        result = kdtree.query([target_x, target_y], k=k)
        
        # Handle both single result and array results
        if k == 1:
            # Single result: returns (distance, index) as scalars
            distances, indices = result
            indices = [int(indices)] if not isinstance(indices, (list, tuple)) else [int(i) for i in indices]
        else:
            # Multiple results: returns (distances_array, indices_array)
            distances, indices = result
            # Convert numpy array to list of integers
            if hasattr(indices, '__iter__') and not isinstance(indices, (str, bytes)):
                indices = [int(i) for i in indices]
            else:
                indices = [int(indices)]
        
        for idx in indices:
            if 0 <= idx < len(available_buffers):
                return available_buffers[idx]
    
    # Fallback to linear search
    nearest = None
    min_distance = float('inf')
    
    for buffer in available_buffers:
        distance = calculate_distance(target_x, target_y, buffer['x'], buffer['y'])
        if distance < min_distance:
            min_distance = distance
            nearest = buffer
    
    return nearest


def partition_sinks(sinks: List[Dict[str, Any]], level: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Partition sinks using H-pattern: alternate horizontal and vertical splits.
    
    Args:
        sinks: List of sink dictionaries with 'x' and 'y' keys
        level: Current recursion level (0 = horizontal, 1 = vertical, etc.)
    
    Returns:
        Tuple of (left_partition, right_partition)
    """
    if not sinks:
        return [], []
    
    if level % 2 == 0:  # Horizontal split (by Y-coordinate)
        sorted_sinks = sorted(sinks, key=lambda s: s['y'])
    else:  # Vertical split (by X-coordinate)
        sorted_sinks = sorted(sinks, key=lambda s: s['x'])
    
    mid = len(sorted_sinks) // 2
    left_partition = sorted_sinks[:mid]
    right_partition = sorted_sinks[mid:]
    
    return left_partition, right_partition


def build_htree(sinks: List[Dict[str, Any]],
                available_buffers: List[Dict[str, Any]],
                used_buffers: Set[str],
                level: int = 0,
                threshold: int = 4,
                kdtree: Optional[Any] = None,
                buffer_coords: Optional[List[Tuple[float, float]]] = None) -> HTreeNode:
    """
    Recursively build H-Tree structure.
    
    Args:
        sinks: List of DFF sink dictionaries with 'dff_name', 'dff_slot', 'x', 'y'
        available_buffers: List of available buffer slots
        used_buffers: Set of already used buffer slot names
        level: Current recursion level
        threshold: Maximum number of sinks per leaf node
        kdtree: Optional KD-tree for fast buffer search
        buffer_coords: Optional list of buffer coordinates for KD-tree
    
    Returns:
        HTreeNode representing the tree structure
    """
    # Base case: create leaf node if sinks <= threshold
    if len(sinks) <= threshold:
        return HTreeNode(
            node_type='sink',
            x=calculate_geometric_center([(s['x'], s['y']) for s in sinks])[0],
            y=calculate_geometric_center([(s['x'], s['y']) for s in sinks])[1],
            sinks=sinks
        )
    
    # Partition sinks
    left_sinks, right_sinks = partition_sinks(sinks, level)
    
    if not left_sinks or not right_sinks:
        # If partition failed, create leaf node
        return HTreeNode(
            node_type='sink',
            x=calculate_geometric_center([(s['x'], s['y']) for s in sinks])[0],
            y=calculate_geometric_center([(s['x'], s['y']) for s in sinks])[1],
            sinks=sinks
        )
    
    # Calculate geometric center of all sinks
    all_positions = [(s['x'], s['y']) for s in sinks]
    center_x, center_y = calculate_geometric_center(all_positions)
    
    # Find nearest available buffer
    # Filter out already used buffers
    unused_buffers = [b for b in available_buffers if b['name'] not in used_buffers]
    
    if not unused_buffers:
        # No buffers available, create internal node without buffer
        # This shouldn't happen in practice, but handle gracefully
        buffer_node = HTreeNode(
            node_type='buffer',
            buffer_slot=None,
            x=center_x,
            y=center_y
        )
    else:
        # Build KD-tree for unused buffers
        unused_kdtree = None
        unused_coords = None
        if HAS_SCIPY and unused_buffers:
            unused_coords = [(b['x'], b['y']) for b in unused_buffers]
            unused_kdtree = KDTree(unused_coords)
        
        best_buffer = find_nearest_buffer(
            center_x, center_y, unused_buffers, unused_kdtree, unused_coords
        )
        
        if best_buffer:
            used_buffers.add(best_buffer['name'])
            buffer_node = HTreeNode(
                node_type='buffer',
                buffer_slot=best_buffer['name'],
                x=best_buffer['x'],
                y=best_buffer['y']
            )
        else:
            buffer_node = HTreeNode(
                node_type='buffer',
                buffer_slot=None,
                x=center_x,
                y=center_y
            )
    
    # Recursively build children
    left_child = build_htree(
        left_sinks, available_buffers, used_buffers, level + 1, threshold,
        kdtree, buffer_coords
    )
    right_child = build_htree(
        right_sinks, available_buffers, used_buffers, level + 1, threshold,
        kdtree, buffer_coords
    )
    
    buffer_node.children = [left_child, right_child]
    
    return buffer_node


def identify_dffs(design_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Identify all DFF instances from the design netlist.
    
    Args:
        design_path: Path to design mapped JSON file
    
    Returns:
        Dict mapping DFF logical_name -> {type, connections}
    """
    logical_db, netlist_graph = parse_design(design_path)
    
    dffs = {}
    dff_type = 'sky130_fd_sc_hd__dfbbp_1'
    
    for cell_name, cell_data in netlist_graph.items():
        if cell_data.get('type') == dff_type:
            dffs[cell_name] = cell_data
    
    return dffs


def get_dff_positions(placement: Dict[str, Dict[str, Any]],
                      dffs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Map DFF logical names to their physical positions from placement.
    
    Args:
        placement: Placement dict mapping logical_name -> {fabric_slot_name, x, y, orient}
        dffs: Dict of DFF instances
    
    Returns:
        List of sink dictionaries with 'dff_name', 'dff_slot', 'x', 'y'
    """
    sinks = []
    
    for dff_name, dff_data in dffs.items():
        if dff_name in placement:
            pos = placement[dff_name]
            sinks.append({
                'dff_name': dff_name,
                'dff_slot': pos['fabric_slot_name'],
                'x': pos['x'],
                'y': pos['y']
            })
        else:
            print(f"WARNING: DFF {dff_name} not found in placement")
    
    return sinks


def get_available_buffers(fabric_cells_path: str,
                          placement: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get all available buffer and inverter slots that are not used in placement.
    
    Args:
        fabric_cells_path: Path to fabric_cells.yaml
        placement: Placement dict to check for used slots
    
    Returns:
        List of buffer dictionaries with 'name', 'x', 'y', 'orient', 'type'
    """
    fabric_db = parse_fabric_cells(fabric_cells_path)
    
    # Buffer and inverter types
    buffer_types = ['sky130_fd_sc_hd__clkbuf_4', 'sky130_fd_sc_hd__clkinv_2']
    
    # Get all used slots from placement
    used_slots = {pos['fabric_slot_name'] for pos in placement.values()}
    
    available_buffers = []
    
    for cell_type in buffer_types:
        if cell_type in fabric_db:
            for slot in fabric_db[cell_type]:
                if slot['name'] not in used_slots:
                    available_buffers.append({
                        'name': slot['name'],
                        'x': slot['x'],
                        'y': slot['y'],
                        'orient': slot['orient'],
                        'type': cell_type
                    })
    
    return available_buffers


def build_spatial_index(buffers: List[Dict[str, Any]]) -> Tuple[Optional[Any], Optional[List[Tuple[float, float]]]]:
    """Build spatial index (KD-tree) for fast buffer lookup."""
    if not HAS_SCIPY or not buffers:
        return None, None
    
    coords = [(b['x'], b['y']) for b in buffers]
    kdtree = KDTree(coords)
    
    return kdtree, coords


def synthesize_clock_tree(placement_path: str,
                         design_path: str,
                         fabric_cells_path: str,
                         threshold: int = 4) -> Tuple[HTreeNode, List[str]]:
    """
    Synthesize H-Tree clock tree for a placed design.
    
    Args:
        placement_path: Path to placement JSON file
        design_path: Path to design mapped JSON file
        fabric_cells_path: Path to fabric_cells.yaml
        threshold: Maximum number of sinks per leaf node
    
    Returns:
        Tuple of (root_node, list_of_used_buffers)
    """
    # Load placement
    print("Loading placement...")
    with open(placement_path, 'r') as f:
        placement = json.load(f)
    
    # Identify DFFs
    print("Identifying DFFs...")
    dffs = identify_dffs(design_path)
    print(f"Found {len(dffs)} DFF instances")
    
    # Get DFF positions
    sinks = get_dff_positions(placement, dffs)
    print(f"Found {len(sinks)} placed DFFs")
    
    if not sinks:
        print("ERROR: No DFFs found in placement")
        return None, []
    
    # Get available buffers
    print("Finding available buffers...")
    available_buffers = get_available_buffers(fabric_cells_path, placement)
    print(f"Found {len(available_buffers)} available buffers/inverters")
    if len(available_buffers) == 0:
        print("ERROR: No available buffers found")
        return None, []
    
    if not available_buffers:
        print("ERROR: No available buffers found")
        return None, []
    
    # Build spatial index
    kdtree, buffer_coords = build_spatial_index(available_buffers)
    if kdtree:
        print("Built KD-tree for fast buffer lookup")
    
    # Build H-Tree
    print("Building H-Tree structure...")
    used_buffers = set()
    root = build_htree(
        sinks, available_buffers, used_buffers, level=0, threshold=threshold,
        kdtree=kdtree, buffer_coords=buffer_coords
    )
    
    print(f"Used {len(used_buffers)} buffers for clock tree")
    
    return root, list(used_buffers)


def save_cts_tree(root: HTreeNode, output_path: str):
    """Save CTS tree structure to JSON file."""
    tree_dict = {
        'root': root.to_dict(),
        'sinks': []
    }
    
    # Collect all sinks from the tree
    def collect_sinks(node: HTreeNode):
        if node.type == 'sink':
            tree_dict['sinks'].extend(node.sinks)
        else:
            for child in node.children:
                collect_sinks(child)
    
    collect_sinks(root)
    
    with open(output_path, 'w') as f:
        json.dump(tree_dict, f, indent=2)
    
    print(f"CTS tree saved to {output_path}")


def save_buffer_mapping(used_buffers: List[str], output_path: str):
    """
    Save buffer mapping to .map file.
    Format: logical_buffer_name buffer_slot_name (matches placement.map format)
    """
    with open(output_path, 'w') as f:
        for i, buffer_slot in enumerate(used_buffers):
            logical_name = f"cts_buf_{i}"
            # Format: logical_name physical_slot_name (consistent with placement.map)
            f.write(f"{logical_name} {buffer_slot}\n")
    
    print(f"Buffer mapping saved to {output_path}")


def update_placement_map(placement_map_path: str, 
                        used_buffers: List[str],
                        fabric_cells_path: str) -> None:
    """
    Update placement.map file by appending CTS buffer mappings.
    
    Args:
        placement_map_path: Path to existing placement.map file
        used_buffers: List of buffer slot names used in CTS
        fabric_cells_path: Path to fabric_cells.yaml to get buffer positions
    """
    # Get buffer positions from fabric
    fabric_db = parse_fabric_cells(fabric_cells_path)
    buffer_types = ['sky130_fd_sc_hd__clkbuf_4', 'sky130_fd_sc_hd__clkinv_2']
    
    # Build mapping of slot name to buffer info
    buffer_info = {}
    for cell_type in buffer_types:
        if cell_type in fabric_db:
            for slot in fabric_db[cell_type]:
                buffer_info[slot['name']] = slot
    
    # Append CTS buffers to placement.map
    with open(placement_map_path, 'a') as f:  # Append mode
        for i, buffer_slot in enumerate(used_buffers):
            logical_name = f"cts_buf_{i}"
            f.write(f"{logical_name} {buffer_slot}\n")
    
    print(f"Updated placement.map with {len(used_buffers)} CTS buffers")


def update_placement_json(placement_json_path: str,
                         used_buffers: List[str],
                         fabric_cells_path: str) -> None:
    """
    Update placement JSON file by adding CTS buffer entries.
    
    Args:
        placement_json_path: Path to existing placement JSON file
        used_buffers: List of buffer slot names used in CTS
        fabric_cells_path: Path to fabric_cells.yaml to get buffer positions
    """
    # Load existing placement
    with open(placement_json_path, 'r') as f:
        placement = json.load(f)
    
    # Get buffer positions from fabric
    fabric_db = parse_fabric_cells(fabric_cells_path)
    buffer_types = ['sky130_fd_sc_hd__clkbuf_4', 'sky130_fd_sc_hd__clkinv_2']
    
    # Build mapping of slot name to buffer info
    buffer_info = {}
    for cell_type in buffer_types:
        if cell_type in fabric_db:
            for slot in fabric_db[cell_type]:
                buffer_info[slot['name']] = slot
    
    # Add CTS buffers to placement
    for i, buffer_slot in enumerate(used_buffers):
        logical_name = f"cts_buf_{i}"
        if buffer_slot in buffer_info:
            slot_info = buffer_info[buffer_slot]
            placement[logical_name] = {
                'fabric_slot_name': buffer_slot,
                'x': slot_info['x'],
                'y': slot_info['y'],
                'orient': slot_info.get('orient', 'N')
            }
    
    # Save updated placement
    with open(placement_json_path, 'w') as f:
        json.dump(placement, f, indent=2)
    
    print(f"Updated placement JSON with {len(used_buffers)} CTS buffers")


def update_netlist_connections(design_path: str,
                              root: HTreeNode,
                              used_buffers: List[str],
                              output_path: str,
                              clock_net_id: int = 1) -> None:
    """
    Update netlist with clock tree connections.
    Creates a new netlist with CTS buffers and clock connections.
    
    Args:
        design_path: Path to original design JSON file
        root: Root node of CTS tree
        used_buffers: List of buffer slot names
        output_path: Path to save updated netlist
        clock_net_id: Net ID to use for clock signal (default: 1)
    """
    # Load original design
    with open(design_path, 'r') as f:
        design = json.load(f)
    
    # Get modules
    modules = design.get('modules', {})
    
    # Find top module
    top_module = None
    for mod_name, mod_data in modules.items():
        if mod_data.get('attributes', {}).get('top') == '00000000000000000000000000000001':
            top_module = mod_name
            break
    
    if not top_module:
        for mod_name, mod_data in modules.items():
            if 'cells' in mod_data and len(mod_data['cells']) > 0:
                top_module = mod_name
                break
    
    if not top_module:
        print("ERROR: Could not find top module")
        return
    
    module_data = modules[top_module]
    cells = module_data.get('cells', {})
    
    # Determine buffer cell types from fabric
    # For now, use clkbuf_4 as default (can be enhanced to choose based on fanout)
    buffer_cell_type = 'sky130_fd_sc_hd__clkbuf_4'
    
    # Add CTS buffer cells to netlist
    for i, buffer_slot in enumerate(used_buffers):
        logical_name = f"cts_buf_{i}"
        cells[logical_name] = {
            'type': buffer_cell_type,
            'connections': {
                'A': [clock_net_id],  # Input from parent buffer or clock source
                'Y': [clock_net_id + 1000 + i]  # Output to children
            }
        }
    
    # Build clock tree connections
    def connect_tree(node: HTreeNode, parent_net: int):
        """Recursively connect tree nodes."""
        if node.type == 'buffer':
            # Buffer input connects to parent net
            buffer_name = f"cts_buf_{used_buffers.index(node.buffer_slot)}"
            if buffer_name in cells:
                # Update buffer input to parent net
                cells[buffer_name]['connections']['A'] = [parent_net]
                # Output net for children
                output_net = clock_net_id + 1000 + used_buffers.index(node.buffer_slot)
                cells[buffer_name]['connections']['Y'] = [output_net]
                
                # Connect children
                for child in node.children:
                    connect_tree(child, output_net)
        else:  # sink
            # Connect DFF clock pins to parent net
            for sink in node.sinks:
                dff_name = sink['dff_name']
                if dff_name in cells:
                    # Add clock connection
                    if 'CLK' in cells[dff_name].get('connections', {}):
                        cells[dff_name]['connections']['CLK'] = [parent_net]
                    else:
                        cells[dff_name].setdefault('connections', {})['CLK'] = [parent_net]
    
    # Connect tree starting from root (clock source net)
    connect_tree(root, clock_net_id)
    
    # Save updated design
    with open(output_path, 'w') as f:
        json.dump(design, f, indent=2)
    
    print(f"Updated netlist saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='H-Tree Clock Tree Synthesis (CTS) Algorithm'
    )
    parser.add_argument('--placement', required=True,
                        help='Path to placement JSON file')
    parser.add_argument('--design', required=True,
                        help='Path to design mapped JSON file')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--output-tree', default='cts_tree.json',
                        help='Output file for CTS tree structure')
    parser.add_argument('--output-buffers', default='cts_buffers.map',
                        help='Output file for buffer mapping')
    parser.add_argument('--placement-map', default=None,
                        help='Path to placement.map file to update (optional)')
    parser.add_argument('--placement-json-update', default=None,
                        help='Path to placement JSON file to update (optional)')
    parser.add_argument('--netlist-update', default=None,
                        help='Path to save updated netlist with clock connections (optional)')
    parser.add_argument('--threshold', type=int, default=4,
                        help='Maximum number of sinks per leaf node (default: 4)')
    
    args = parser.parse_args()
    
    # Synthesize clock tree
    root, used_buffers = synthesize_clock_tree(
        args.placement, args.design, args.fabric_cells, args.threshold
    )
    
    if root is None:
        print("ERROR: Failed to synthesize clock tree")
        sys.exit(1)
    
    # Save outputs
    save_cts_tree(root, args.output_tree)
    save_buffer_mapping(used_buffers, args.output_buffers)
    
    # Update placement files if requested
    if args.placement_map:
        update_placement_map(args.placement_map, used_buffers, args.fabric_cells)
    
    if args.placement_json_update:
        update_placement_json(args.placement_json_update, used_buffers, args.fabric_cells)
    
    # Update netlist if requested
    if args.netlist_update:
        update_netlist_connections(args.design, root, used_buffers, args.netlist_update)
    
    print("\nCTS synthesis completed successfully!")


if __name__ == '__main__':
    main()

