#!/usr/bin/env python3
"""
Minimal API for Clock Tree Synthesis (CTS)

Provides a clean interface for generating CTS trees that can be called by eco_generator.py.
"""

import json
import math
import os
from typing import Dict, List, Tuple, Any, Optional
from buffer_manager import BufferManager, parse_placement_map
from parse_design import parse_design
from parse_fabric import parse_fabric_cells


def calculate_geometric_center(positions: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate the geometric center (centroid) of a list of positions."""
    if not positions:
        return (0.0, 0.0)
    
    total_x = sum(x for x, y in positions)
    total_y = sum(y for x, y in positions)
    count = len(positions)
    
    return (total_x / count, total_y / count)


def partition_sinks_htree(sinks: List[Dict[str, Any]], level: int) -> Tuple[List[Dict], List[Dict]]:
    """Partition sinks using H-pattern: alternate horizontal and vertical splits."""
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


def partition_sinks_xtree(sinks: List[Dict[str, Any]], level: int) -> Tuple[List[Dict], List[Dict]]:
    """Partition sinks using X-pattern: alternate diagonal splits (NW/SE and NE/SW)."""
    if not sinks:
        return [], []
    
    # Calculate center point for diagonal partitioning
    center_x = sum(s['x'] for s in sinks) / len(sinks)
    center_y = sum(s['y'] for s in sinks) / len(sinks)
    
    if level % 2 == 0:  # NW/SE split (x + y < center_x + center_y)
        partition1 = [s for s in sinks if s['x'] + s['y'] < center_x + center_y]
        partition2 = [s for s in sinks if s['x'] + s['y'] >= center_x + center_y]
    else:  # NE/SW split (x - y < center_x - center_y)
        partition1 = [s for s in sinks if s['x'] - s['y'] < center_x - center_y]
        partition2 = [s for s in sinks if s['x'] - s['y'] >= center_x - center_y]
    
    # If one partition is empty, fall back to median split on the diagonal
    if not partition1 or not partition2:
        if level % 2 == 0:
            sorted_sinks = sorted(sinks, key=lambda s: s['x'] + s['y'])
        else:
            sorted_sinks = sorted(sinks, key=lambda s: s['x'] - s['y'])
        
        mid = len(sorted_sinks) // 2
        partition1 = sorted_sinks[:mid]
        partition2 = sorted_sinks[mid:]
    
    return partition1, partition2


def build_tree_with_manager(sinks: List[Dict[str, Any]],
                           buffer_manager: BufferManager,
                           tree_type: str,  # 'h' or 'x'
                           level: int = 0,
                           threshold: int = 4) -> Dict[str, Any]:
    """
    Recursively build tree structure using BufferManager.
    
    Args:
        sinks: List of DFF sink dictionaries
        buffer_manager: BufferManager instance
        tree_type: 'h' for H-Tree or 'x' for X-Tree
        level: Current recursion level
        threshold: Maximum number of sinks per leaf node
    
    Returns:
        Tree node as dictionary
    """
    # Base case: create leaf node if sinks <= threshold
    if len(sinks) <= threshold:
        center_x, center_y = calculate_geometric_center([(s['x'], s['y']) for s in sinks])
        return {
            'type': 'sink',
            'x': center_x,
            'y': center_y,
            'sinks': sinks
        }
    
    # Partition sinks based on tree type
    if tree_type == 'h':
        left_sinks, right_sinks = partition_sinks_htree(sinks, level)
    else:  # 'x'
        left_sinks, right_sinks = partition_sinks_xtree(sinks, level)
    
    if not left_sinks or not right_sinks:
        # If partition failed, create leaf node
        center_x, center_y = calculate_geometric_center([(s['x'], s['y']) for s in sinks])
        return {
            'type': 'sink',
            'x': center_x,
            'y': center_y,
            'sinks': sinks
        }
    
    # Calculate geometric center of all sinks
    all_positions = [(s['x'], s['y']) for s in sinks]
    center_x, center_y = calculate_geometric_center(all_positions)
    
    # Claim buffer using BufferManager
    try:
        buffer_name = buffer_manager.claim_buffer(
            target_x=center_x,
            target_y=center_y,
            preferred_type='BUF',
            level=level
        )
    except ValueError as e:
        raise ValueError(f"Failed to claim buffer: {e}")
    
    # Get buffer info from manager
    buffer_info = buffer_manager.claimed[buffer_name]
    buffer_x = buffer_info['x']
    buffer_y = buffer_info['y']
    
    # Recursively build children
    left_child = build_tree_with_manager(
        left_sinks, buffer_manager, tree_type, level + 1, threshold
    )
    right_child = build_tree_with_manager(
        right_sinks, buffer_manager, tree_type, level + 1, threshold
    )
    
    # Update buffer's children list in manager
    child_buffer_names = []
    if left_child['type'] == 'buffer' and 'buffer_name' in left_child:
        child_buffer_names.append(left_child['buffer_name'])
    if right_child['type'] == 'buffer' and 'buffer_name' in right_child:
        child_buffer_names.append(right_child['buffer_name'])
    
    buffer_manager.claimed[buffer_name]['children'] = child_buffer_names
    
    # Create buffer node
    buffer_node = {
        'type': 'buffer',
        'buffer_name': buffer_name,
        'x': buffer_x,
        'y': buffer_y,
        'children': [left_child, right_child]
    }
    
    return buffer_node


def get_dff_positions_from_map(placement_map: Dict[str, str],
                               fabric_db: Dict[str, List[Dict[str, Any]]],
                               dff_logical_names: List[str]) -> List[Dict[str, Any]]:
    """
    Get DFF positions from placement map and fabric database.
    
    Args:
        placement_map: Dict mapping logical_name -> physical_slot_name
        fabric_db: Fabric database from parse_fabric_cells
        dff_logical_names: List of DFF logical names to find
    
    Returns:
        List of sink dictionaries with 'dff_name', 'dff_slot', 'x', 'y'
    """
    sinks = []
    
    # Build slot lookup from fabric_db
    slot_lookup = {}
    for cell_type, slots in fabric_db.items():
        for slot in slots:
            slot_lookup[slot['name']] = slot
    
    # Find DFF positions
    for dff_name in dff_logical_names:
        if dff_name in placement_map:
            slot_name = placement_map[dff_name]
            if slot_name in slot_lookup:
                slot_info = slot_lookup[slot_name]
                sinks.append({
                    'dff_name': dff_name,
                    'dff_slot': slot_name,
                    'x': slot_info['x'],
                    'y': slot_info['y']
                })
    
    return sinks


def generate_cts_tree(placement_map_path: str,
                     fabric_cells_path: str,
                     design_path: str,
                     tree_type: str = 'h',
                     threshold: int = 4) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Generate CTS tree and update placement map.
    
    This is the main API function that can be called by eco_generator.py.
    
    Args:
        placement_map_path: Path to [design].map file
        fabric_cells_path: Path to fabric_cells.yaml
        design_path: Path to design mapped JSON file (needed to identify DFFs)
        tree_type: 'h' for H-Tree, 'x' for X-Tree (default: 'h')
        threshold: Maximum number of sinks per leaf node (default: 4)
    
    Returns:
        Tuple of (cts_tree_structure, updated_placement_map)
        - cts_tree_structure: Dict keyed by buffer_name
          Format: {buffer_name: {children: [...], sinks: [...], x, y, slot, type, level, ...}}
          Also includes '_metadata' key with tree info and '_root_sinks' if root is a sink
        - updated_placement_map: Dict mapping logical_name -> slot_name
          Includes both original placement and new CTS buffers
    
    Raises:
        ValueError: If tree_type is invalid, no DFFs found, or insufficient buffers
        FileNotFoundError: If input files don't exist
    """
    # Validate tree_type
    tree_type = tree_type.lower()
    if tree_type not in ['h', 'x']:
        raise ValueError(f"Invalid tree_type '{tree_type}'. Must be 'h' (H-Tree) or 'x' (X-Tree)")
    
    # Validate files exist
    if not os.path.exists(placement_map_path):
        raise FileNotFoundError(f"Placement map file not found: {placement_map_path}")
    if not os.path.exists(fabric_cells_path):
        raise FileNotFoundError(f"Fabric cells file not found: {fabric_cells_path}")
    if not os.path.exists(design_path):
        raise FileNotFoundError(f"Design file not found: {design_path}")
    
    # Parse placement map
    placement_map = parse_placement_map(placement_map_path)
    if not placement_map:
        raise ValueError(f"No valid entries found in placement map: {placement_map_path}")
    
    # Parse fabric database
    fabric_db = parse_fabric_cells(fabric_cells_path)
    
    # Identify DFFs from design
    logical_db, netlist_graph = parse_design(design_path)
    dff_type = 'sky130_fd_sc_hd__dfbbp_1'
    dffs = {name: data for name, data in netlist_graph.items() 
            if data.get('type') == dff_type}
    
    if not dffs:
        raise ValueError("No DFF instances found in design")
    
    # Get DFF positions from placement map
    dff_logical_names = list(dffs.keys())
    sinks = get_dff_positions_from_map(placement_map, fabric_db, dff_logical_names)
    
    if not sinks:
        raise ValueError("No DFFs found in placement map. All DFFs must be placed before CTS.")
    
    # Initialize BufferManager
    buffer_manager = BufferManager(fabric_cells_path, placement_map_path)
    available = buffer_manager.get_available_count()
    
    # Validate sufficient buffers
    estimated_required = len(sinks) // threshold
    if available['total'] < estimated_required:
        raise ValueError(
            f"Insufficient buffers: required at least {estimated_required}, "
            f"available {available['total']} (BUF: {available['BUF']}, INV: {available['INV']})"
        )
    
    # Build tree
    root_node = build_tree_with_manager(sinks, buffer_manager, tree_type, level=0, threshold=threshold)
    
    # Convert tree to the required format
    # Format: {buffer_name: {children: [...], sinks: [...], x, y, ...}}
    cts_tree_structure = {}
    
    def extract_tree_structure(node: Dict[str, Any]):
        """Extract tree structure into flat dictionary keyed by buffer_name."""
        if node['type'] == 'buffer':
            buffer_name = node['buffer_name']
            buffer_info = buffer_manager.claimed[buffer_name]
            
            # Get children buffer names
            children = []
            for child in node['children']:
                if child['type'] == 'buffer':
                    children.append(child['buffer_name'])
                    # Recursively process child buffers
                    extract_tree_structure(child)
            
            # Get sinks from leaf nodes
            sinks_list = []
            for child in node['children']:
                if child['type'] == 'sink':
                    sinks_list.extend(child['sinks'])
            
            # Add to tree structure
            cts_tree_structure[buffer_name] = {
                'children': children,
                'sinks': sinks_list,
                'x': node['x'],
                'y': node['y'],
                'slot': buffer_info['slot'],
                'type': buffer_info['type'],
                'level': buffer_info['level']
            }
        else:
            # Sink node - no buffer, just sinks
            pass
    
    # Extract structure starting from root
    extract_tree_structure(root_node)
    
    # Handle case where root is a sink (very few sinks, no buffers needed)
    if root_node['type'] == 'sink':
        cts_tree_structure['_root_sinks'] = root_node['sinks']
    
    # Add metadata
    cts_tree_structure['_metadata'] = {
        'tree_type': 'H-Tree' if tree_type == 'h' else 'X-Tree',
        'num_sinks': len(sinks),
        'num_buffers': buffer_manager.get_claimed_count(),
        'root_buffer': root_node['buffer_name'] if root_node['type'] == 'buffer' else None
    }
    
    # Build updated placement map (original + CTS buffers)
    updated_placement_map = placement_map.copy()
    for buffer_name, claim_data in buffer_manager.claimed.items():
        updated_placement_map[buffer_name] = claim_data['slot']
    
    return cts_tree_structure, updated_placement_map


def main():
    """Example usage of the CTS API."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Minimal API for Clock Tree Synthesis'
    )
    parser.add_argument('--placement-map', required=True,
                        help='Path to placement.map file')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--design', required=True,
                        help='Path to design mapped JSON file')
    parser.add_argument('--tree-type', choices=['h', 'x'], default='h',
                        help='Tree type: h for H-Tree, x for X-Tree (default: h)')
    parser.add_argument('--threshold', type=int, default=4,
                        help='Maximum number of sinks per leaf node (default: 4)')
    parser.add_argument('--output-tree', default='cts_tree.json',
                        help='Output file for CTS tree structure')
    parser.add_argument('--output-map', default='placement_with_cts.map',
                        help='Output file for updated placement map')
    
    args = parser.parse_args()
    
    try:
        # Generate CTS tree
        cts_tree, updated_map = generate_cts_tree(
            args.placement_map,
            args.fabric_cells,
            args.design,
            tree_type=args.tree_type,
            threshold=args.threshold
        )
        
        # Save tree structure
        with open(args.output_tree, 'w') as f:
            json.dump(cts_tree, f, indent=2)
        print(f"CTS tree saved to {args.output_tree}")
        
        # Save updated placement map
        with open(args.output_map, 'w') as f:
            for logical_name in sorted(updated_map.keys()):
                f.write(f"{logical_name} {updated_map[logical_name]}\n")
        print(f"Updated placement map saved to {args.output_map}")
        
        print(f"\nCTS generation completed successfully!")
        print(f"  Tree type: {cts_tree['tree_type']}")
        print(f"  Sinks: {cts_tree['num_sinks']}")
        print(f"  Buffers: {cts_tree['num_buffers']}")
        
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

