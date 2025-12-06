#!/usr/bin/env python3
"""
Example: Modified H-Tree CTS using BufferManager

This shows how to integrate BufferManager into the existing CTS code.
"""

import sys
import json
import math
from typing import Dict, List, Tuple, Any, Optional
from buffer_manager import BufferManager
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


def partition_sinks(sinks: List[Dict[str, Any]], level: int) -> Tuple[List[Dict], List[Dict]]:
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


class HTreeNode:
    """Represents a node in the H-Tree structure."""
    def __init__(self, node_type: str, buffer_name: Optional[str] = None,
                 x: float = 0.0, y: float = 0.0, sinks: List[Dict] = None):
        self.type = node_type  # 'buffer' or 'sink'
        self.buffer_name = buffer_name  # Logical buffer name from BufferManager
        self.x = x
        self.y = y
        self.sinks = sinks or []
        self.children = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization."""
        result = {
            'type': self.type,
            'x': self.x,
            'y': self.y
        }
        
        if self.type == 'buffer':
            result['buffer_name'] = self.buffer_name
            result['children'] = [child.to_dict() for child in self.children]
        else:  # sink (leaf)
            result['sinks'] = self.sinks
        
        return result


def build_htree_with_manager(sinks: List[Dict[str, Any]],
                            buffer_manager: BufferManager,
                            level: int = 0,
                            threshold: int = 4) -> HTreeNode:
    """
    Recursively build H-Tree structure using BufferManager.
    
    Args:
        sinks: List of DFF sink dictionaries
        buffer_manager: BufferManager instance
        level: Current recursion level
        threshold: Maximum number of sinks per leaf node
    
    Returns:
        HTreeNode representing the tree structure
    """
    # Base case: create leaf node if sinks <= threshold
    if len(sinks) <= threshold:
        center_x, center_y = calculate_geometric_center([(s['x'], s['y']) for s in sinks])
        return HTreeNode(
            node_type='sink',
            x=center_x,
            y=center_y,
            sinks=sinks
        )
    
    # Partition sinks
    left_sinks, right_sinks = partition_sinks(sinks, level)
    
    if not left_sinks or not right_sinks:
        # If partition failed, create leaf node
        center_x, center_y = calculate_geometric_center([(s['x'], s['y']) for s in sinks])
        return HTreeNode(
            node_type='sink',
            x=center_x,
            y=center_y,
            sinks=sinks
        )
    
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
        print(f"ERROR: Failed to claim buffer: {e}")
        # Fallback: create internal node without buffer
        buffer_node = HTreeNode(
            node_type='buffer',
            buffer_name=None,
            x=center_x,
            y=center_y
        )
        # Still build children
        left_child = build_htree_with_manager(
            left_sinks, buffer_manager, level + 1, threshold
        )
        right_child = build_htree_with_manager(
            right_sinks, buffer_manager, level + 1, threshold
        )
        buffer_node.children = [left_child, right_child]
        return buffer_node
    
    # Get buffer position from manager
    buffer_info = buffer_manager.claimed[buffer_name]
    buffer_x = buffer_info['x']
    buffer_y = buffer_info['y']
    
    # Recursively build children
    left_child = build_htree_with_manager(
        left_sinks, buffer_manager, level + 1, threshold
    )
    right_child = build_htree_with_manager(
        right_sinks, buffer_manager, level + 1, threshold
    )
    
    # Update buffer's children list in manager
    child_buffer_names = []
    if left_child.type == 'buffer' and left_child.buffer_name:
        child_buffer_names.append(left_child.buffer_name)
    if right_child.type == 'buffer' and right_child.buffer_name:
        child_buffer_names.append(right_child.buffer_name)
    
    buffer_manager.claimed[buffer_name]['children'] = child_buffer_names
    
    # Create buffer node
    buffer_node = HTreeNode(
        node_type='buffer',
        buffer_name=buffer_name,
        x=buffer_x,
        y=buffer_y
    )
    buffer_node.children = [left_child, right_child]
    
    return buffer_node


def synthesize_clock_tree_with_manager(placement_path: str,
                                      design_path: str,
                                      fabric_cells_path: str,
                                      placement_map_path: Optional[str] = None,
                                      threshold: int = 4) -> Tuple[HTreeNode, BufferManager]:
    """
    Synthesize H-Tree clock tree using BufferManager.
    
    Args:
        placement_path: Path to placement JSON file
        design_path: Path to design mapped JSON file
        fabric_cells_path: Path to fabric_cells.yaml
        placement_map_path: Optional path to placement.map file
        threshold: Maximum number of sinks per leaf node
    
    Returns:
        Tuple of (root_node, buffer_manager)
    """
    # Load placement
    print("Loading placement...")
    with open(placement_path, 'r') as f:
        placement = json.load(f)
    
    # Identify DFFs
    print("Identifying DFFs...")
    logical_db, netlist_graph = parse_design(design_path)
    dff_type = 'sky130_fd_sc_hd__dfbbp_1'
    dffs = {name: data for name, data in netlist_graph.items() 
            if data.get('type') == dff_type}
    print(f"Found {len(dffs)} DFF instances")
    
    # Get DFF positions
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
    print(f"Found {len(sinks)} placed DFFs")
    
    if not sinks:
        print("ERROR: No DFFs found in placement")
        return None, None
    
    # Initialize BufferManager
    print("Initializing BufferManager...")
    buffer_manager = BufferManager(fabric_cells_path, placement_map_path)
    available = buffer_manager.get_available_count()
    print(f"Available buffers: {available['BUF']} BUF, {available['INV']} INV, "
          f"{available['total']} total")
    
    # Validate sufficient buffers
    estimated_required = len(sinks) // threshold
    try:
        buffer_manager.validate_sufficient_buffers(estimated_required)
    except ValueError as e:
        print(f"WARNING: {e}")
    
    # Build H-Tree
    print("Building H-Tree structure...")
    root = build_htree_with_manager(sinks, buffer_manager, level=0, threshold=threshold)
    
    print(f"Used {buffer_manager.get_claimed_count()} buffers for clock tree")
    
    return root, buffer_manager


def main():
    """Example usage of CTS with BufferManager."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='H-Tree CTS using BufferManager'
    )
    parser.add_argument('--placement', required=True,
                        help='Path to placement JSON file')
    parser.add_argument('--design', required=True,
                        help='Path to design mapped JSON file')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--placement-map', default=None,
                        help='Path to placement.map file (optional)')
    parser.add_argument('--output-tree', default='cts_tree_with_manager.json',
                        help='Output file for CTS tree structure')
    parser.add_argument('--output-claims', default='claimed_buffers.json',
                        help='Output file for claimed buffers')
    parser.add_argument('--update-map', default=None,
                        help='Path to placement.map file to update (optional)')
    parser.add_argument('--threshold', type=int, default=4,
                        help='Maximum number of sinks per leaf node (default: 4)')
    
    args = parser.parse_args()
    
    # Synthesize clock tree
    root, buffer_manager = synthesize_clock_tree_with_manager(
        args.placement, args.design, args.fabric_cells, args.placement_map, args.threshold
    )
    
    if root is None:
        print("ERROR: Failed to synthesize clock tree")
        sys.exit(1)
    
    # Save tree structure
    tree_dict = {
        'root': root.to_dict(),
        'sinks': []
    }
    
    def collect_sinks(node: HTreeNode):
        if node.type == 'sink':
            tree_dict['sinks'].extend(node.sinks)
        else:
            for child in node.children:
                collect_sinks(child)
    
    collect_sinks(root)
    
    with open(args.output_tree, 'w') as f:
        json.dump(tree_dict, f, indent=2)
    print(f"CTS tree saved to {args.output_tree}")
    
    # Export claimed buffers
    buffer_manager.export_claims(args.output_claims)
    
    # Update placement map if requested
    if args.update_map:
        buffer_manager.update_placement_map(args.update_map, append=True)
    
    print("\nCTS synthesis completed successfully!")


if __name__ == '__main__':
    main()

