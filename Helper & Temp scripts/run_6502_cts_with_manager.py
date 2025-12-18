#!/usr/bin/env python3
"""
Run CTS with BufferManager for 6502 design - both H-Tree and X-Tree
for greedy and best SA placements
"""
import os
import sys
import json
import math
from typing import Dict, List, Tuple, Any, Optional
from buffer_manager import BufferManager
from parse_design import parse_design


# Configuration
design_file = "designs/6502_mapped.json"
fabric_cells = "fabric/fabric_cells.yaml"

# Placement configurations
placements = {
    'greedy': {
        'placement_json': 'build/6502/greedy/6502_placement.json',
        'placement_map': 'build/6502/greedy/6502.map',
    },
    'best': {
        'placement_json': 'build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/6502_placement.json',
        'placement_map': 'build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/6502.map',
    }
}

# Output directory
base_output_dir = 'build/6502/cts_with_manager'


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


class TreeNode:
    """Represents a node in the tree structure (works for both H-Tree and X-Tree)."""
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


def build_tree_with_manager(sinks: List[Dict[str, Any]],
                           buffer_manager: BufferManager,
                           tree_type: str,  # 'htree' or 'xtree'
                           level: int = 0,
                           threshold: int = 4) -> TreeNode:
    """
    Recursively build tree structure using BufferManager.
    
    Args:
        sinks: List of DFF sink dictionaries
        buffer_manager: BufferManager instance
        tree_type: 'htree' or 'xtree'
        level: Current recursion level
        threshold: Maximum number of sinks per leaf node
    
    Returns:
        TreeNode representing the tree structure
    """
    # Base case: create leaf node if sinks <= threshold
    if len(sinks) <= threshold:
        center_x, center_y = calculate_geometric_center([(s['x'], s['y']) for s in sinks])
        return TreeNode(
            node_type='sink',
            x=center_x,
            y=center_y,
            sinks=sinks
        )
    
    # Partition sinks based on tree type
    if tree_type == 'htree':
        left_sinks, right_sinks = partition_sinks_htree(sinks, level)
    else:  # xtree
        left_sinks, right_sinks = partition_sinks_xtree(sinks, level)
    
    if not left_sinks or not right_sinks:
        # If partition failed, create leaf node
        center_x, center_y = calculate_geometric_center([(s['x'], s['y']) for s in sinks])
        return TreeNode(
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
        buffer_node = TreeNode(
            node_type='buffer',
            buffer_name=None,
            x=center_x,
            y=center_y
        )
        # Still build children
        left_child = build_tree_with_manager(
            left_sinks, buffer_manager, tree_type, level + 1, threshold
        )
        right_child = build_tree_with_manager(
            right_sinks, buffer_manager, tree_type, level + 1, threshold
        )
        buffer_node.children = [left_child, right_child]
        return buffer_node
    
    # Get buffer position from manager
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
    if left_child.type == 'buffer' and left_child.buffer_name:
        child_buffer_names.append(left_child.buffer_name)
    if right_child.type == 'buffer' and right_child.buffer_name:
        child_buffer_names.append(right_child.buffer_name)
    
    buffer_manager.claimed[buffer_name]['children'] = child_buffer_names
    
    # Create buffer node
    buffer_node = TreeNode(
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
                                      placement_map_path: Optional[str],
                                      tree_type: str,
                                      threshold: int = 4) -> Tuple[TreeNode, BufferManager]:
    """
    Synthesize clock tree using BufferManager.
    
    Args:
        placement_path: Path to placement JSON file
        design_path: Path to design mapped JSON file
        fabric_cells_path: Path to fabric_cells.yaml
        placement_map_path: Path to placement.map file
        tree_type: 'htree' or 'xtree'
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
    
    # Build tree
    tree_name = "H-Tree" if tree_type == 'htree' else "X-Tree"
    print(f"Building {tree_name} structure...")
    root = build_tree_with_manager(sinks, buffer_manager, tree_type, level=0, threshold=threshold)
    
    print(f"Used {buffer_manager.get_claimed_count()} buffers for clock tree")
    
    return root, buffer_manager


def save_cts_tree(root: TreeNode, output_path: str):
    """Save CTS tree structure to JSON file."""
    tree_dict = {
        'root': root.to_dict(),
        'sinks': []
    }
    
    # Collect all sinks from the tree
    def collect_sinks(node: TreeNode):
        if node.type == 'sink':
            tree_dict['sinks'].extend(node.sinks)
        else:
            for child in node.children:
                collect_sinks(child)
    
    collect_sinks(root)
    
    with open(output_path, 'w') as f:
        json.dump(tree_dict, f, indent=2)
    
    print(f"CTS tree saved to {output_path}")


def run_cts_for_placement_and_tree(placement_name: str, placement_config: dict, 
                                    tree_type: str, threshold: int = 4):
    """Run CTS for a specific placement and tree type."""
    tree_name = "H-Tree" if tree_type == 'htree' else "X-Tree"
    print(f"\n{'=' * 60}")
    print(f"Running {tree_name} CTS with BufferManager for {placement_name.upper()} placement")
    print(f"{'=' * 60}")
    
    placement_json = placement_config['placement_json']
    placement_map = placement_config['placement_map']
    
    # Check if placement files exist
    if not os.path.exists(placement_json):
        print(f"ERROR: Placement file {placement_json} not found!")
        return False
    
    if not os.path.exists(placement_map):
        print(f"WARNING: Placement map file {placement_map} not found!")
        placement_map = None
    
    # Create output directory
    output_dir = f"{base_output_dir}/{placement_name}_{tree_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Synthesize clock tree
    root, buffer_manager = synthesize_clock_tree_with_manager(
        placement_json,
        design_file,
        fabric_cells,
        placement_map,
        tree_type,
        threshold
    )
    
    if root is None:
        print(f"ERROR: CTS synthesis failed for {placement_name} {tree_name}")
        return False
    
    # Save outputs
    save_cts_tree(root, f"{output_dir}/6502_cts_tree.json")
    
    # Export claimed buffers
    buffer_manager.export_claims(f"{output_dir}/claimed_buffers.json")
    
    # Update placement map (create new file, don't modify original)
    placement_map_updated = f"{output_dir}/6502_with_cts.map"
    if placement_map:
        # Copy original if it exists
        import shutil
        shutil.copy(placement_map, placement_map_updated)
        buffer_manager.update_placement_map(placement_map_updated, append=True)
    else:
        # Create new file
        buffer_manager.update_placement_map(placement_map_updated, append=False)
    
    print(f"✓ Created updated placement.map: {placement_map_updated}")
    print(f"\n✓ {placement_name.upper()} {tree_name} CTS completed successfully!")
    print(f"  Output directory: {output_dir}")
    return True


# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("Running CTS with BufferManager for 6502 design")
    print("H-Tree and X-Tree for both Greedy and Best placements")
    print("=" * 60)
    
    results = {}
    
    # Run all combinations
    for placement_name, placement_config in placements.items():
        for tree_type in ['htree', 'xtree']:
            key = f"{placement_name}_{tree_type}"
            results[key] = run_cts_for_placement_and_tree(
                placement_name, placement_config, tree_type, threshold=4
            )
    
    # Summary
    print("\n" + "=" * 60)
    print("CTS Synthesis Summary")
    print("=" * 60)
    for key, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {key.upper()}: {status}")
    
    if all(results.values()):
        print("\n✓ All CTS runs completed successfully!")
        print(f"  Output directory: {base_output_dir}")
        sys.exit(0)
    else:
        print("\n✗ Some CTS runs failed!")
        sys.exit(1)

