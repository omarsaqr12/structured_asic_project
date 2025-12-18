#!/usr/bin/env python3
"""
X-Tree and H-Tree Animation Script: Creates animations of X-Tree and H-Tree construction
by recording frames during tree building and saving to MP4 video.

This script can generate animations for:
- X-Tree (diagonal partitioning: NW/SE and NE/SW splits)
- H-Tree (horizontal/vertical partitioning)
- Both tree types in a single run
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    import imageio
except ImportError:
    imageio = None

# Import tree building functions
from cts_xtree import (
    XTreeNode, synthesize_clock_tree as synthesize_xtree,
    identify_dffs, get_dff_positions, get_available_buffers, build_spatial_index
)
from cts_htree import (
    HTreeNode, synthesize_clock_tree as synthesize_htree
)
from parse_fabric import parse_fabric_cells, parse_pins

# Color mapping
DFF_COLOR = '#FF69B4'  # Hot Pink
BUFFER_COLOR = '#4169E1'  # Royal Blue
ROOT_BUFFER_COLOR = '#FF0000'  # Red
CONNECTION_COLOR = '#0000FF'  # Blue
SINK_CONNECTION_COLOR = '#00FF00'  # Green


class TreeBuilderWithRecording:
    """Wrapper class to build tree with frame recording."""
    
    def __init__(self, tree_type: str, recorded_frames: List, sinks: List[Dict],
                 available_buffers: List[Dict], threshold: int = 4):
        self.tree_type = tree_type.lower()
        self.recorded_frames = recorded_frames
        self.sinks = sinks
        self.available_buffers = available_buffers
        self.threshold = threshold
        self.used_buffers = set()
        self.buffer_nodes = []  # Track all buffer nodes added so far
        
        # Build spatial index
        from cts_xtree import HAS_SCIPY
        if HAS_SCIPY:
            try:
                from scipy.spatial import KDTree
                buffer_coords = [(b['x'], b['y']) for b in available_buffers]
                self.kdtree = KDTree(buffer_coords)
                self.buffer_coords = buffer_coords
            except:
                self.kdtree = None
                self.buffer_coords = None
        else:
            self.kdtree = None
            self.buffer_coords = None
    
    def find_nearest_buffer(self, target_x: float, target_y: float) -> Optional[Dict]:
        """Find nearest available buffer."""
        unused_buffers = [b for b in self.available_buffers if b['name'] not in self.used_buffers]
        if not unused_buffers:
            return None
        
        if self.kdtree is not None:
            from cts_xtree import find_nearest_buffer
            unused_coords = [(b['x'], b['y']) for b in unused_buffers]
            from scipy.spatial import KDTree
            unused_kdtree = KDTree(unused_coords)
            return find_nearest_buffer(target_x, target_y, unused_buffers, unused_kdtree, unused_coords)
        else:
            # Linear search
            best = None
            best_dist = float('inf')
            for buf in unused_buffers:
                dist = ((buf['x'] - target_x)**2 + (buf['y'] - target_y)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best = buf
            return best
    
    def calculate_geometric_center(self, positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate geometric center."""
        if not positions:
            return (0.0, 0.0)
        total_x = sum(x for x, y in positions)
        total_y = sum(y for x, y in positions)
        count = len(positions)
        return (total_x / count, total_y / count)
    
    def partition_sinks_xtree(self, sinks: List[Dict], level: int) -> Tuple[List[Dict], List[Dict]]:
        """Partition sinks for X-tree (diagonal splits)."""
        if len(sinks) <= 1:
            return sinks, []
        
        # Alternate between NW/SE and NE/SW splits
        if level % 2 == 0:
            # NW/SE split: sort by (x - y)
            sorted_sinks = sorted(sinks, key=lambda s: s['x'] - s['y'])
        else:
            # NE/SW split: sort by (x + y)
            sorted_sinks = sorted(sinks, key=lambda s: s['x'] + s['y'])
        
        mid = len(sorted_sinks) // 2
        return sorted_sinks[:mid], sorted_sinks[mid:]
    
    def partition_sinks_htree(self, sinks: List[Dict], level: int) -> Tuple[List[Dict], List[Dict]]:
        """Partition sinks for H-tree (horizontal/vertical splits)."""
        if len(sinks) <= 1:
            return sinks, []
        
        # Alternate between horizontal and vertical splits
        if level % 2 == 0:
            # Horizontal split: sort by y
            sorted_sinks = sorted(sinks, key=lambda s: s['y'])
        else:
            # Vertical split: sort by x
            sorted_sinks = sorted(sinks, key=lambda s: s['x'])
        
        mid = len(sorted_sinks) // 2
        return sorted_sinks[:mid], sorted_sinks[mid:]
    
    def build_tree_with_recording(self, sinks: List[Dict], level: int = 0, 
                                   parent_node: Optional[Any] = None) -> Any:
        """Build tree recursively while recording frames."""
        # Base case: create leaf node
        if len(sinks) <= self.threshold:
            center_x, center_y = self.calculate_geometric_center([(s['x'], s['y']) for s in sinks])
            if self.tree_type == 'x':
                node = XTreeNode('sink', None, center_x, center_y, sinks)
            else:
                node = HTreeNode('sink', None, center_x, center_y, sinks)
            
            # Record frame with new sink node
            self.record_frame(f"Added sink node with {len(sinks)} DFFs")
            return node
        
        # Partition sinks
        if self.tree_type == 'x':
            left_sinks, right_sinks = self.partition_sinks_xtree(sinks, level)
        else:
            left_sinks, right_sinks = self.partition_sinks_htree(sinks, level)
        
        if not left_sinks or not right_sinks:
            # Fallback to leaf node
            center_x, center_y = self.calculate_geometric_center([(s['x'], s['y']) for s in sinks])
            if self.tree_type == 'x':
                node = XTreeNode('sink', None, center_x, center_y, sinks)
            else:
                node = HTreeNode('sink', None, center_x, center_y, sinks)
            self.record_frame(f"Added sink node with {len(sinks)} DFFs (fallback)")
            return node
        
        # Calculate center and find buffer
        all_positions = [(s['x'], s['y']) for s in sinks]
        center_x, center_y = self.calculate_geometric_center(all_positions)
        
        best_buffer = self.find_nearest_buffer(center_x, center_y)
        
        if best_buffer:
            self.used_buffers.add(best_buffer['name'])
            if self.tree_type == 'x':
                buffer_node = XTreeNode('buffer', best_buffer['name'], 
                                       best_buffer['x'], best_buffer['y'])
            else:
                buffer_node = HTreeNode('buffer', best_buffer['name'],
                                       best_buffer['x'], best_buffer['y'])
            self.buffer_nodes.append({
                'node': buffer_node,
                'level': level,
                'parent': parent_node
            })
            
            # Record frame with new buffer
            self.record_frame(f"Added buffer at level {level}")
        else:
            # No buffer available
            if self.tree_type == 'x':
                buffer_node = XTreeNode('buffer', None, center_x, center_y)
            else:
                buffer_node = HTreeNode('buffer', None, center_x, center_y)
            self.record_frame(f"Added buffer node (no physical buffer) at level {level}")
        
        # Recursively build children
        left_child = self.build_tree_with_recording(left_sinks, level + 1, buffer_node)
        right_child = self.build_tree_with_recording(right_sinks, level + 1, buffer_node)
        
        buffer_node.children = [left_child, right_child]
        
        # Record frame with children connected
        self.record_frame(f"Connected children at level {level}")
        
        return buffer_node
    
    def record_frame(self, description: str):
        """Record current tree state as a frame."""
        # Create a snapshot of current tree state
        frame_data = {
            'description': description,
            'buffer_nodes': [{
                'x': bn['node'].x,
                'y': bn['node'].y,
                'level': bn['level'],
                'buffer_slot': bn['node'].buffer_slot,
                'has_parent': bn['parent'] is not None,
                'parent_x': bn['parent'].x if bn['parent'] else None,
                'parent_y': bn['parent'].y if bn['parent'] else None
            } for bn in self.buffer_nodes],
            'sinks': self.sinks,
            'used_buffers': list(self.used_buffers)
        }
        self.recorded_frames.append(frame_data)


def create_tree_frame(frame_data: Dict[str, Any], pins_db: Dict[str, Any],
                     tree_type: str, frame_num: int, total_frames: int) -> np.ndarray:
    """Create a single frame showing current tree state."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    # Get die and core dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    core_x = core_margin
    core_y = core_margin
    
    # Draw die boundary
    die_rect = patches.Rectangle(
        (0, 0), die_width, die_height,
        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
    )
    ax.add_patch(die_rect)
    
    # Draw core boundary
    core_rect = patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
    )
    ax.add_patch(core_rect)
    
    # Draw all DFFs (sinks)
    sinks = frame_data.get('sinks', [])
    for sink in sinks:
        dff_circle = plt.Circle((sink['x'], sink['y']), radius=2, 
                               color=DFF_COLOR, alpha=0.6, zorder=3)
        ax.add_patch(dff_circle)
    
    # Draw buffer nodes and connections
    buffer_nodes = frame_data.get('buffer_nodes', [])
    for i, buf_data in enumerate(buffer_nodes):
        x = buf_data['x']
        y = buf_data['y']
        level = buf_data['level']
        
        # Draw connection from parent
        if buf_data['has_parent'] and buf_data['parent_x'] is not None:
            ax.plot([buf_data['parent_x'], x], [buf_data['parent_y'], y],
                   color=CONNECTION_COLOR, linewidth=2, alpha=0.7, zorder=2)
        
        # Draw buffer (red if root/level 0, blue otherwise)
        if level == 0:
            buffer_rect = patches.Rectangle(
                (x - 5, y - 5), 10, 10,
                linewidth=2, edgecolor=ROOT_BUFFER_COLOR, 
                facecolor=ROOT_BUFFER_COLOR, alpha=0.8, zorder=5
            )
            ax.add_patch(buffer_rect)
            ax.text(x, y + 8, 'ROOT', fontsize=10, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7), zorder=6)
        else:
            buffer_rect = patches.Rectangle(
                (x - 3, y - 3), 6, 6,
                linewidth=1.5, edgecolor=BUFFER_COLOR, 
                facecolor=BUFFER_COLOR, alpha=0.7, zorder=4
            )
            ax.add_patch(buffer_rect)
    
    # Set axis properties
    ax.set_xlim(0, die_width)
    ax.set_ylim(0, die_height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    
    # Add title
    tree_name = 'X-Tree' if tree_type.lower() == 'x' else 'H-Tree'
    title = f'{tree_name} Clock Tree Synthesis Animation\n'
    title += f'{frame_data.get("description", "")}\n'
    title += f'Frame {frame_num + 1}/{total_frames} | Buffers: {len(buffer_nodes)} | DFFs: {len(sinks)}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    if imageio is None:
        raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")
    try:
        import imageio.v2 as imageio_v2
        img = imageio_v2.imread(buf)
    except ImportError:
        img = imageio.imread(buf)
    buf.close()
    plt.close(fig)
    return img


def animate_tree(design_name: str,
                 tree_type: str,
                 placement_path: Optional[str] = None,
                 fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                 pins_path: str = 'fabric/pins.yaml',
                 output_path: Optional[str] = None,
                 fps: int = 5,
                 threshold: int = 4) -> bool:
    """
    Create an animation of X-Tree or H-Tree clock tree construction.
    
    Args:
        design_name: Name of the design (e.g., '6502')
        tree_type: 'h' for H-Tree, 'x' for X-Tree
        placement_path: Path to placement JSON file
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        output_path: Path to save MP4 video
        fps: Frames per second for video (default: 5)
        threshold: Maximum number of sinks per leaf node (default: 4)
    
    Returns:
        True if successful, False otherwise
    """
    design_path = f'designs/{design_name}_mapped.json'
    
    if not os.path.exists(design_path):
        print(f"ERROR: Design file not found: {design_path}")
        return False
    
    # Determine placement path
    if placement_path is None:
        # Try to use the "best" placement file first (used for CTS tree images)
        best_placement = f'build/{design_name}/Best_sa_alpha0.99_moves1000_Tfinal0.001/{design_name}_placement.json'
        default_placement = f'build/{design_name}/{design_name}_placement.json'
        
        if os.path.exists(best_placement):
            placement_path = best_placement
            print(f"Using best placement file: {placement_path}")
        elif os.path.exists(default_placement):
            placement_path = default_placement
            print(f"Using default placement file: {placement_path}")
        else:
            print(f"ERROR: No placement file found. Tried:")
            print(f"  - {best_placement}")
            print(f"  - {default_placement}")
            return False
    
    if not os.path.exists(placement_path):
        print(f"ERROR: Placement file not found: {placement_path}")
        return False
    
    # Validate tree type
    tree_type = tree_type.lower()
    if tree_type not in ['h', 'x']:
        print(f"ERROR: Invalid tree_type '{tree_type}'. Must be 'h' or 'x'")
        return False
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Creating {tree_type.upper()}-Tree Animation for {design_name}")
    print(f"{'='*60}")
    print("Loading placement and design data...")
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
        return False
    
    # Get available buffers
    print("Finding available buffers...")
    available_buffers = get_available_buffers(fabric_cells_path, placement)
    print(f"Found {len(available_buffers)} available buffers/inverters")
    
    if not available_buffers:
        print("ERROR: No available buffers found")
        return False
    
    # Parse pins
    print("Loading pins data...")
    pins_db = parse_pins(pins_path)
    
    # Initialize frame recording
    recorded_frames = []
    
    # Record initial frame (just DFFs)
    recorded_frames.append({
        'description': 'Initial state: DFFs only',
        'buffer_nodes': [],
        'sinks': sinks,
        'used_buffers': []
    })
    
    # Build tree with recording
    print(f"\nBuilding {tree_type.upper()}-Tree with frame recording...")
    builder = TreeBuilderWithRecording(tree_type, recorded_frames, sinks, 
                                      available_buffers, threshold)
    root = builder.build_tree_with_recording(sinks, level=0)
    
    # Record final frame
    # Reconstruct full tree structure for final frame
    def collect_all_nodes(node, parent=None, level=0):
        """Collect all nodes from tree."""
        nodes = []
        if node.type == 'buffer':
            nodes.append({
                'x': node.x,
                'y': node.y,
                'level': level,
                'buffer_slot': node.buffer_slot,
                'has_parent': parent is not None,
                'parent_x': parent.x if parent else None,
                'parent_y': parent.y if parent else None
            })
            for child in node.children:
                nodes.extend(collect_all_nodes(child, node, level + 1))
        return nodes
    
    all_buffer_nodes = collect_all_nodes(root)
    recorded_frames.append({
        'description': 'Final tree structure',
        'buffer_nodes': all_buffer_nodes,
        'sinks': sinks,
        'used_buffers': list(builder.used_buffers)
    })
    
    print(f"Recorded {len(recorded_frames)} frames")
    
    # Determine output path
    if output_path is None:
        output_dir = f'build/{design_name}'
        os.makedirs(output_dir, exist_ok=True)
        tree_name = 'xtree' if tree_type == 'x' else 'htree'
        output_path = f'{output_dir}/{design_name}_cts_{tree_name}_animation.mp4'
    
    # Ensure output path has .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    
    # Check for imageio
    if imageio is None:
        print("ERROR: imageio required for video export.")
        print("Install with: pip install imageio imageio-ffmpeg")
        return False
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate frames
    print(f"\nGenerating {len(recorded_frames)} frames...")
    frames_images = []
    for i, frame_data in enumerate(recorded_frames):
        frame_img = create_tree_frame(frame_data, pins_db, tree_type, i, len(recorded_frames))
        frames_images.append(frame_img)
        
        if (i + 1) % 5 == 0 or i == len(recorded_frames) - 1:
            pct = int((i + 1) / len(recorded_frames) * 100)
            print(f"  Generated {i + 1}/{len(recorded_frames)} frames ({pct}%)", end='\r')
    
    print()  # New line
    
    # Save video
    print("Writing video file...")
    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        for frame in frames_images:
            writer.append_data(frame)
        writer.close()
        
        print(f"\n✓ Video saved successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Error saving video: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure imageio-ffmpeg is installed: pip install imageio-ffmpeg")
        print("2. Ensure ffmpeg is installed on your system")
        import traceback
        traceback.print_exc()
        return False


def animate_xh_trees(design_name: str,
                     tree_types: List[str] = ['h', 'x'],
                     placement_path: Optional[str] = None,
                     fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                     pins_path: str = 'fabric/pins.yaml',
                     output_dir: Optional[str] = None,
                     fps: int = 5,
                     threshold: int = 4):
    """
    Create animations for X-Tree and/or H-Tree clock tree construction.
    
    Args:
        design_name: Name of the design (e.g., '6502')
        tree_types: List of tree types to animate: ['h'], ['x'], or ['h', 'x'] (default: ['h', 'x'])
        placement_path: Path to placement JSON file
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        output_dir: Directory to save MP4 videos (default: build/{design})
        fps: Frames per second for video (default: 5)
        threshold: Maximum number of sinks per leaf node (default: 4)
    """
    # Validate tree types
    valid_types = {'h', 'x'}
    tree_types = [t.lower() for t in tree_types]
    tree_types = [t for t in tree_types if t in valid_types]
    
    if not tree_types:
        print("ERROR: No valid tree types specified. Must include 'h' and/or 'x'")
        return
    
    # Determine output directory
    if output_dir is None:
        output_dir = f'build/{design_name}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Animate each tree type
    results = {}
    for tree_type in tree_types:
        output_path = None  # Let animate_tree determine the path
        success = animate_tree(
            design_name=design_name,
            tree_type=tree_type,
            placement_path=placement_path,
            fabric_cells_path=fabric_cells_path,
            pins_path=pins_path,
            output_path=output_path,
            fps=fps,
            threshold=threshold
        )
        results[tree_type] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("Animation Summary")
    print(f"{'='*60}")
    for tree_type, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{tree_type.upper()}-Tree: {status}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create animations of X-Tree and/or H-Tree clock tree synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Animate both X-Tree and H-Tree:
  python animate_xh_trees.py --design 6502
  
  # Animate only H-Tree:
  python animate_xh_trees.py --design 6502 --tree-types h
  
  # Animate only X-Tree:
  python animate_xh_trees.py --design 6502 --tree-types x
  
  # Custom output directory and FPS:
  python animate_xh_trees.py --design 6502 --output-dir build/custom --fps 10
        """
    )
    parser.add_argument('--design', type=str, required=True,
                       help='Design name (e.g., 6502)')
    parser.add_argument('--tree-types', type=str, nargs='+', default=['h', 'x'],
                       choices=['h', 'x'],
                       help='Tree types to animate: h (H-Tree), x (X-Tree), or both (default: h x)')
    parser.add_argument('--placement', type=str, default=None,
                       help='Path to placement JSON file (default: build/{design}/{design}_placement.json)')
    parser.add_argument('--fabric-cells', type=str, default='fabric/fabric_cells.yaml',
                       help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', type=str, default='fabric/pins.yaml',
                       help='Path to pins.yaml')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for MP4 videos (default: build/{design})')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for video (default: 5)')
    parser.add_argument('--threshold', type=int, default=4,
                       help='Maximum number of sinks per leaf node (default: 4)')
    
    args = parser.parse_args()
    
    animate_xh_trees(
        args.design,
        args.tree_types,
        args.placement,
        args.fabric_cells,
        args.pins,
        args.output_dir,
        args.fps,
        args.threshold
    )


if __name__ == '__main__':
    main()
