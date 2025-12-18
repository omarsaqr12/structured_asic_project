#!/usr/bin/env python3
"""
Clock Skew Visualization Animation: Creates visualizations showing clock arrival times
at each DFF, color-coded by delay, highlighting critical paths and skew violations.

This script visualizes:
- DFFs color-coded by clock arrival time (delay from root)
- Clock tree structure (buffers and connections)
- Critical paths (longest delays)
- Skew violations (if any)
- Clock tree building animation (optional)
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
from matplotlib.colors import LinearSegmentedColormap

try:
    import imageio
except ImportError:
    imageio = None

from parse_fabric import parse_fabric_cells, parse_pins
from cts_simulator import simulate_cts_tree, trace_path_to_sinks, get_buffer_delay, calculate_wire_delay, calculate_distance

# Color mapping
DFF_COLOR = '#FF69B4'  # Hot Pink (default)
BUFFER_COLOR = '#4169E1'  # Royal Blue
ROOT_BUFFER_COLOR = '#FF0000'  # Red
CONNECTION_COLOR = '#0000FF'  # Blue
CRITICAL_PATH_COLOR = '#FF0000'  # Red for critical paths


def create_clock_skew_frame(cts_tree: Dict[str, Any],
                            sink_paths: List[Dict[str, Any]],
                            pins_db: Dict[str, Any],
                            phase: str,
                            tree_type: str = 'h',
                            claimed_buffers_path: Optional[str] = None,
                            show_tree: bool = True,
                            show_dffs: bool = True,
                            show_buffers: bool = True,
                            highlight_critical: bool = True,
                            skew_threshold: Optional[float] = None) -> np.ndarray:
    """
    Create a single frame showing clock skew visualization.
    
    Args:
        cts_tree: CTS tree dictionary (with 'root' key)
        sink_paths: List of path info from simulate_cts_tree (or None to calculate)
        pins_db: Pins database
        phase: Description of current phase
        tree_type: 'h' for H-Tree, 'x' for X-Tree
        claimed_buffers_path: Path to claimed_buffers.json for buffer types
        show_tree: Whether to draw clock tree structure
        show_dffs: Whether to draw DFFs
        show_buffers: Whether to draw buffers
        highlight_critical: Whether to highlight critical paths
        skew_threshold: Optional skew threshold to highlight violations (ns)
    
    Returns:
        Image as numpy array
    """
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
    
    # Background for core area
    ax.add_patch(patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=0, facecolor='lightgray', alpha=0.1
    ))
    
    # Calculate sink paths if not provided
    if sink_paths is None:
        # Load buffer types
        buffer_types = {}
        if claimed_buffers_path and os.path.exists(claimed_buffers_path):
            with open(claimed_buffers_path, 'r') as f:
                claimed_buffers = json.load(f)
                for buf_name, buf_info in claimed_buffers.items():
                    buffer_types[buf_name] = buf_info.get('type', 'BUF')
        
        root = cts_tree.get('root')
        if root:
            sink_paths = trace_path_to_sinks(root, 0.0, 0.0, 0, buffer_types, None, None, None)
    
    if not sink_paths:
        # Fallback: no timing data available
        ax.text(die_width/2, die_height/2, 'No timing data available', 
               ha='center', va='center', fontsize=16)
        ax.set_xlim(0, die_width)
        ax.set_ylim(0, die_height)
        ax.set_aspect('equal')
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
    
    # Calculate timing metrics
    delays = [path['delay'] for path in sink_paths]
    max_delay = max(delays)
    min_delay = min(delays)
    skew = max_delay - min_delay
    avg_delay = sum(delays) / len(delays) if delays else 0.0
    
    # Create colormap: blue (early/fast) -> green -> yellow -> red (late/slow)
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']  # Blue -> Green -> Yellow -> Red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('delay', colors, N=n_bins)
    
    # Draw clock tree structure
    if show_tree:
        root = cts_tree.get('root')
        if root:
            def draw_tree_node(node, parent_x=None, parent_y=None, level=0):
                """Recursively draw clock tree."""
                if node['type'] == 'buffer':
                    x = node['x']
                    y = node['y']
                    
                    # Draw connection from parent
                    if parent_x is not None and parent_y is not None:
                        ax.plot([parent_x, x], [parent_y, y],
                               color=CONNECTION_COLOR, linewidth=2, alpha=0.7, zorder=2)
                    
                    # Draw buffer
                    if level == 0:
                        # Root buffer
                        buffer_rect = patches.Rectangle(
                            (x - 5, y - 5), 10, 10,
                            linewidth=2, edgecolor=ROOT_BUFFER_COLOR,
                            facecolor=ROOT_BUFFER_COLOR, alpha=0.8, zorder=5
                        )
                        ax.add_patch(buffer_rect)
                        ax.text(x, y + 8, 'ROOT', fontsize=10, ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7), zorder=6)
                    elif show_buffers:
                        buffer_rect = patches.Rectangle(
                            (x - 3, y - 3), 6, 6,
                            linewidth=1.5, edgecolor=BUFFER_COLOR,
                            facecolor=BUFFER_COLOR, alpha=0.7, zorder=4
                        )
                        ax.add_patch(buffer_rect)
                    
                    # Draw children
                    for child in node.get('children', []):
                        draw_tree_node(child, x, y, level + 1)
                
                elif node['type'] == 'sink':
                    sink_x = node['x']
                    sink_y = node['y']
                    
                    # Draw connection from parent
                    if parent_x is not None and parent_y is not None:
                        ax.plot([parent_x, sink_x], [parent_y, sink_y],
                               color=CONNECTION_COLOR, linewidth=1.5, alpha=0.5, zorder=2)
            
            draw_tree_node(root)
    
    # Draw DFFs color-coded by arrival time
    if show_dffs:
        # Create delay-to-color mapping
        delay_range = max_delay - min_delay if max_delay > min_delay else 1.0
        
        for path in sink_paths:
            dff_x = path['dff_x']
            dff_y = path['dff_y']
            delay = path['delay']
            
            # Normalize delay for color mapping
            if delay_range > 0:
                normalized_delay = (delay - min_delay) / delay_range
            else:
                normalized_delay = 0.5
            
            # Get color from colormap
            color = cmap(normalized_delay)
            
            # Highlight critical paths (longest delays)
            is_critical = highlight_critical and delay >= max_delay * 0.95
            if is_critical:
                # Draw with thicker border for critical paths
                dff_circle = plt.Circle((dff_x, dff_y), radius=4,
                                       color=color, alpha=0.9, zorder=5,
                                       edgecolor='red', linewidth=2)
            else:
                dff_circle = plt.Circle((dff_x, dff_y), radius=3,
                                       color=color, alpha=0.7, zorder=3)
            ax.add_patch(dff_circle)
    
    # Highlight skew violations if threshold provided
    if skew_threshold is not None and skew > skew_threshold:
        # Draw warning box
        warning_text = f'⚠ SKEW VIOLATION: {skew:.3f}ns > {skew_threshold:.3f}ns'
        ax.text(die_width/2, die_height - 50, warning_text,
               ha='center', va='top', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
               zorder=10)
    
    # Set axis properties
    ax.set_xlim(0, die_width)
    ax.set_ylim(0, die_height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    
    # Add title with timing metrics
    tree_name = 'H-Tree' if tree_type.lower() == 'h' else 'X-Tree'
    title = f'Clock Skew Visualization - {tree_name} - {phase}\n'
    title += f'Skew: {skew:.3f} ns | Max Delay: {max_delay:.3f} ns | Min Delay: {min_delay:.3f} ns\n'
    title += f'Avg Delay: {avg_delay:.3f} ns | DFFs: {len(sink_paths)}'
    if skew_threshold is not None:
        status = '✓ PASS' if skew <= skew_threshold else '✗ FAIL'
        title += f' | Skew Check ({skew_threshold:.3f}ns): {status}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_delay, vmax=max_delay))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Clock Arrival Time (ns)', rotation=270, labelpad=20, fontsize=10)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Early Arrival (Fast)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Late Arrival (Slow)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=ROOT_BUFFER_COLOR, markersize=10, label='Root Buffer'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=BUFFER_COLOR, markersize=8, label='Clock Buffer'),
    ]
    if highlight_critical:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='red', markerfacecolor='w', markersize=12, 
                  markeredgewidth=2, label='Critical Path (Top 5%)')
        )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
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


def animate_clock_skew_static(design_name: str,
                              cts_tree_path: str,
                              pins_path: str = 'fabric/pins.yaml',
                              claimed_buffers_path: Optional[str] = None,
                              output_path: Optional[str] = None,
                              tree_type: str = 'h',
                              show_tree: bool = True,
                              show_dffs: bool = True,
                              show_buffers: bool = True,
                              highlight_critical: bool = True,
                              skew_threshold: Optional[float] = None) -> bool:
    """
    Create a static visualization of clock skew.
    
    Args:
        design_name: Name of the design (e.g., '6502')
        cts_tree_path: Path to CTS tree JSON file
        pins_path: Path to pins.yaml
        claimed_buffers_path: Path to claimed_buffers.json (optional)
        output_path: Path to save PNG image
        tree_type: 'h' for H-Tree, 'x' for X-Tree
        show_tree: Whether to draw clock tree structure
        show_dffs: Whether to draw DFFs
        show_buffers: Whether to draw buffers
        highlight_critical: Whether to highlight critical paths
        skew_threshold: Optional skew threshold to highlight violations (ns)
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(cts_tree_path):
        print(f"ERROR: CTS tree file not found: {cts_tree_path}")
        return False
    
    # Load CTS tree
    print(f"Loading CTS tree from: {cts_tree_path}")
    with open(cts_tree_path, 'r') as f:
        cts_tree = json.load(f)
    
    # Parse pins
    print("Loading pins data...")
    pins_db = parse_pins(pins_path)
    
    # Simulate CTS tree to get timing data
    print("Simulating clock tree timing...")
    try:
        metrics = simulate_cts_tree(cts_tree_path, claimed_buffers_path, show_progress=True)
        sink_paths = metrics['sink_paths']
        print(f"  Skew: {metrics['skew']:.3f} ns")
        print(f"  Max Delay: {metrics['max_delay']:.3f} ns")
        print(f"  Min Delay: {metrics['min_delay']:.3f} ns")
    except Exception as e:
        print(f"ERROR: Failed to simulate CTS tree: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Determine output path
    if output_path is None:
        output_dir = f'build/{design_name}'
        os.makedirs(output_dir, exist_ok=True)
        tree_name = 'htree' if tree_type.lower() == 'h' else 'xtree'
        output_path = f'{output_dir}/{design_name}_clock_skew_{tree_name}.png'
    
    # Create visualization
    print("Generating clock skew visualization...")
    frame_img = create_clock_skew_frame(
        cts_tree, sink_paths, pins_db, "Clock Skew Analysis",
        tree_type, claimed_buffers_path,
        show_tree, show_dffs, show_buffers, highlight_critical, skew_threshold
    )
    
    # Save as PNG
    try:
        import imageio.v2 as imageio_v2
        imageio_v2.imwrite(output_path, frame_img)
    except ImportError:
        imageio.imwrite(output_path, frame_img)
    
    print(f"✓ Clock skew visualization saved to: {output_path}")
    return True


def animate_clock_skew_building(design_name: str,
                                cts_tree_path: str,
                                pins_path: str = 'fabric/pins.yaml',
                                claimed_buffers_path: Optional[str] = None,
                                output_path: Optional[str] = None,
                                tree_type: str = 'h',
                                fps: int = 5,
                                skew_threshold: Optional[float] = None) -> bool:
    """
    Create an animation of clock tree building with skew visualization.
    
    This shows the clock tree being built level by level, with DFFs
    color-coded as they get connected to the tree.
    """
    if not os.path.exists(cts_tree_path):
        print(f"ERROR: CTS tree file not found: {cts_tree_path}")
        return False
    
    # Load CTS tree
    print(f"Loading CTS tree from: {cts_tree_path}")
    with open(cts_tree_path, 'r') as f:
        cts_tree = json.load(f)
    
    # Parse pins
    print("Loading pins data...")
    pins_db = parse_pins(pins_path)
    
    # Load buffer types
    buffer_types = {}
    if claimed_buffers_path and os.path.exists(claimed_buffers_path):
        with open(claimed_buffers_path, 'r') as f:
            claimed_buffers = json.load(f)
            for buf_name, buf_info in claimed_buffers.items():
                buffer_types[buf_name] = buf_info.get('type', 'BUF')
    
    # Simulate final tree to get all timing data
    print("Simulating final clock tree timing...")
    try:
        metrics = simulate_cts_tree(cts_tree_path, claimed_buffers_path, show_progress=False)
        final_sink_paths = metrics['sink_paths']
        final_delays = {path['dff_name']: path['delay'] for path in final_sink_paths}
        max_delay = metrics['max_delay']
        min_delay = metrics['min_delay']
    except Exception as e:
        print(f"ERROR: Failed to simulate CTS tree: {e}")
        return False
    
    # Build tree incrementally for animation
    root = cts_tree.get('root')
    if not root:
        print("ERROR: No root found in CTS tree")
        return False
    
    # Collect frames by building tree level by level
    recorded_frames = []
    
    def collect_tree_levels(node, level=0, parent_x=None, parent_y=None, connected_dffs=None):
        """Recursively collect tree structure level by level."""
        if connected_dffs is None:
            connected_dffs = set()
        
        if node['type'] == 'buffer':
            x = node['x']
            y = node['y']
            
            # Record frame at this level
            frame_data = {
                'level': level,
                'buffer_x': x,
                'buffer_y': y,
                'parent_x': parent_x,
                'parent_y': parent_y,
                'connected_dffs': connected_dffs.copy(),
                'node': node
            }
            recorded_frames.append(frame_data)
            
            # Process children
            for child in node.get('children', []):
                collect_tree_levels(child, level + 1, x, y, connected_dffs)
        
        elif node['type'] == 'sink':
            # Add DFFs from this sink to connected set
            sinks = node.get('sinks', [])
            for sink in sinks:
                dff_name = sink.get('dff_name', '')
                if dff_name:
                    connected_dffs.add(dff_name)
    
    # Collect all levels
    collect_tree_levels(root)
    
    # Add final frame
    recorded_frames.append({
        'level': -1,  # Final frame
        'connected_dffs': set(final_delays.keys()),
        'all_paths': final_sink_paths
    })
    
    print(f"Recorded {len(recorded_frames)} frames for animation")
    
    # Determine output path
    if output_path is None:
        output_dir = f'build/{design_name}'
        os.makedirs(output_dir, exist_ok=True)
        tree_name = 'htree' if tree_type.lower() == 'h' else 'xtree'
        output_path = f'{output_dir}/{design_name}_clock_skew_{tree_name}_animation.mp4'
    
    # Ensure output path has .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    
    # Check for imageio
    if imageio is None:
        print("ERROR: imageio required for video export.")
        print("Install with: pip install imageio imageio-ffmpeg")
        return False
    
    # Generate frames
    print(f"\nGenerating {len(recorded_frames)} frames...")
    frames_images = []
    
    for i, frame_data in enumerate(recorded_frames):
        # Create partial sink paths for this frame
        if frame_data['level'] == -1:
            # Final frame - use all paths
            sink_paths = frame_data.get('all_paths', final_sink_paths)
            phase = "Final Clock Tree"
        else:
            # Partial frame - only show connected DFFs
            connected_dffs = frame_data.get('connected_dffs', set())
            sink_paths = [p for p in final_sink_paths if p['dff_name'] in connected_dffs]
            phase = f"Building Level {frame_data['level']}"
        
        frame_img = create_clock_skew_frame(
            cts_tree, sink_paths, pins_db, phase,
            tree_type, claimed_buffers_path,
            show_tree=True, show_dffs=True, show_buffers=True,
            highlight_critical=True, skew_threshold=skew_threshold
        )
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
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create Clock Skew visualization animation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create static visualization from CTS tree:
  python animate_clock_skew.py --design 6502 --cts-tree build/6502/cts_tree_htree.json
  
  # Create animation of clock tree building:
  python animate_clock_skew.py --design 6502 --cts-tree build/6502/cts_tree_htree.json --animate
  
  # Check skew against threshold:
  python animate_clock_skew.py --design 6502 --cts-tree build/6502/cts_tree_htree.json --skew-threshold 0.5
        """
    )
    parser.add_argument('--design', type=str, required=True,
                       help='Design name (e.g., 6502)')
    parser.add_argument('--cts-tree', type=str, default=None,
                       help='Path to CTS tree JSON file (auto-detects if not provided)')
    parser.add_argument('--claimed-buffers', type=str, default=None,
                       help='Path to claimed_buffers.json (optional)')
    parser.add_argument('--pins', type=str, default='fabric/pins.yaml',
                       help='Path to pins.yaml')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for image/video')
    parser.add_argument('--tree-type', type=str, default='h', choices=['h', 'x'],
                       help='Tree type: h for H-Tree, x for X-Tree (default: h)')
    parser.add_argument('--animate', action='store_true',
                       help='Create animation of clock tree building')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for video (default: 5)')
    parser.add_argument('--skew-threshold', type=float, default=None,
                       help='Skew threshold in ns to highlight violations')
    parser.add_argument('--no-tree', action='store_true',
                       help='Hide clock tree structure')
    parser.add_argument('--no-dffs', action='store_true',
                       help='Hide DFFs')
    parser.add_argument('--no-buffers', action='store_true',
                       help='Hide buffers')
    parser.add_argument('--no-critical', action='store_true',
                       help='Don\'t highlight critical paths')
    
    args = parser.parse_args()
    
    show_tree = not args.no_tree
    show_dffs = not args.no_dffs
    show_buffers = not args.no_buffers
    highlight_critical = not args.no_critical
    
    # Auto-detect CTS tree path if not provided
    if args.cts_tree is None:
        # Try common locations for best placement CTS trees
        possible_paths = [
            f'build/{args.design}/cts_with_manager/best_{args.tree_type}tree/{args.design}_cts_tree.json',
            f'build/{args.design}/cts_with_manager/greedy_{args.tree_type}tree/{args.design}_cts_tree.json',
            f'build/{args.design}/cts/best/{args.design}_cts_tree.json',
            f'build/{args.design}/cts/greedy/{args.design}_cts_tree.json',
            f'build/{args.design}/{args.design}_cts_tree.json',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.cts_tree = path
                print(f"Auto-detected CTS tree: {path}")
                break
        
        if args.cts_tree is None:
            print(f"ERROR: No CTS tree file found. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease specify --cts-tree explicitly or generate CTS tree first.")
            sys.exit(1)
    
    # Auto-detect claimed buffers path if not provided
    if args.claimed_buffers is None:
        # Try common locations (relative to CTS tree path)
        cts_dir = os.path.dirname(args.cts_tree)
        possible_paths = [
            os.path.join(cts_dir, 'claimed_buffers.json'),
            args.cts_tree.replace('_tree.json', '_buffers.json').replace('tree.json', 'buffers.json'),
            args.cts_tree.replace('cts_tree', 'claimed_buffers'),
            f'build/{args.design}/claimed_buffers.json'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.claimed_buffers = path
                print(f"Auto-detected claimed buffers: {path}")
                break
    
    if args.animate:
        success = animate_clock_skew_building(
            args.design,
            args.cts_tree,
            args.pins,
            args.claimed_buffers,
            args.output,
            args.tree_type,
            args.fps,
            args.skew_threshold
        )
    else:
        success = animate_clock_skew_static(
            args.design,
            args.cts_tree,
            args.pins,
            args.claimed_buffers,
            args.output,
            args.tree_type,
            show_tree,
            show_dffs,
            show_buffers,
            highlight_critical,
            args.skew_threshold
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
