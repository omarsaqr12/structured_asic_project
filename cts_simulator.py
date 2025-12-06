#!/usr/bin/env python3
"""
CTS Simulation Engine

Simulates clock tree timing by calculating buffer delays and wire delays.
Compares H-Tree vs X-Tree for skew, wirelength, and buffer usage metrics.
"""

import json
import math
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


# Default buffer delays (in nanoseconds) for sky130 technology
# These are typical values - in real flow, would parse from liberty file
DEFAULT_BUFFER_DELAYS = {
    'sky130_fd_sc_hd__clkbuf_4': 0.15,  # Buffer delay: ~150ps
    'sky130_fd_sc_hd__clkinv_2': 0.12,  # Inverter delay: ~120ps
}

# Wire delay model: delay per micron (simplified RC model)
# Typical value: ~0.001 ns/um for metal layers
WIRE_DELAY_PER_UM = 0.001  # nanoseconds per micron


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points in microns."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_buffer_delay(buffer_type: str) -> float:
    """
    Get buffer delay from timing library or use default.
    
    Args:
        buffer_type: Buffer cell type (BUF or INV)
    
    Returns:
        Delay in nanoseconds
    """
    if buffer_type == 'BUF':
        return DEFAULT_BUFFER_DELAYS['sky130_fd_sc_hd__clkbuf_4']
    elif buffer_type == 'INV':
        return DEFAULT_BUFFER_DELAYS['sky130_fd_sc_hd__clkinv_2']
    else:
        # Default to buffer delay if unknown
        return DEFAULT_BUFFER_DELAYS['sky130_fd_sc_hd__clkbuf_4']


def calculate_wire_delay(distance_um: float) -> float:
    """
    Calculate wire delay based on distance.
    
    Uses simplified RC model: delay = distance * delay_per_um
    
    Args:
        distance_um: Wire length in microns
    
    Returns:
        Wire delay in nanoseconds
    """
    return distance_um * WIRE_DELAY_PER_UM


def trace_path_to_sinks(node: Dict[str, Any],
                        path_delay: float,
                        path_wirelength: float,
                        buffer_count: int,
                        buffer_types: Dict[str, str],
                        parent_x: Optional[float] = None,
                        parent_y: Optional[float] = None,
                        progress_callback=None) -> List[Dict[str, Any]]:
    """
    Recursively trace paths from root to all DFF sinks.
    Optimized version with progress tracking.
    
    Args:
        node: Current tree node (buffer or sink)
        path_delay: Accumulated delay so far (ns)
        path_wirelength: Accumulated wirelength so far (um)
        buffer_count: Number of buffers in path so far
        buffer_types: Dict mapping buffer_name -> type (BUF/INV)
        parent_x: Parent buffer X coordinate
        parent_y: Parent buffer Y coordinate
        progress_callback: Optional function(count) called when DFF is processed
    
    Returns:
        List of path information for each sink: {delay, wirelength, buffer_count, dff_name, dff_x, dff_y}
    """
    paths = []
    
    if node['type'] == 'buffer':
        x = node['x']
        y = node['y']
        buffer_name = node.get('buffer_name', '')
        
        # Get buffer type from buffer_types dict or default to BUF
        buffer_type_str = buffer_types.get(buffer_name, 'BUF')
        
        # Calculate wire delay from parent to this buffer (optimize: only if parent exists)
        if parent_x is not None and parent_y is not None:
            wire_length = calculate_distance(parent_x, parent_y, x, y)
            wire_delay = calculate_wire_delay(wire_length)
        else:
            wire_length = 0.0
            wire_delay = 0.0
        
        # Add buffer delay (cache this value)
        buffer_delay = get_buffer_delay(buffer_type_str)
        
        # Update accumulated values
        new_path_delay = path_delay + wire_delay + buffer_delay
        new_path_wirelength = path_wirelength + wire_length
        new_buffer_count = buffer_count + 1
        
        # Process children (batch process)
        for child in node.get('children', []):
            child_paths = trace_path_to_sinks(
                child, new_path_delay, new_path_wirelength, new_buffer_count, 
                buffer_types, x, y, progress_callback
            )
            paths.extend(child_paths)
    
    elif node['type'] == 'sink':
        # Calculate wire delay from parent buffer to sink center
        sink_x = node['x']
        sink_y = node['y']
        
        if parent_x is not None and parent_y is not None:
            wire_length = calculate_distance(parent_x, parent_y, sink_x, sink_y)
            wire_delay = calculate_wire_delay(wire_length)
        else:
            wire_length = 0.0
            wire_delay = 0.0
        
        final_delay = path_delay + wire_delay
        final_wirelength = path_wirelength + wire_length
        
        # For each DFF in this sink group (batch process all DFFs)
        sinks = node.get('sinks', [])
        for i, sink in enumerate(sinks):
            dff_x = sink['x']
            dff_y = sink['y']
            dff_name = sink.get('dff_name', 'unknown')
            
            # Calculate final wire from sink center to DFF
            final_wire_length = calculate_distance(sink_x, sink_y, dff_x, dff_y)
            final_wire_delay = calculate_wire_delay(final_wire_length)
            
            paths.append({
                'dff_name': dff_name,
                'dff_x': dff_x,
                'dff_y': dff_y,
                'delay': final_delay + final_wire_delay,
                'wirelength': final_wirelength + final_wire_length,
                'buffer_count': buffer_count,
                'path_length': buffer_count
            })
            
            # Progress callback
            if progress_callback:
                progress_callback(len(paths))
    
    return paths


def simulate_cts_tree(cts_tree_path: str,
                      claimed_buffers_path: Optional[str] = None,
                      show_progress: bool = True) -> Dict[str, Any]:
    """
    Simulate CTS tree and calculate timing metrics.
    
    Args:
        cts_tree_path: Path to CTS tree JSON file
        claimed_buffers_path: Optional path to claimed_buffers.json for buffer types
        show_progress: Whether to show progress updates
    
    Returns:
        Dictionary with metrics: {skew, max_delay, min_delay, total_wirelength, 
                                 buffer_count, avg_path_length, sink_paths}
    """
    if show_progress:
        print(f"  Loading CTS tree from: {cts_tree_path}")
    
    # Load CTS tree
    with open(cts_tree_path, 'r') as f:
        cts_tree = json.load(f)
    
    if show_progress:
        print("  Loading buffer types...", end='', flush=True)
    
    # Load claimed buffers to get buffer types
    buffer_types = {}  # Map buffer_name -> type (BUF/INV)
    if claimed_buffers_path and Path(claimed_buffers_path).exists():
        with open(claimed_buffers_path, 'r') as f:
            claimed_buffers = json.load(f)
            for buf_name, buf_info in claimed_buffers.items():
                buffer_types[buf_name] = buf_info.get('type', 'BUF')
    
    if show_progress:
        print(f" Done. Found {len(buffer_types)} buffer types")
    
    # Trace all paths from root to sinks
    root = cts_tree.get('root')
    if not root:
        raise ValueError("No 'root' key found in CTS tree")
    
    # Count total sinks for progress
    if show_progress:
        print("  Counting total sinks...", end='', flush=True)
    
    def count_sinks(node):
        count = 0
        if node['type'] == 'sink':
            count = len(node.get('sinks', []))
        elif node['type'] == 'buffer':
            for child in node.get('children', []):
                count += count_sinks(child)
        return count
    
    total_sinks = count_sinks(root)
    processed = [0]
    last_update = [0]
    
    # Progress callback with more frequent updates
    def progress(count):
        processed[0] = count
        if show_progress and total_sinks > 0:
            progress_pct = int((count / total_sinks) * 100)
            # Update every 1% or every 5 sinks, whichever is more frequent
            update_interval = max(1, min(total_sinks // 100, 5))
            if count - last_update[0] >= update_interval or count == total_sinks:
                print(f"\r  Processing paths: {progress_pct}% ({count}/{total_sinks} DFFs)", end='', flush=True)
                last_update[0] = count
    
    if show_progress:
        print(f"\r  Counting complete. Tracing {total_sinks} sink paths...", end='', flush=True)
    
    # Start tracing from root (no parent, no delay yet)
    sink_paths = trace_path_to_sinks(root, 0.0, 0.0, 0, buffer_types, None, None, progress)
    
    if show_progress:
        print(f"\r  Processing paths: 100% ({total_sinks}/{total_sinks} DFFs) - Complete!")
    
    if not sink_paths:
        raise ValueError("No sink paths found in CTS tree")
    
    if show_progress:
        print("  Calculating metrics...", end='', flush=True)
    
    # Calculate metrics
    delays = [path['delay'] for path in sink_paths]
    wirelengths = [path['wirelength'] for path in sink_paths]
    buffer_counts = [path['buffer_count'] for path in sink_paths]
    
    max_delay = max(delays)
    min_delay = min(delays)
    skew = max_delay - min_delay
    total_wirelength = sum(wirelengths)
    avg_wirelength = total_wirelength / len(wirelengths) if wirelengths else 0.0
    avg_path_length = sum(buffer_counts) / len(buffer_counts) if buffer_counts else 0.0
    
    # Count unique buffers (from tree structure)
    def count_buffers(node):
        count = 0
        if node['type'] == 'buffer':
            count = 1
            for child in node.get('children', []):
                count += count_buffers(child)
        return count
    
    buffer_count = count_buffers(root)
    
    if show_progress:
        print(" Done.")
    
    return {
        'skew': skew,
        'max_delay': max_delay,
        'min_delay': min_delay,
        'total_wirelength': total_wirelength,
        'avg_wirelength': avg_wirelength,
        'buffer_count': buffer_count,
        'avg_path_length': avg_path_length,
        'num_sinks': len(sink_paths),
        'sink_paths': sink_paths
    }


def compare_trees(htree_path: str,
                  xtree_path: str,
                  htree_buffers_path: Optional[str] = None,
                  xtree_buffers_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare H-Tree and X-Tree metrics.
    
    Args:
        htree_path: Path to H-Tree JSON file
        xtree_path: Path to X-Tree JSON file
        htree_buffers_path: Optional path to H-Tree claimed_buffers.json
        xtree_buffers_path: Optional path to X-Tree claimed_buffers.json
    
    Returns:
        Dictionary with comparison metrics
    """
    print("Simulating H-Tree...")
    try:
        htree_metrics = simulate_cts_tree(htree_path, htree_buffers_path, show_progress=True)
        print("✓ H-Tree simulation complete")
    except Exception as e:
        print(f"\n✗ Error simulating H-Tree: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\nSimulating X-Tree...")
    try:
        xtree_metrics = simulate_cts_tree(xtree_path, xtree_buffers_path, show_progress=True)
        print("✓ X-Tree simulation complete")
    except Exception as e:
        print(f"\n✗ Error simulating X-Tree: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return {
        'htree': htree_metrics,
        'xtree': xtree_metrics,
        'comparison': {
            'skew_diff': htree_metrics['skew'] - xtree_metrics['skew'],
            'wirelength_diff': htree_metrics['total_wirelength'] - xtree_metrics['total_wirelength'],
            'buffer_diff': htree_metrics['buffer_count'] - xtree_metrics['buffer_count'],
            'skew_winner': 'H-Tree' if htree_metrics['skew'] < xtree_metrics['skew'] else 'X-Tree',
            'wirelength_winner': 'H-Tree' if htree_metrics['total_wirelength'] < xtree_metrics['total_wirelength'] else 'X-Tree',
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='CTS Simulation Engine - Calculate clock skew and wirelength metrics'
    )
    parser.add_argument('--htree', type=str, required=True,
                        help='Path to H-Tree CTS JSON file')
    parser.add_argument('--xtree', type=str, required=True,
                        help='Path to X-Tree CTS JSON file')
    parser.add_argument('--htree-buffers', type=str, default=None,
                        help='Path to H-Tree claimed_buffers.json (optional)')
    parser.add_argument('--xtree-buffers', type=str, default=None,
                        help='Path to X-Tree claimed_buffers.json (optional)')
    parser.add_argument('--output', type=str, default='cts_comparison.json',
                        help='Output JSON file for comparison results (default: cts_comparison.json)')
    parser.add_argument('--design', type=str, default=None,
                        help='Design name (e.g., 6502) - used to construct default paths')
    
    args = parser.parse_args()
    
    # Handle default paths based on design name
    if args.design:
        if not args.htree_buffers:
            possible_htree = f'build/{args.design}/cts_with_manager/best_htree/claimed_buffers.json'
            if Path(possible_htree).exists():
                args.htree_buffers = possible_htree
        if not args.xtree_buffers:
            possible_xtree = f'build/{args.design}/cts_with_manager/best_xtree/claimed_buffers.json'
            if Path(possible_xtree).exists():
                args.xtree_buffers = possible_xtree
        if args.output == 'cts_comparison.json':
            args.output = f'build/{args.design}/cts_comparison.json'
    
    # Compare trees
    comparison = compare_trees(
        args.htree,
        args.xtree,
        args.htree_buffers,
        args.xtree_buffers
    )
    
    # Save results
    print("\nSaving comparison results...", end='', flush=True)
    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f" Done.")
    
    print(f"\nComparison results saved to: {args.output}")
    print("\n=== Comparison Summary ===")
    print(f"H-Tree: skew={comparison['htree']['skew']:.3f}ns, "
          f"wirelength={comparison['htree']['total_wirelength']:.1f}um, "
          f"buffers={comparison['htree']['buffer_count']}")
    print(f"X-Tree: skew={comparison['xtree']['skew']:.3f}ns, "
          f"wirelength={comparison['xtree']['total_wirelength']:.1f}um, "
          f"buffers={comparison['xtree']['buffer_count']}")
    print(f"\nWinner - Skew: {comparison['comparison']['skew_winner']}")
    print(f"Winner - Wirelength: {comparison['comparison']['wirelength_winner']}")


if __name__ == '__main__':
    main()

