#!/usr/bin/env python3
"""
Greedy Placement Animator: Creates an animation of the greedy placement process
by recording frames during placement and saving to MP4 video.
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
from validator import validate_design
from parse_fabric import parse_fabric_cells, parse_pins
from parse_design import parse_design
from placer import (
    extract_nets, calculate_hpwl, get_port_to_net_mapping,
    build_net_index, precompute_cell_nets, build_slot_spatial_index,
    find_nearest_slot_linear, find_placed_neighbors_fast, add_io_pin_neighbors,
    calculate_barycenter, get_fallback_position
)

# Color mapping for different cell types
CELL_COLORS = {
    'sky130_fd_sc_hd__nand2_2': '#FFD700',      # Gold/Yellow
    'sky130_fd_sc_hd__or2_2': '#FFA500',        # Orange
    'sky130_fd_sc_hd__clkinv_2': '#87CEEB',     # Sky Blue
    'sky130_fd_sc_hd__clkbuf_4': '#4169E1',     # Royal Blue
    'sky130_fd_sc_hd__dfbbp_1': '#FF69B4',      # Hot Pink
    'sky130_fd_sc_hd__conb_1': '#32CD32',       # Lime Green
    'sky130_fd_sc_hd__tapvpwrvgnd_1': '#808080', # Gray
    'sky130_fd_sc_hd__decap_3': '#DDA0DD',      # Plum
    'sky130_fd_sc_hd__decap_4': '#DDA0DD',      # Plum
    'unknown': '#000000'                        # Black
}

SITE_WIDTH = 0.46  # um
SITE_HEIGHT = 2.72  # um

CELL_WIDTH_SITES = {
    'sky130_fd_sc_hd__nand2_2': 5,
    'sky130_fd_sc_hd__or2_2': 5,
    'sky130_fd_sc_hd__clkinv_2': 4,
    'sky130_fd_sc_hd__clkbuf_4': 6,
    'sky130_fd_sc_hd__dfbbp_1': 26,
    'sky130_fd_sc_hd__conb_1': 3,
    'sky130_fd_sc_hd__tapvpwrvgnd_1': 1,
    'sky130_fd_sc_hd__decap_3': 3,
    'sky130_fd_sc_hd__decap_4': 4,
}

def get_cell_dimensions(cell_type: str) -> Tuple[float, float]:
    """Get cell width and height in microns."""
    width_sites = CELL_WIDTH_SITES.get(cell_type, 5)
    width_um = width_sites * SITE_WIDTH
    height_um = SITE_HEIGHT
    return width_um, height_um


def create_placement_frame(placement: Dict[str, Dict[str, Any]],
                           fabric_db: Dict[str, List[Dict[str, Any]]],
                           pins_db: Dict[str, Any],
                           netlist_graph: Dict[str, Dict[str, Any]],
                           phase: str,
                           cells_placed: int,
                           total_cells: int) -> np.ndarray:
    """
    Create a single frame showing current placement state.
    Returns image as numpy array.
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
    
    # Skip drawing all fabric slots - too slow (10,000+ slots)
    # Just show the core area as a light background
    ax.add_patch(patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=0, facecolor='lightgray', alpha=0.1
    ))
    
    # Draw placed cells
    for cell_name, cell_data in placement.items():
        if cell_name not in netlist_graph:
            continue
        cell_type = netlist_graph[cell_name]['type']
        color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
        width_um, height_um = get_cell_dimensions(cell_type)
        
        x = cell_data['x']
        y = cell_data['y']
        
        rect = patches.Rectangle(
            (x - width_um/2, y - height_um/2),
            width_um, height_um,
            linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)
    
    # Draw pins
    for pin in pins_db.get('pins', []):
        if pin.get('status') == 'FIXED':
            ax.plot(pin['x_um'], pin['y_um'], 'ro', markersize=4, alpha=0.6)
    
    # Set axis properties
    ax.set_xlim(0, die_width)
    ax.set_ylim(0, die_height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    
    # Add title with phase and progress
    progress_pct = (cells_placed / total_cells * 100) if total_cells > 0 else 0
    title = f'Greedy Placement Animation - {phase}\n'
    title += f'{cells_placed}/{total_cells} cells placed ({progress_pct:.1f}%)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Convert to numpy array
    fig.canvas.draw()
    # Use a more reliable method to get image data
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    if imageio is None:
        raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")
    # Use imageio.v2 to avoid deprecation warning
    try:
        import imageio.v2 as imageio_v2
        img = imageio_v2.imread(buf)
    except ImportError:
        img = imageio.imread(buf)
    buf.close()
    plt.close(fig)
    return img


def place_io_connected_cells_with_recording(fabric_db, pins_db, netlist_graph, port_to_nets,
                                           cell_to_nets, placement, used_slots, 
                                           recorded_frames, total_cells, max_cells=None):
    """Place I/O-connected cells (seed phase) with frame recording."""
    # Build net-to-pin mapping
    net_to_pin_locations = defaultdict(list)
    for pin in pins_db.get('pins', []):
        if pin.get('status') == 'FIXED':
            pin_name = pin['name']
            if pin_name in port_to_nets:
                for net_id in port_to_nets[pin_name]:
                    net_to_pin_locations[net_id].append((pin['x_um'], pin['y_um']))
    
    # Find all I/O-connected cells
    io_connected_cells = []
    for cell_name in netlist_graph.keys():
        if cell_name in placement:
            continue
        cell_nets = cell_to_nets.get(cell_name, set())
        for net_id in cell_nets:
            if net_id in net_to_pin_locations:
                io_connected_cells.append(cell_name)
                break
    
    phase = "Seed Phase: I/O-connected cells"
    cells_placed = 0
    last_printed_pct = -1
    
    for cell_name in io_connected_cells:
        if cell_name in placement:
            continue
        
        # Check if we've reached max_cells limit
        if max_cells is not None and len(placement) >= max_cells:
            break
        
        cell_nets = cell_to_nets.get(cell_name, set())
        cell_type = netlist_graph[cell_name]['type']
        
        # Collect pin locations
        pin_locations = []
        for net_id in cell_nets:
            if net_id in net_to_pin_locations:
                pin_locations.extend(net_to_pin_locations[net_id])
        
        if not pin_locations:
            continue
        
        # Calculate barycenter
        target_x = sum(x for x, y in pin_locations) / len(pin_locations)
        target_y = sum(y for x, y in pin_locations) / len(pin_locations)
        
        # Find nearest slot
        available_slots = [slot for slot in fabric_db.get(cell_type, [])
                         if slot['name'] not in used_slots]
        
        if not available_slots:
            continue
            
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
            
            # Print progress
            progress_pct = int((cells_placed / total_cells * 100)) if total_cells > 0 else 0
            if progress_pct != last_printed_pct:
                print(f"Progress: {progress_pct}% ({cells_placed}/{total_cells} cells)", end='\r')
                sys.stdout.flush()
                last_printed_pct = progress_pct
            
            # Record frame every 5 cells or at milestones
            if cells_placed % 5 == 0 or progress_pct in [10, 25, 50, 75, 90]:
                recorded_frames.append((
                    placement.copy(),
                    phase,
                    cells_placed
                ))
    
    if total_cells > 0:
        print()  # New line after progress
    return cells_placed


def place_greedy_barycenter_with_recording(fabric_db, netlist_graph, placement, used_slots,
                                          net_index, cell_to_nets, pins_db, port_to_nets,
                                          recorded_frames, total_cells, seed_cells, max_cells=None):
    """Place remaining cells using greedy barycenter with frame recording."""
    unplaced_cells = {cell for cell in netlist_graph.keys() if cell not in placement}
    
    if not unplaced_cells:
        return 0
    
    phase = "Grow Phase: Greedy placement"
    fallback_pos = get_fallback_position(pins_db, fabric_db)
    
    # Initialize scores
    cell_scores = {}
    for cell in unplaced_cells:
        neighbors = find_placed_neighbors_fast(
            cell, net_index, cell_to_nets[cell], placement
        )
        if pins_db and port_to_nets:
            add_io_pin_neighbors(cell_to_nets[cell], pins_db, port_to_nets, neighbors)
        cell_scores[cell] = (len(neighbors), len(cell_to_nets[cell]))
    
    cells_placed = 0
    last_printed_pct = -1
    
    while unplaced_cells:
        # Check if we've reached max_cells limit
        if max_cells is not None and len(placement) >= max_cells:
            break
        
        # Find best cell
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
        
        # Find nearest slot
        available_slots = [slot for slot in fabric_db.get(cell_type, [])
                         if slot['name'] not in used_slots]
        
        if not available_slots:
            unplaced_cells.remove(best_cell)
            del cell_scores[best_cell]
            continue
        
        best_slot = find_nearest_slot_linear(target_x, target_y, available_slots)
        
        if not best_slot:
            unplaced_cells.remove(best_cell)
            del cell_scores[best_cell]
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
        
        # Print progress
        total_placed = seed_cells + cells_placed
        progress_pct = int((total_placed / total_cells * 100)) if total_cells > 0 else 0
        if progress_pct != last_printed_pct:
            print(f"Progress: {progress_pct}% ({total_placed}/{total_cells} cells)", end='\r')
            sys.stdout.flush()
            last_printed_pct = progress_pct
        
        # Record frame every 5 cells or at milestones
        if total_placed % 5 == 0 or progress_pct in [10, 25, 50, 75, 90]:
            recorded_frames.append((
                placement.copy(),
                phase,
                total_placed
            ))
        
        # Update scores for affected cells
        placed_nets = cell_to_nets[best_cell]
        affected_cells = set()
        for net_id in placed_nets:
            affected_cells.update(net_index.get(net_id, set()))
        
        for affected_cell in affected_cells:
            if affected_cell in unplaced_cells:
                neighbors = find_placed_neighbors_fast(
                    affected_cell, net_index, cell_to_nets[affected_cell], placement
                )
                if pins_db and port_to_nets:
                    add_io_pin_neighbors(cell_to_nets[affected_cell], pins_db, 
                                       port_to_nets, neighbors)
                cell_scores[affected_cell] = (len(neighbors), len(cell_to_nets[affected_cell]))
    
    if total_cells > 0:
        print()  # New line after progress
    
    return cells_placed


def animate_greedy_placement(design_name: str,
                            fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                            pins_path: str = 'fabric/pins.yaml',
                            max_cells: Optional[int] = None,
                            output_path: Optional[str] = None,
                            fps: int = 10):
    """
    Create an animation of the greedy placement process.
    
    Args:
        design_name: Name of the design (e.g., '6502')
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        max_cells: Limit number of cells to place (for testing)
        output_path: Path to save MP4 video (default: build/{design}/placement_animation.mp4)
        fps: Frames per second for video (default: 10)
    """
    design_path = f'designs/{design_name}_mapped.json'
    
    if not os.path.exists(design_path):
        print(f"ERROR: Design file not found: {design_path}")
        sys.exit(1)
    
    # Validate design
    print("Validating design...")
    if not validate_design(fabric_cells_path, design_path):
        print("ERROR: Design validation failed.")
        sys.exit(1)
    
    # Parse data
    print("Loading fabric and design data...")
    fabric_db = parse_fabric_cells(fabric_cells_path)
    logical_db, netlist_graph = parse_design(design_path)
    pins_db = parse_pins(pins_path)
    port_to_nets = get_port_to_net_mapping(design_path)
    
    # Build indices
    print("Building optimization indices...")
    net_index = build_net_index(netlist_graph)
    cell_to_nets = precompute_cell_nets(netlist_graph)
    slot_lists = build_slot_spatial_index(fabric_db)
    
    actual_total_cells = len(netlist_graph)
    total_cells = min(max_cells, actual_total_cells) if max_cells is not None else actual_total_cells
    
    if max_cells is not None:
        print(f"\nTEST MODE: Limiting to {max_cells} cells (out of {actual_total_cells} total)")
    
    # Initialize placement
    placement = {}
    used_slots = set()
    recorded_frames = []
    
    # Record initial frame
    recorded_frames.append((
        placement.copy(),
        "Initializing",
        0
    ))
    
    # Determine output path
    if output_path is None:
        output_dir = f'build/{design_name}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/placement_animation.mp4'
    
    print(f"\nStarting placement...")
    print(f"Video will be saved to: {output_path}\n")
    
    # Place I/O-connected cells
    print("Seed Phase: Placing I/O-connected cells...")
    io_cells = place_io_connected_cells_with_recording(
        fabric_db, pins_db, netlist_graph, port_to_nets,
        cell_to_nets, placement, used_slots, recorded_frames, total_cells, max_cells
    )
    print(f"Placed {io_cells} I/O-connected cells.\n")
    
    # Place remaining cells
    print("Grow Phase: Placing remaining cells...")
    remaining_cells = place_greedy_barycenter_with_recording(
        fabric_db, netlist_graph, placement, used_slots,
        net_index, cell_to_nets, pins_db, port_to_nets,
        recorded_frames, total_cells, io_cells, max_cells
    )
    print(f"Placed {remaining_cells} remaining cells.\n")
    
    # Record final frame
    total_placed = io_cells + remaining_cells
    recorded_frames.append((
        placement.copy(),
        "Complete",
        total_placed
    ))
    
    print(f"Placement complete! {total_placed}/{total_cells} cells placed.")
    print(f"\nGenerating video from {len(recorded_frames)} frames...")
    
    # Check for imageio
    if imageio is None:
        print("ERROR: imageio required for video export.")
        print("Install with: pip install imageio imageio-ffmpeg")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate frames
    print("  Generating frames...")
    frames_images = []
    for i, (placement_state, phase, cells_placed) in enumerate(recorded_frames):
        frame_img = create_placement_frame(
            placement_state, fabric_db, pins_db, netlist_graph,
            phase, cells_placed, total_cells
        )
        frames_images.append(frame_img)
        
        # Show progress every frame or every 5 frames for large sets
        if len(recorded_frames) <= 50 or (i + 1) % 5 == 0 or i == len(recorded_frames) - 1:
            pct = int((i + 1) / len(recorded_frames) * 100)
            print(f"  Generated {i + 1}/{len(recorded_frames)} frames ({pct}%)", end='\r')
    
    print()  # New line
    
    # Save video
    print("Writing video file...")
    try:
        imageio.mimsave(output_path, frames_images, fps=fps, codec='libx264', quality=8)
        print(f"\nVideo saved successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"\nError saving video: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create animation of greedy placement process'
    )
    parser.add_argument('--design', type=str, required=True,
                       help='Design name (e.g., 6502)')
    parser.add_argument('--fabric-cells', type=str, default='fabric/fabric_cells.yaml',
                       help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', type=str, default='fabric/pins.yaml',
                       help='Path to pins.yaml')
    parser.add_argument('--max-cells', type=int, default=None,
                       help='Limit number of cells to place (for testing)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for MP4 video (default: build/{design}/placement_animation.mp4)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for video (default: 10)')
    
    args = parser.parse_args()
    
    animate_greedy_placement(
        args.design,
        args.fabric_cells,
        args.pins,
        args.max_cells,
        args.output,
        args.fps
    )


if __name__ == '__main__':
    main()
