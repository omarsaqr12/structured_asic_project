#!/usr/bin/env python3
"""
Simulated Annealing Placement Animator: Creates an animation of the SA optimization process
by recording frames during optimization and saving to MP4 video.
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
    calculate_total_hpwl, generate_move, should_accept_move,
    cool_temperature, update_window_size, calculate_initial_temperature,
    get_affected_nets, HPWLCache, place_design
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
                           temperature: float,
                           current_hpwl: float,
                           best_hpwl: float,
                           acceptance_rate: float,
                           window_size: float,
                           iteration: int) -> np.ndarray:
    """
    Create a single frame showing current placement state with SA metrics.
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
    
    # Add title with SA metrics
    improvement = current_hpwl - best_hpwl if current_hpwl > best_hpwl else 0
    title = f'Simulated Annealing Optimization - {phase}\n'
    title += f'Temperature: {temperature:.2f} | Window: {window_size:.1f} um | Iteration: {iteration}\n'
    title += f'Current HPWL: {current_hpwl:.2f} um | Best HPWL: {best_hpwl:.2f} um'
    if improvement > 0:
        title += f' | Improvement: {improvement:.2f} um'
    title += f'\nAcceptance Rate: {acceptance_rate:.1%}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
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


def simulated_annealing_with_recording(initial_placement: Dict[str, Dict[str, Any]],
                                       fabric_db: Dict[str, List[Dict[str, Any]]],
                                       netlist_graph: Dict[str, Dict[str, Any]],
                                       nets_dict: Dict[int, List[str]],
                                       pins_db: Optional[Dict[str, Any]] = None,
                                       port_to_nets: Optional[Dict[str, List[int]]] = None,
                                       T_initial: Optional[float] = None,
                                       alpha: float = 0.95,
                                       T_final: float = 0.1,
                                       moves_per_temp: int = 100,
                                       generate_move_func=None,
                                       W_initial: Optional[float] = None,
                                       beta: float = 0.98,
                                       P_refine: float = 0.7,
                                       P_explore: float = 0.3,
                                       recorded_frames: Optional[List] = None,
                                       frame_interval: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Simulated Annealing optimization algorithm with frame recording for animation.
    
    This is a modified version of simulated_annealing that records frames during optimization.
    
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
        generate_move_func: Optional function to generate moves (uses default if None)
        W_initial: Initial window size for explore moves (auto-calculated if None, 50% of die width)
        beta: Window cooling rate (default: 0.98)
        P_refine: Probability of refine move (default: 0.7)
        P_explore: Probability of explore move (default: 0.3)
        recorded_frames: List to append frame data to
        frame_interval: Record frame every N iterations (default: 10)
    
    Returns:
        Best placement found
    """
    if recorded_frames is None:
        recorded_frames = []
    
    # Initialize state
    current_placement = {k: v.copy() for k, v in initial_placement.items()}
    
    # Pre-compute cell-to-nets mapping for incremental HPWL calculation
    cell_to_nets = precompute_cell_nets(netlist_graph)
    
    # Create HPWL cache for incremental calculation (CRITICAL OPTIMIZATION)
    hpwl_cache = HPWLCache(current_placement, nets_dict, fabric_db, pins_db, port_to_nets)
    current_hpwl = hpwl_cache.get_total_hpwl()
    
    best_placement = {k: v.copy() for k, v in current_placement.items()}
    best_hpwl = current_hpwl
    
    # Calculate initial temperature if not provided
    if T_initial is None:
        T_initial = calculate_initial_temperature(current_hpwl)
    
    # Calculate initial window size if not provided
    if W_initial is None and pins_db:
        die_width = pins_db.get('die', {}).get('width_um', 1000.0)
        W_initial = die_width * 0.5  # 50% of die width
    elif W_initial is None:
        W_initial = 500.0  # Default fallback
    
    # Calculate minimum window size (10% of die width)
    if pins_db:
        die_width = pins_db.get('die', {}).get('width_um', 1000.0)
        min_window = die_width * 0.1
    else:
        min_window = 100.0  # Default fallback
    
    T = T_initial
    W = W_initial
    iteration = 0
    
    # Statistics tracking
    refine_acceptances = 0
    refine_attempts = 0
    explore_acceptances = 0
    explore_attempts = 0
    
    # Record initial frame
    initial_hpwl = calculate_total_hpwl(initial_placement, nets_dict, fabric_db, pins_db, port_to_nets)
    recorded_frames.append((
        initial_placement.copy(),
        "Initial (Greedy Placement)",
        T,
        initial_hpwl,
        initial_hpwl,
        0.0,
        W,
        0
    ))
    
    print(f"\n{'='*60}")
    print("Simulated Annealing Optimization (with Animation)")
    print(f"{'='*60}")
    print(f"Initial HPWL: {current_hpwl:.2f} um")
    print(f"Initial Temperature: {T:.2f}")
    print(f"Cooling Rate (alpha): {alpha}")
    print(f"Moves per Temperature: {moves_per_temp}")
    print(f"Final Temperature: {T_final}")
    print(f"Initial Window Size: {W:.2f} um")
    print(f"Window Cooling Rate (beta): {beta}")
    print(f"Move Probabilities: Refine={P_refine:.1%}, Explore={P_explore:.1%}")
    print(f"Frame Recording: Every {frame_interval} iterations")
    print(f"{'='*60}\n")
    
    # Main SA loop
    while T > T_final:
        acceptance_count = 0
        improvement_count = 0
        
        # Attempt N moves at this temperature
        for move_idx in range(moves_per_temp):
            # Generate move using hybrid move set
            if generate_move_func is None:
                # Use default generate_move function with incremental HPWL
                new_placement, delta_cost, move_type, moved_cells = generate_move(
                    current_placement, T, W, fabric_db, netlist_graph, nets_dict,
                    pins_db, port_to_nets, P_refine, P_explore,
                    hpwl_cache, cell_to_nets
                )
            else:
                # Use provided move generation function
                # Note: Custom functions may not support incremental HPWL
                result = generate_move_func(
                    current_placement, T, W, fabric_db, netlist_graph, nets_dict, 
                    pins_db, port_to_nets, P_refine, P_explore
                )
                if len(result) == 4:
                    new_placement, delta_cost, move_type, moved_cells = result
                else:
                    # Backward compatibility: old functions return 3-tuple
                    new_placement, delta_cost, move_type = result
                    moved_cells = set()
                    # Find moved cells by comparing slot names
                    for cell_name in new_placement:
                        if cell_name in current_placement:
                            old_slot = current_placement[cell_name].get('fabric_slot_name')
                            new_slot = new_placement[cell_name].get('fabric_slot_name')
                            if old_slot != new_slot:
                                moved_cells.add(cell_name)
            
            new_hpwl = current_hpwl + delta_cost
            
            # Track move statistics
            if move_type == 'refine':
                refine_attempts += 1
            elif move_type == 'explore':
                explore_attempts += 1
            
            # Accept or reject move (with early rejection optimization)
            if should_accept_move(delta_cost, T):
                # Update HPWL cache for accepted move
                if moved_cells and hpwl_cache is not None and cell_to_nets is not None:
                    affected_nets = get_affected_nets(moved_cells, cell_to_nets)
                    hpwl_cache.update_nets(new_placement, affected_nets)
                    # Recalculate current_hpwl from cache to ensure consistency
                    current_hpwl = hpwl_cache.get_total_hpwl()
                else:
                    # Fallback: use calculated new_hpwl (for custom move functions)
                    current_hpwl = new_hpwl
                
                current_placement = new_placement
                acceptance_count += 1
                
                # Track move-specific acceptances
                if move_type == 'refine':
                    refine_acceptances += 1
                elif move_type == 'explore':
                    explore_acceptances += 1
                
                if delta_cost < 0:
                    improvement_count += 1
                
                # Update best if needed
                if current_hpwl < best_hpwl:
                    best_placement = {k: v.copy() for k, v in current_placement.items()}
                    best_hpwl = current_hpwl
        
        # Calculate acceptance rate
        acceptance_rate = acceptance_count / moves_per_temp if moves_per_temp > 0 else 0.0
        
        # Record frame at regular intervals
        if iteration % frame_interval == 0:
            recorded_frames.append((
                current_placement.copy(),
                f"Iteration {iteration}",
                T,
                current_hpwl,
                best_hpwl,
                acceptance_rate,
                W,
                iteration
            ))
        
        # Print progress
        if iteration % 10 == 0 or T <= T_final * 2:  # Print more frequently near end
            refine_rate = (refine_acceptances / refine_attempts * 100) if refine_attempts > 0 else 0.0
            explore_rate = (explore_acceptances / explore_attempts * 100) if explore_attempts > 0 else 0.0
            improvement_pct = (improvement_count / moves_per_temp * 100) if moves_per_temp > 0 else 0.0
            
            print(f"T={T:.2f} | W={W:.1f}um | Current HPWL={current_hpwl:.2f} um | "
                  f"Best HPWL={best_hpwl:.2f} um | "
                  f"Accept Rate={acceptance_rate:.1%} | "
                  f"Improvements: {improvement_count}/{moves_per_temp} ({improvement_pct:.1f}%) | "
                  f"Refine: {refine_acceptances}/{refine_attempts} ({refine_rate:.1f}%) | "
                  f"Explore: {explore_acceptances}/{explore_attempts} ({explore_rate:.1f}%)")
        
        # Cool temperature and window
        T = cool_temperature(T, alpha)
        W = update_window_size(W, beta, min_window)
        iteration += 1
    
    # Record final frame
    recorded_frames.append((
        best_placement.copy(),
        "Final (Best Solution)",
        T,
        best_hpwl,
        best_hpwl,
        acceptance_rate,
        W,
        iteration
    ))
    
    # Final statistics
    print(f"\n{'='*60}")
    print("SA Optimization Complete")
    print(f"{'='*60}")
    initial_hpwl = calculate_total_hpwl(initial_placement, nets_dict, fabric_db, pins_db, port_to_nets)
    print(f"Initial HPWL: {initial_hpwl:.2f} um")
    print(f"Final Total HPWL: {best_hpwl:.2f} um")
    improvement = initial_hpwl - best_hpwl
    improvement_pct = (improvement / initial_hpwl) * 100 if initial_hpwl > 0 else 0
    print(f"Improvement: {improvement:.2f} um ({improvement_pct:.2f}%)")
    print(f"Total Iterations: {iteration}")
    print(f"{'='*60}\n")
    
    return best_placement


def animate_sa_placement(design_name: str,
                         fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                         pins_path: str = 'fabric/pins.yaml',
                         initial_placement_path: Optional[str] = None,
                         output_path: Optional[str] = None,
                         fps: int = 10,
                         T_initial: Optional[float] = None,
                         alpha: float = 0.95,
                         T_final: float = 0.1,
                         moves_per_temp: int = 100,
                         W_initial: Optional[float] = None,
                         beta: float = 0.98,
                         P_refine: float = 0.7,
                         P_explore: float = 0.3,
                         frame_interval: int = 10):
    """
    Create an animation of the simulated annealing optimization process.
    
    Args:
        design_name: Name of the design (e.g., '6502')
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        initial_placement_path: Path to initial placement JSON (if None, runs greedy first)
        output_path: Path to save MP4 video (default: build/{design}/sa_animation.mp4)
        fps: Frames per second for video (default: 10)
        T_initial: Initial temperature (auto-calculated if None)
        alpha: Cooling rate (default: 0.95)
        T_final: Final temperature (default: 0.1)
        moves_per_temp: Moves per temperature step (default: 100)
        W_initial: Initial window size (auto-calculated if None)
        beta: Window cooling rate (default: 0.98)
        P_refine: Probability of refine move (default: 0.7)
        P_explore: Probability of explore move (default: 0.3)
        frame_interval: Record frame every N iterations (default: 10)
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
    nets_dict = extract_nets(netlist_graph)
    
    # Get initial placement
    if initial_placement_path and os.path.exists(initial_placement_path):
        print(f"Loading initial placement from: {initial_placement_path}")
        with open(initial_placement_path, 'r') as f:
            initial_placement = json.load(f)
    else:
        print("Running greedy placement to get initial solution...")
        initial_placement = place_design(fabric_cells_path, design_path, pins_path)
        if initial_placement is None:
            print("ERROR: Greedy placement failed.")
            sys.exit(1)
        print("Greedy placement complete.")
    
    # Initialize frame recording
    recorded_frames = []
    
    # Determine output path
    if output_path is None:
        output_dir = f'build/{design_name}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/sa_animation.mp4'
    
    # Ensure output path has .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    
    print(f"\nStarting simulated annealing optimization...")
    print(f"Video will be saved to: {output_path}\n")
    
    # Run SA with recording
    sa_placement = simulated_annealing_with_recording(
        initial_placement,
        fabric_db,
        netlist_graph,
        nets_dict,
        pins_db,
        port_to_nets,
        T_initial=T_initial,
        alpha=alpha,
        T_final=T_final,
        moves_per_temp=moves_per_temp,
        generate_move_func=None,
        W_initial=W_initial,
        beta=beta,
        P_refine=P_refine,
        P_explore=P_explore,
        recorded_frames=recorded_frames,
        frame_interval=frame_interval
    )
    
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
    for i, (placement_state, phase, temp, curr_hpwl, best_hpwl, accept_rate, window, iter_num) in enumerate(recorded_frames):
        frame_img = create_placement_frame(
            placement_state, fabric_db, pins_db, netlist_graph,
            phase, temp, curr_hpwl, best_hpwl, accept_rate, window, iter_num
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
        # Use explicit video writer to ensure MP4 format
        # Check if imageio-ffmpeg is available
        try:
            import imageio.plugins.ffmpeg
        except ImportError:
            print("\nWARNING: imageio-ffmpeg not found. Trying alternative method...")
            print("If this fails, install with: pip install imageio-ffmpeg")
        
        # Force MP4 format by using extension and explicit writer
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        for frame in frames_images:
            writer.append_data(frame)
        writer.close()
        
        print(f"\nVideo saved successfully to: {output_path}")
        
        # Save final optimized placement
        placement_output_path = output_path.replace('.mp4', '_placement.json')
        print(f"Saving final optimized placement to: {placement_output_path}")
        with open(placement_output_path, 'w') as f:
            json.dump(sa_placement, f, indent=2)
        print(f"Final placement saved successfully!")
        
        return True
    except Exception as e:
        print(f"\nError saving video: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure imageio-ffmpeg is installed: pip install imageio-ffmpeg")
        print("2. Ensure ffmpeg is installed on your system")
        print("3. Check that output path has .mp4 extension")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create animation of simulated annealing placement optimization'
    )
    parser.add_argument('--design', type=str, required=True,
                       help='Design name (e.g., 6502)')
    parser.add_argument('--fabric-cells', type=str, default='fabric/fabric_cells.yaml',
                       help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', type=str, default='fabric/pins.yaml',
                       help='Path to pins.yaml')
    parser.add_argument('--initial-placement', type=str, default=None,
                       help='Path to initial placement JSON (if None, runs greedy first)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for MP4 video (default: build/{design}/sa_animation.mp4)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for video (default: 10)')
    parser.add_argument('--T-initial', type=float, default=None,
                       help='Initial temperature (auto-calculated if None)')
    parser.add_argument('--alpha', type=float, default=0.95,
                       help='Cooling rate (default: 0.95)')
    parser.add_argument('--T-final', type=float, default=0.1,
                       help='Final temperature (default: 0.1)')
    parser.add_argument('--moves-per-temp', type=int, default=100,
                       help='Moves per temperature step (default: 100)')
    parser.add_argument('--W-initial', type=float, default=None,
                       help='Initial window size (auto-calculated if None)')
    parser.add_argument('--beta', type=float, default=0.98,
                       help='Window cooling rate (default: 0.98)')
    parser.add_argument('--P-refine', type=float, default=0.7,
                       help='Probability of refine move (default: 0.7)')
    parser.add_argument('--P-explore', type=float, default=0.3,
                       help='Probability of explore move (default: 0.3)')
    parser.add_argument('--frame-interval', type=int, default=10,
                       help='Record frame every N iterations (default: 10)')
    
    args = parser.parse_args()
    
    animate_sa_placement(
        args.design,
        args.fabric_cells,
        args.pins,
        args.initial_placement,
        args.output,
        args.fps,
        args.T_initial,
        args.alpha,
        args.T_final,
        args.moves_per_temp,
        args.W_initial,
        args.beta,
        args.P_refine,
        args.P_explore,
        args.frame_interval
    )


if __name__ == '__main__':
    main()
