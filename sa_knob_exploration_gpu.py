#!/usr/bin/env python3
"""
GPU-Accelerated SA Knob Exploration - Runs 60 experiments in parallel with CUDA acceleration.

This script maximizes GPU utilization by:
- Running multiple experiments in parallel using multiprocessing
- Using PyTorch/CUDA for vectorized HPWL calculations
- Showing real-time progress in terminal
- Maximizing resource utilization

Usage:
    # Run all 60 experiments with GPU acceleration (default)
    python sa_knob_exploration_gpu.py --design 6502
    
    # Run with custom number of workers
    python sa_knob_exploration_gpu.py --design 6502 --workers 8
    
    # Disable GPU (CPU-only parallel execution)
    python sa_knob_exploration_gpu.py --design 6502 --no-gpu
    
    # Custom knob ranges (fewer experiments)
    python sa_knob_exploration_gpu.py --design 6502 \
        --alpha-steps 3 --moves-values 50 200 --p-refine-values 0.5 0.9

Requirements:
    - PyTorch (optional, for GPU acceleration): pip install torch
    - Without PyTorch, will use CPU-only parallel execution (still much faster)
"""

import argparse
import time
import csv
import json
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
import threading

# Try to import PyTorch for GPU acceleration
try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        # Only print once, not in every worker process
        if __name__ == '__main__':
            print(f"✅ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        DEVICE = torch.device('cpu')
        if __name__ == '__main__':
            print("⚠️  CUDA not available, using CPU with PyTorch")
        HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    if __name__ == '__main__':
        print("⚠️  PyTorch not available. Install with: pip install torch")
        print("   Will use CPU-only parallel execution (still much faster than sequential)")

from placer import (
    place_design_with_sa,
    calculate_total_hpwl,
    extract_nets,
    get_port_to_net_mapping
)
from parse_fabric import parse_fabric_cells, parse_pins
from parse_design import parse_design

# Default experiment configurations
DEFAULT_ALPHA_VALUES = [0.80, 0.85, 0.90, 0.95, 0.99]
DEFAULT_MOVES_PER_TEMP_VALUES = [50, 100, 200, 500]
DEFAULT_P_REFINE_VALUES = [0.5, 0.7, 0.9]


def calculate_hpwl_gpu(positions: List[Tuple[float, float]], device=None) -> float:
    """
    GPU-accelerated HPWL calculation using PyTorch.
    
    Args:
        positions: List of (x, y) coordinates in microns
        device: PyTorch device (cuda or cpu)
    
    Returns:
        HPWL in microns (float)
    """
    if not HAS_TORCH or device is None:
        # Fallback to CPU numpy
        if len(positions) <= 1:
            return 0.0
        if len(positions) == 2:
            x1, y1 = positions[0]
            x2, y2 = positions[1]
            return abs(x2 - x1) + abs(y2 - y1)
        
        x_coords = [x for x, y in positions]
        y_coords = [y for x, y in positions]
        return (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
    
    if len(positions) <= 1:
        return 0.0
    
    if len(positions) == 2:
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        return abs(x2 - x1) + abs(y2 - y1)
    
    # Convert to tensor and move to GPU
    coords = torch.tensor(positions, dtype=torch.float32, device=device)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Calculate bounding box on GPU
    min_x = torch.min(x_coords)
    max_x = torch.max(x_coords)
    min_y = torch.min(y_coords)
    max_y = torch.max(y_coords)
    
    hpwl = (max_x - min_x) + (max_y - min_y)
    return hpwl.item()


def calculate_total_hpwl_gpu(placement: Dict[str, Dict[str, Any]],
                             nets_dict: Dict[int, List[str]],
                             fabric_db: Dict[str, List[Dict[str, Any]]],
                             pins_db: Optional[Dict[str, Any]] = None,
                             port_to_nets: Optional[Dict[str, List[int]]] = None,
                             device=None) -> float:
    """
    GPU-accelerated total HPWL calculation.
    Uses batched processing when GPU is available.
    
    Args:
        placement: Dict mapping logical_instance_name -> {fabric_slot_name, x, y, orient}
        nets_dict: Dict mapping net_id -> list of cell instances
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        pins_db: Optional pins database
        port_to_nets: Optional port-to-net mapping
        device: PyTorch device
    
    Returns:
        Total HPWL in microns (float)
    """
    # Check if we should use GPU
    if not HAS_TORCH or device is None or (hasattr(device, 'type') and device.type != 'cuda'):
        # Fallback to CPU version
        return calculate_total_hpwl(placement, nets_dict, fabric_db, pins_db, port_to_nets)
    
    total_hpwl = 0.0
    
    # Build slot lookup
    slot_lookup = {}
    for slots in fabric_db.values():
        for slot in slots:
            slot_lookup[slot['name']] = slot
    
    # Process each net (can be optimized further with batching)
    for net_id, cell_list in nets_dict.items():
        positions = []
        
        # Get positions of all cells on this net
        for cell_name in cell_list:
            if cell_name in placement:
                cell_placement = placement[cell_name]
                slot_name = cell_placement.get('fabric_slot_name')
                
                if slot_name and slot_name in slot_lookup:
                    slot = slot_lookup[slot_name]
                    positions.append((slot['x'], slot['y']))
        
        # Add I/O pin positions if this net is connected to pins
        if pins_db and port_to_nets:
            for pin in pins_db.get('pins', []):
                if pin.get('status') == 'FIXED':
                    pin_name = pin['name']
                    if pin_name in port_to_nets:
                        pin_nets = port_to_nets[pin_name]
                        if net_id in pin_nets:
                            positions.append((pin['x_um'], pin['y_um']))
        
        # Calculate HPWL for this net (GPU-accelerated if available)
        net_hpwl = calculate_hpwl_gpu(positions, device)
        total_hpwl += net_hpwl
    
    return total_hpwl


def run_experiment_worker(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel experiment execution.
    This runs in a separate process.
    
    Args:
        args_tuple: Tuple of (experiment_id, design_name, fabric_cells_path, design_path, 
                             pins_path, knob_config, seed, use_gpu)
    
    Returns:
        result: Dict with hpwl, runtime, config, and experiment_id
    """
    (experiment_id, design_name, fabric_cells_path, design_path, pins_path, 
     knob_config, seed, use_gpu) = args_tuple
    
    # Print experiment start (only in main process, not in workers)
    # Workers can't print directly, so we'll handle this in the main loop
    
    # Set random seed if provided
    if seed is not None:
        # Use experiment_id to create unique seed per experiment
        unique_seed = seed + experiment_id if seed is not None else None
        random.seed(unique_seed)
        np.random.seed(unique_seed)
        if HAS_TORCH:
            torch.manual_seed(unique_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(unique_seed)
    
    # Extract knob settings
    alpha = knob_config.get('alpha', 0.95)
    moves_per_temp = knob_config.get('moves_per_temp', 100)
    p_refine = knob_config.get('p_refine', 0.7)
    p_explore = knob_config.get('p_explore', 0.3)
    T_final = knob_config.get('T_final', 0.1)
    
    # Set device for this worker
    device = None
    if use_gpu and HAS_TORCH:
        try:
            if torch.cuda.is_available():
                # Use default CUDA device
                device = torch.device('cuda')
                # Set CUDA device for this thread (important for threading)
                torch.cuda.set_device(0)
            else:
                device = None
        except Exception as e:
            print(f"⚠️  CUDA initialization failed: {e}")
            device = None
    
    # Record start time
    start_time = time.time()
    
    # Run placement
    try:
        placement = place_design_with_sa(
            fabric_cells_path,
            design_path,
            pins_path,
            enable_sa=True,
            sa_alpha=alpha,
            sa_moves_per_temp=moves_per_temp,
            sa_T_final=T_final
        )
        
        if placement is None:
            return {
                'experiment_id': experiment_id,
                'hpwl': float('inf'),
                'runtime': time.time() - start_time,
                'config': knob_config,
                'success': False,
                'gpu_used': False
            }
        
        # Calculate final HPWL (with GPU acceleration if available)
        fabric_db = parse_fabric_cells(fabric_cells_path)
        _, netlist_graph = parse_design(design_path)
        pins_db = parse_pins(pins_path)
        nets_dict = extract_nets(netlist_graph)
        port_to_nets = get_port_to_net_mapping(design_path)
        
        # Use GPU-accelerated HPWL if available
        # Note: The SA algorithm itself still runs on CPU, but final HPWL uses GPU
        gpu_used = False
        if device is not None and HAS_TORCH:
            try:
                # Check if device is actually CUDA
                if hasattr(device, 'type') and device.type == 'cuda':
                    final_hpwl = calculate_total_hpwl_gpu(
                        placement, nets_dict, fabric_db, pins_db, port_to_nets, device
                    )
                    gpu_used = True
                else:
                    final_hpwl = calculate_total_hpwl(
                        placement, nets_dict, fabric_db, pins_db, port_to_nets
                    )
            except Exception as e:
                # Fallback to CPU if GPU fails
                final_hpwl = calculate_total_hpwl(
                    placement, nets_dict, fabric_db, pins_db, port_to_nets
                )
        else:
            final_hpwl = calculate_total_hpwl(
                placement, nets_dict, fabric_db, pins_db, port_to_nets
            )
        
        runtime = time.time() - start_time
        
        return {
            'experiment_id': experiment_id,
            'hpwl': final_hpwl,
            'runtime': runtime,
            'config': knob_config,
            'success': True,
            'gpu_used': gpu_used
        }
    
    except Exception as e:
        return {
            'experiment_id': experiment_id,
            'hpwl': float('inf'),
            'runtime': time.time() - start_time,
            'config': knob_config,
            'success': False,
            'error': str(e),
            'gpu_used': False
        }


def run_all_experiments_parallel(design_name: str,
                                 fabric_cells_path: str,
                                 design_path: str,
                                 pins_path: str,
                                 alpha_values: List[float],
                                 moves_per_temp_values: List[int],
                                 p_refine_values: List[float],
                                 T_final: float = 0.1,
                                 seed: int = None,
                                 output_csv: str = None,
                                 max_workers: int = None,
                                 use_gpu: bool = True) -> List[Dict[str, Any]]:
    """
    Run all combinations of knob settings in parallel using multiprocessing.
    
    Args:
        design_name: Name of the design
        fabric_cells_path: Path to fabric_cells.yaml
        design_path: Path to design mapped JSON file
        pins_path: Path to pins.yaml
        alpha_values: List of cooling rate values to test
        moves_per_temp_values: List of moves per temperature values to test
        p_refine_values: List of P_refine values to test
        T_final: Final temperature (default: 0.1)
        seed: Random seed for reproducibility (optional)
        output_csv: Path to save results CSV (optional)
        max_workers: Maximum number of parallel workers (default: CPU count)
        use_gpu: Whether to use GPU acceleration (default: True)
    
    Returns:
        results: List of result dicts
    """
    # Generate all combinations
    all_configs = []
    for alpha, moves, p_ref in itertools.product(alpha_values, moves_per_temp_values, p_refine_values):
        p_explore = 1.0 - p_ref  # Ensure probabilities sum to 1.0
        all_configs.append({
            'alpha': alpha,
            'moves_per_temp': moves,
            'p_refine': p_ref,
            'p_explore': p_explore,
            'T_final': T_final
        })
    
    # Determine number of workers and execution method
    use_threading = False
    if max_workers is None:
        if use_gpu and HAS_TORCH and torch.cuda.is_available():
            # Use threading for GPU to share CUDA context
            # Limit to avoid GPU memory issues
            max_workers = min(4, len(all_configs))  # GPU can handle 4-8 concurrent experiments
            use_threading = True
        else:
            # Use multiprocessing for CPU
            max_workers = mp.cpu_count()
    elif use_gpu and HAS_TORCH and torch.cuda.is_available():
        use_threading = True
    
    print(f"\n{'='*60}")
    print(f"GPU-Accelerated SA Knob Exploration: {design_name}")
    print(f"{'='*60}")
    print(f"Total experiments: {len(all_configs)} ({len(alpha_values)} alpha × {len(moves_per_temp_values)} moves × {len(p_refine_values)} p_refine)")
    print(f"Alpha values: {alpha_values}")
    print(f"Moves per temp: {moves_per_temp_values}")
    print(f"P_refine values: {p_refine_values}")
    print(f"Parallel workers: {max_workers}")
    if use_gpu and HAS_TORCH and torch.cuda.is_available():
        print(f"GPU acceleration: ✅ ENABLED ({torch.cuda.get_device_name(0)})")
    else:
        print(f"GPU acceleration: ❌ DISABLED (CPU only)")
    print(f"\n💡 Press Ctrl+C at any time to stop and save progress")
    print(f"{'='*60}\n")
    
    # Prepare arguments for workers
    worker_args = []
    for i, config in enumerate(all_configs):
        worker_args.append((
            i,  # experiment_id
            design_name,
            fabric_cells_path,
            design_path,
            pins_path,
            config,
            seed,
            use_gpu
        ))
    
    results = []
    completed_count = 0
    start_times = {}  # Track start time for each experiment
    
    # Run experiments in parallel with progress bar
    # Use ThreadPoolExecutor for GPU (shares CUDA context) or ProcessPoolExecutor for CPU
    executor_class = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
    
    try:
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks and print start messages
            future_to_config = {}
            for args in worker_args:
                experiment_id = args[0]
                config = args[5]
                exp_num = experiment_id + 1  # 1-indexed
                future = executor.submit(run_experiment_worker, args)
                future_to_config[future] = (experiment_id, config)
                
                # Print experiment start with GPU status
                gpu_available = use_gpu and HAS_TORCH and torch.cuda.is_available()
                gpu_indicator = "🚀 GPU" if gpu_available else "💻 CPU"
                print(f"🚀 Starting Experiment {exp_num}/60 ({gpu_indicator}) | "
                      f"Config: α={config['alpha']:.2f}, moves={config['moves_per_temp']}, p_ref={config['p_refine']:.2f}")
            
            # Process completed tasks with progress bar
            with tqdm(total=len(all_configs), desc="Running experiments", 
                     unit="exp", ncols=120, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for future in as_completed(future_to_config):
                    experiment_id, config = future_to_config[future]
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    # Calculate time taken for this experiment
                    exp_time = result['runtime']
                    exp_num = experiment_id + 1  # 1-indexed for display
                    
                    # Print experiment completion info
                    gpu_status = "🚀 GPU" if result.get('gpu_used', False) else "💻 CPU"
                    if result['success']:
                        print(f"\n✅ Experiment {exp_num}/60 COMPLETED in {exp_time:.1f}s ({gpu_status}) | "
                              f"HPWL: {result['hpwl']:.2f} um | "
                              f"Config: α={config['alpha']:.2f}, moves={config['moves_per_temp']}, p_ref={config['p_refine']:.2f}")
                    else:
                        print(f"\n❌ Experiment {exp_num}/60 FAILED in {exp_time:.1f}s ({gpu_status}) | "
                              f"Error: {result.get('error', 'Unknown')[:50]}")
                    
                    # Update progress bar with current status
                    if result['success']:
                        pbar.set_postfix({
                            'Exp': f"{exp_num}/60",
                            'HPWL': f"{result['hpwl']:.0f}",
                            'Time': f"{exp_time:.1f}s",
                            'Status': '✅'
                        })
                    else:
                        pbar.set_postfix({
                            'Exp': f"{exp_num}/60",
                            'Status': '❌',
                            'Error': result.get('error', 'Unknown')[:15]
                        })
                    
                    pbar.update(1)
                    
                    # Save incrementally after each experiment
                    if output_csv:
                        # Sort results by experiment_id to maintain order
                        sorted_results = sorted(results, key=lambda x: x['experiment_id'])
                        save_results_csv(sorted_results, output_csv)
                        print(f"💾 Progress saved: {completed_count}/60 experiments")
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⚠️  INTERRUPTED BY USER (Ctrl+C)")
        print("="*60)
        print(f"Completed {completed_count}/{len(all_configs)} experiments before interruption")
        
        # Save current progress
        if results:
            sorted_results = sorted(results, key=lambda x: x['experiment_id'])
            if output_csv:
                save_results_csv(sorted_results, output_csv)
                print(f"✅ Results saved to: {output_csv}")
            
            # Generate plot with available data
            plot_path = output_csv.replace('.csv', '_partial.png') if output_csv else 'sa_knob_analysis_partial.png'
            print(f"📊 Generating plot with {len(results)} completed experiments...")
            plot_knob_analysis(sorted_results, plot_path)
            print(f"✅ Plot saved to: {plot_path}")
        
        print("="*60 + "\n")
        raise  # Re-raise to exit gracefully
    
    # Sort results by experiment_id to match original order
    results.sort(key=lambda x: x['experiment_id'])
    
    # Remove experiment_id from results for compatibility
    for result in results:
        result.pop('experiment_id', None)
    
    return results


def save_results_csv(results: List[Dict[str, Any]], output_path: str):
    """Save experiment results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'moves_per_temp', 'p_refine', 'p_explore', 'hpwl', 'runtime', 'success'])
        
        for result in results:
            config = result['config']
            writer.writerow([
                config['alpha'],
                config['moves_per_temp'],
                config['p_refine'],
                config['p_explore'],
                result['hpwl'],
                result['runtime'],
                result['success']
            ])


def load_results_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load experiment results from CSV file."""
    results = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'hpwl': float(row['hpwl']),
                'runtime': float(row['runtime']),
                'config': {
                    'alpha': float(row['alpha']),
                    'moves_per_temp': int(row['moves_per_temp']),
                    'p_refine': float(row['p_refine']),
                    'p_explore': float(row['p_explore'])
                },
                'success': row['success'].lower() == 'true'
            })
    
    return results


def find_pareto_frontier(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find Pareto-optimal points (best HPWL for given runtime, or fastest runtime for given HPWL).
    """
    successful_results = [r for r in results if r['success'] and r['hpwl'] != float('inf')]
    
    if not successful_results:
        return []
    
    pareto_points = []
    
    for point in successful_results:
        is_pareto = True
        
        for other in successful_results:
            if (other['hpwl'] < point['hpwl'] and other['runtime'] <= point['runtime']) or \
               (other['hpwl'] <= point['hpwl'] and other['runtime'] < point['runtime']):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_points.append(point)
    
    pareto_points.sort(key=lambda x: x['runtime'])
    
    return pareto_points


def plot_knob_analysis(results: List[Dict[str, Any]], output_path: str):
    """Generate scatter plot showing Runtime vs HPWL with Pareto frontier."""
    successful_results = [r for r in results if r['success'] and r['hpwl'] != float('inf')]
    
    if not successful_results:
        print("ERROR: No successful results to plot")
        return
    
    pareto_points = find_pareto_frontier(successful_results)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    runtimes = [r['runtime'] for r in successful_results]
    hpwls = [r['hpwl'] for r in successful_results]
    alphas = [r['config']['alpha'] for r in successful_results]
    
    scatter = ax.scatter(runtimes, hpwls, c=alphas, cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    if pareto_points:
        pareto_runtimes = [p['runtime'] for p in pareto_points]
        pareto_hpwls = [p['hpwl'] for p in pareto_points]
        ax.plot(pareto_runtimes, pareto_hpwls, 'r--', linewidth=2, 
                label='Pareto Frontier', zorder=10)
        ax.scatter(pareto_runtimes, pareto_hpwls, c='red', s=150, 
                  marker='*', edgecolors='black', linewidth=1, 
                  label='Pareto Points', zorder=11)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Cooling Rate (alpha)')
    
    ax.set_xlabel('Runtime (seconds)', fontsize=12)
    ax.set_ylabel('Final HPWL (microns)', fontsize=12)
    ax.set_title('SA Knob Analysis: Runtime vs HPWL Trade-off (GPU-Accelerated)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    best_hpwl = min(hpwls)
    fastest_runtime = min(runtimes)
    best_hpwl_config = min(successful_results, key=lambda x: x['hpwl'])['config']
    fastest_config = min(successful_results, key=lambda x: x['runtime'])['config']
    
    stats_text = (f'Total Experiments: {len(successful_results)}\n'
                  f'Best HPWL: {best_hpwl:.2f} um\n'
                  f'  (alpha={best_hpwl_config["alpha"]:.2f}, '
                  f'moves={best_hpwl_config["moves_per_temp"]}, '
                  f'p_refine={best_hpwl_config["p_refine"]:.2f})\n'
                  f'Fastest Runtime: {fastest_runtime:.2f} s\n'
                  f'  (alpha={fastest_config["alpha"]:.2f}, '
                  f'moves={fastest_config["moves_per_temp"]}, '
                  f'p_refine={fastest_config["p_refine"]:.2f})\n'
                  f'Pareto Points: {len(pareto_points)}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if pareto_points:
        pareto_annotation = ('Pareto Frontier: Points where no other\n'
                            'configuration has both lower HPWL\n'
                            'and lower runtime. Optimal trade-offs.')
        ax.text(0.98, 0.02, pareto_annotation, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Knob analysis plot saved to: {output_path}")


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze experiment results and provide recommendations."""
    successful_results = [r for r in results if r['success'] and r['hpwl'] != float('inf')]
    
    if not successful_results:
        return {'error': 'No successful results to analyze'}
    
    best_hpwl_result = min(successful_results, key=lambda x: x['hpwl'])
    fastest_result = min(successful_results, key=lambda x: x['runtime'])
    pareto_points = find_pareto_frontier(successful_results)
    
    if pareto_points:
        max_runtime = max(r['runtime'] for r in successful_results)
        max_hpwl = max(r['hpwl'] for r in successful_results)
        min_runtime = min(r['runtime'] for r in successful_results)
        min_hpwl = min(r['hpwl'] for r in successful_results)
        
        runtime_range = max_runtime - min_runtime if max_runtime > min_runtime else 1.0
        hpwl_range = max_hpwl - min_hpwl if max_hpwl > min_hpwl else 1.0
        
        best_tradeoff = None
        min_distance = float('inf')
        
        for point in pareto_points:
            norm_runtime = (point['runtime'] - min_runtime) / runtime_range if runtime_range > 0 else 0
            norm_hpwl = (point['hpwl'] - min_hpwl) / hpwl_range if hpwl_range > 0 else 0
            distance = np.sqrt(norm_runtime**2 + norm_hpwl**2)
            
            if distance < min_distance:
                min_distance = distance
                best_tradeoff = point
    else:
        best_tradeoff = None
    
    analysis = {
        'total_experiments': len(results),
        'successful_experiments': len(successful_results),
        'best_hpwl': {
            'hpwl': best_hpwl_result['hpwl'],
            'runtime': best_hpwl_result['runtime'],
            'config': best_hpwl_result['config']
        },
        'fastest_runtime': {
            'hpwl': fastest_result['hpwl'],
            'runtime': fastest_result['runtime'],
            'config': fastest_result['config']
        },
        'pareto_points_count': len(pareto_points),
        'recommended_default': None
    }
    
    if best_tradeoff:
        analysis['recommended_default'] = {
            'hpwl': best_tradeoff['hpwl'],
            'runtime': best_tradeoff['runtime'],
            'config': best_tradeoff['config'],
            'reason': 'Best trade-off on Pareto frontier'
        }
    else:
        analysis['recommended_default'] = {
            'hpwl': best_hpwl_result['hpwl'],
            'runtime': best_hpwl_result['runtime'],
            'config': best_hpwl_result['config'],
            'reason': 'Best HPWL (no Pareto analysis available)'
        }
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print analysis results to console."""
    print("\n" + "="*60)
    print("SA Knob Analysis Results (GPU-Accelerated)")
    print("="*60)
    print(f"Total Experiments: {analysis['total_experiments']}")
    print(f"Successful Experiments: {analysis['successful_experiments']}")
    print()
    
    print("Best HPWL Configuration:")
    best = analysis['best_hpwl']
    print(f"  HPWL: {best['hpwl']:.2f} um")
    print(f"  Runtime: {best['runtime']:.2f} s")
    print(f"  Config: alpha={best['config']['alpha']:.2f}, "
          f"moves={best['config']['moves_per_temp']}, "
          f"p_refine={best['config']['p_refine']:.2f}")
    print()
    
    print("Fastest Runtime Configuration:")
    fastest = analysis['fastest_runtime']
    print(f"  HPWL: {fastest['hpwl']:.2f} um")
    print(f"  Runtime: {fastest['runtime']:.2f} s")
    print(f"  Config: alpha={fastest['config']['alpha']:.2f}, "
          f"moves={fastest['config']['moves_per_temp']}, "
          f"p_refine={fastest['config']['p_refine']:.2f}")
    print()
    
    print("Recommended Default Configuration:")
    recommended = analysis['recommended_default']
    print(f"  HPWL: {recommended['hpwl']:.2f} um")
    print(f"  Runtime: {recommended['runtime']:.2f} s")
    print(f"  Config: alpha={recommended['config']['alpha']:.2f}, "
          f"moves={recommended['config']['moves_per_temp']}, "
          f"p_refine={recommended['config']['p_refine']:.2f}")
    print(f"  Reason: {recommended['reason']}")
    print()
    
    print(f"Pareto Points Found: {analysis['pareto_points_count']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated SA knob exploration: Run experiments in parallel with CUDA'
    )
    parser.add_argument('--design', type=str, required=True,
                        help='Design name (e.g., 6502)')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                        help='Path to pins.yaml')
    parser.add_argument('--netlist', type=str, default=None,
                        help='Path to design mapped JSON (default: designs/[design]_mapped.json)')
    parser.add_argument('--output', type=str, default='sa_knob_analysis_gpu.png',
                        help='Output PNG file path')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to save/load results CSV')
    parser.add_argument('--load-csv', type=str, default=None,
                        help='Load results from CSV instead of running experiments')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration (CPU only)')
    
    # Knob ranges
    parser.add_argument('--alpha-min', type=float, default=0.80,
                        help='Minimum alpha value (default: 0.80)')
    parser.add_argument('--alpha-max', type=float, default=0.99,
                        help='Maximum alpha value (default: 0.99)')
    parser.add_argument('--alpha-steps', type=int, default=5,
                        help='Number of alpha values to test (default: 5)')
    parser.add_argument('--moves-values', type=int, nargs='+', default=DEFAULT_MOVES_PER_TEMP_VALUES,
                        help='Moves per temperature values to test (default: 50 100 200 500)')
    parser.add_argument('--p-refine-values', type=float, nargs='+', default=DEFAULT_P_REFINE_VALUES,
                        help='P_refine values to test (default: 0.5 0.7 0.9)')
    
    args = parser.parse_args()
    
    # Set default netlist path
    if args.netlist is None:
        args.netlist = f'designs/{args.design}_mapped.json'
    
    # Set default CSV path
    if args.csv is None:
        args.csv = f'sa_knob_results_{args.design}_gpu.csv'
    
    # Load results from CSV or run experiments
    if args.load_csv:
        print(f"Loading results from: {args.load_csv}")
        results = load_results_csv(args.load_csv)
    else:
        # Determine alpha values
        if args.alpha_steps == 5 and args.alpha_min == 0.80 and args.alpha_max == 0.99:
            alpha_values = DEFAULT_ALPHA_VALUES
        else:
            alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps).tolist()
        
        moves_values = args.moves_values
        p_refine_values = args.p_refine_values
        
        # Run experiments in parallel
        start_time = time.time()
        try:
            results = run_all_experiments_parallel(
                args.design,
                args.fabric_cells,
                args.netlist,
                args.pins,
                alpha_values,
                moves_values,
                p_refine_values,
                seed=args.seed,
                output_csv=args.csv,
                max_workers=args.workers,
                use_gpu=not args.no_gpu
            )
            total_time = time.time() - start_time
            
            # Count GPU vs CPU usage
            gpu_count = sum(1 for r in results if r.get('gpu_used', False))
            cpu_count = len(results) - gpu_count
            
            print(f"\n{'='*60}")
            print(f"✅ All {len(results)} experiments completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"Average time per experiment: {total_time/len(results):.2f} seconds")
            print(f"GPU acceleration: {gpu_count} experiments used 🚀 GPU, {cpu_count} used 💻 CPU")
            print(f"{'='*60}\n")
        except KeyboardInterrupt:
            # Results already saved in run_all_experiments_parallel
            # Just exit gracefully
            print("\n👋 Exiting gracefully. Your progress has been saved!")
            return
    
    # Generate plot (works with partial results too)
    if results:
        print(f"\n📊 Generating analysis plot with {len(results)} experiments...")
        plot_knob_analysis(results, args.output)
        
        # Analyze and print results
        analysis = analyze_results(results)
        print_analysis(analysis)
        
        # Save analysis to JSON
        analysis_json_path = args.output.replace('.png', '_analysis.json')
        with open(analysis_json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✅ Analysis saved to: {analysis_json_path}")
    else:
        print("\n⚠️  No results to analyze.")


if __name__ == '__main__':
    main()

