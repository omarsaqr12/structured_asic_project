#!/usr/bin/env python3
"""
SA Knob Exploration and Analysis (Issue #9).

This script explores how different Simulated Annealing parameters (knobs) affect
placement quality (HPWL) and runtime. It generates a scatter plot showing the
trade-off between runtime and HPWL, identifies the Pareto frontier, and provides
recommendations for optimal knob settings.
"""

import argparse
import time
import csv
import json
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
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


def run_experiment(design_name: str,
                  fabric_cells_path: str,
                  design_path: str,
                  pins_path: str,
                  knob_config: Dict[str, Any],
                  seed: int = None) -> Dict[str, Any]:
    """
    Run a single placement experiment with specified knob settings.
    
    Args:
        design_name: Name of the design (e.g., '6502')
        fabric_cells_path: Path to fabric_cells.yaml
        design_path: Path to design mapped JSON file
        pins_path: Path to pins.yaml
        knob_config: Dict with SA parameters (alpha, moves_per_temp, p_refine, etc.)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        result: Dict with hpwl, runtime, and config
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Extract knob settings
    alpha = knob_config.get('alpha', 0.95)
    moves_per_temp = knob_config.get('moves_per_temp', 100)
    p_refine = knob_config.get('p_refine', 0.7)
    p_explore = knob_config.get('p_explore', 0.3)
    T_final = knob_config.get('T_final', 0.1)
    
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
                'hpwl': float('inf'),
                'runtime': time.time() - start_time,
                'config': knob_config,
                'success': False
            }
        
        # Calculate final HPWL
        fabric_db = parse_fabric_cells(fabric_cells_path)
        _, netlist_graph = parse_design(design_path)
        pins_db = parse_pins(pins_path)
        nets_dict = extract_nets(netlist_graph)
        port_to_nets = get_port_to_net_mapping(design_path)
        
        final_hpwl = calculate_total_hpwl(
            placement, nets_dict, fabric_db, pins_db, port_to_nets
        )
        
        runtime = time.time() - start_time
        
        return {
            'hpwl': final_hpwl,
            'runtime': runtime,
            'config': knob_config,
            'success': True
        }
    
    except Exception as e:
        print(f"ERROR in experiment with config {knob_config}: {e}")
        return {
            'hpwl': float('inf'),
            'runtime': time.time() - start_time,
            'config': knob_config,
            'success': False,
            'error': str(e)
        }


def run_all_experiments(design_name: str,
                       fabric_cells_path: str,
                       design_path: str,
                       pins_path: str,
                       alpha_values: List[float],
                       moves_per_temp_values: List[int],
                       p_refine_values: List[float],
                       T_final: float = 0.1,
                       seed: int = None,
                       output_csv: str = None) -> List[Dict[str, Any]]:
    """
    Run all combinations of knob settings.
    
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
    
    print(f"Running {len(all_configs)} experiments...")
    print(f"Design: {design_name}")
    print(f"Alpha values: {alpha_values}")
    print(f"Moves per temp: {moves_per_temp_values}")
    print(f"P_refine values: {p_refine_values}")
    print()
    
    results = []
    
    for i, config in enumerate(all_configs, 1):
        print(f"Experiment {i}/{len(all_configs)}: alpha={config['alpha']:.2f}, "
              f"moves={config['moves_per_temp']}, p_refine={config['p_refine']:.2f}")
        
        result = run_experiment(
            design_name, fabric_cells_path, design_path, pins_path, config, seed
        )
        results.append(result)
        
        if result['success']:
            print(f"  → HPWL: {result['hpwl']:.2f} um, Runtime: {result['runtime']:.2f} s")
        else:
            print(f"  → FAILED")
        print()
    
    # Save to CSV if requested
    if output_csv:
        save_results_csv(results, output_csv)
    
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
    
    print(f"Results saved to: {output_path}")


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
    
    A point is Pareto-optimal if no other point has both lower HPWL and lower runtime.
    
    Args:
        results: List of result dicts
    
    Returns:
        pareto_points: List of Pareto-optimal result dicts
    """
    # Filter successful results
    successful_results = [r for r in results if r['success'] and r['hpwl'] != float('inf')]
    
    if not successful_results:
        return []
    
    pareto_points = []
    
    for point in successful_results:
        is_pareto = True
        
        # Check if any other point dominates this one
        for other in successful_results:
            if (other['hpwl'] < point['hpwl'] and other['runtime'] <= point['runtime']) or \
               (other['hpwl'] <= point['hpwl'] and other['runtime'] < point['runtime']):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_points.append(point)
    
    # Sort by runtime for plotting
    pareto_points.sort(key=lambda x: x['runtime'])
    
    return pareto_points


def plot_knob_analysis(results: List[Dict[str, Any]], output_path: str):
    """
    Generate scatter plot showing Runtime vs HPWL with Pareto frontier.
    
    Args:
        results: List of result dicts
        output_path: Path to output PNG file
    """
    # Filter successful results
    successful_results = [r for r in results if r['success'] and r['hpwl'] != float('inf')]
    
    if not successful_results:
        print("ERROR: No successful results to plot")
        return
    
    # Find Pareto frontier
    pareto_points = find_pareto_frontier(successful_results)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Extract data
    runtimes = [r['runtime'] for r in successful_results]
    hpwls = [r['hpwl'] for r in successful_results]
    alphas = [r['config']['alpha'] for r in successful_results]
    
    # Create scatter plot, color-coded by alpha value
    scatter = ax.scatter(runtimes, hpwls, c=alphas, cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot Pareto frontier
    if pareto_points:
        pareto_runtimes = [p['runtime'] for p in pareto_points]
        pareto_hpwls = [p['hpwl'] for p in pareto_points]
        ax.plot(pareto_runtimes, pareto_hpwls, 'r--', linewidth=2, 
                label='Pareto Frontier', zorder=10)
        ax.scatter(pareto_runtimes, pareto_hpwls, c='red', s=150, 
                  marker='*', edgecolors='black', linewidth=1, 
                  label='Pareto Points', zorder=11)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Cooling Rate (alpha)')
    
    # Set axis properties
    ax.set_xlabel('Runtime (seconds)', fontsize=12)
    ax.set_ylabel('Final HPWL (microns)', fontsize=12)
    ax.set_title('SA Knob Analysis: Runtime vs HPWL Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add statistics text
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
    
    # Add annotation explaining Pareto frontier
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
    """
    Analyze experiment results and provide recommendations.
    
    Args:
        results: List of result dicts
    
    Returns:
        analysis: Dict with recommendations and statistics
    """
    successful_results = [r for r in results if r['success'] and r['hpwl'] != float('inf')]
    
    if not successful_results:
        return {'error': 'No successful results to analyze'}
    
    # Find best configurations
    best_hpwl_result = min(successful_results, key=lambda x: x['hpwl'])
    fastest_result = min(successful_results, key=lambda x: x['runtime'])
    
    # Find Pareto frontier
    pareto_points = find_pareto_frontier(successful_results)
    
    # Find best trade-off (point on Pareto frontier closest to origin)
    if pareto_points:
        # Normalize runtime and HPWL to [0, 1] for distance calculation
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
        # Fallback: use configuration with good balance
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
    print("SA Knob Analysis Results")
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


def format_analysis_for_readme(analysis: Dict[str, Any], design_name: str = None) -> str:
    """
    Format analysis results as markdown text for README.md.
    
    Args:
        analysis: Analysis dict from analyze_results()
        design_name: Optional design name for context
    
    Returns:
        markdown_text: Formatted markdown string
    """
    lines = []
    
    if design_name:
        lines.append(f"## SA Knob Analysis - {design_name}")
    else:
        lines.append("## SA Knob Analysis")
    
    lines.append("")
    lines.append("### Experiment Description")
    lines.append("")
    lines.append("This analysis explores the impact of Simulated Annealing (SA) parameters on placement quality and runtime:")
    lines.append("- **Cooling Rate (alpha)**: Controls temperature decay rate (0.80-0.99)")
    lines.append("- **Moves per Temperature**: Number of moves attempted at each temperature (50-500)")
    lines.append("- **P_refine**: Probability of using refine moves vs. explore moves (0.5-0.9)")
    lines.append("")
    lines.append(f"Total experiments run: {analysis['total_experiments']}")
    lines.append(f"Successful experiments: {analysis['successful_experiments']}")
    lines.append("")
    
    lines.append("### Key Findings")
    lines.append("")
    
    # Best HPWL
    best = analysis['best_hpwl']
    lines.append("#### Best HPWL Configuration")
    lines.append(f"- **HPWL**: {best['hpwl']:.2f} μm")
    lines.append(f"- **Runtime**: {best['runtime']:.2f} seconds")
    lines.append(f"- **Parameters**: alpha={best['config']['alpha']:.2f}, moves_per_temp={best['config']['moves_per_temp']}, p_refine={best['config']['p_refine']:.2f}")
    lines.append("")
    
    # Fastest Runtime
    fastest = analysis['fastest_runtime']
    lines.append("#### Fastest Runtime Configuration")
    lines.append(f"- **HPWL**: {fastest['hpwl']:.2f} μm")
    lines.append(f"- **Runtime**: {fastest['runtime']:.2f} seconds")
    lines.append(f"- **Parameters**: alpha={fastest['config']['alpha']:.2f}, moves_per_temp={fastest['config']['moves_per_temp']}, p_refine={fastest['config']['p_refine']:.2f}")
    lines.append("")
    
    # Recommended Default
    recommended = analysis['recommended_default']
    lines.append("#### Recommended Default Configuration")
    lines.append(f"- **HPWL**: {recommended['hpwl']:.2f} μm")
    lines.append(f"- **Runtime**: {recommended['runtime']:.2f} seconds")
    lines.append(f"- **Parameters**: alpha={recommended['config']['alpha']:.2f}, moves_per_temp={recommended['config']['moves_per_temp']}, p_refine={recommended['config']['p_refine']:.2f}")
    lines.append(f"- **Justification**: {recommended['reason']}")
    lines.append("")
    
    lines.append("### Visualization")
    lines.append("")
    lines.append("The following plot shows the trade-off between runtime and HPWL, with points color-coded by alpha value:")
    lines.append("")
    lines.append("![SA Knob Analysis](sa_knob_analysis.png)")
    lines.append("")
    lines.append("**Pareto Frontier**: The red dashed line and star markers indicate Pareto-optimal configurations where no other configuration has both lower HPWL and lower runtime. These represent the best trade-offs between quality and speed.")
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Explore SA knob settings and analyze trade-offs'
    )
    parser.add_argument('--design', type=str, required=True,
                        help='Design name (e.g., 6502)')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                        help='Path to pins.yaml')
    parser.add_argument('--netlist', type=str, default=None,
                        help='Path to design mapped JSON (default: designs/[design]_mapped.json)')
    parser.add_argument('--output', type=str, default='sa_knob_analysis.png',
                        help='Output PNG file path')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to save/load results CSV')
    parser.add_argument('--load-csv', type=str, default=None,
                        help='Load results from CSV instead of running experiments')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
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
        args.csv = f'sa_knob_results_{args.design}.csv'
    
    # Load results from CSV or run experiments
    if args.load_csv:
        print(f"Loading results from: {args.load_csv}")
        results = load_results_csv(args.load_csv)
    else:
        # Determine alpha values: use exact defaults if using default range, otherwise generate
        if args.alpha_steps == 5 and args.alpha_min == 0.80 and args.alpha_max == 0.99:
            alpha_values = DEFAULT_ALPHA_VALUES
        else:
            alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps).tolist()
        
        # Use provided values (defaults are already set in argparse, so use them directly)
        moves_values = args.moves_values
        p_refine_values = args.p_refine_values
        
        # Run experiments (itertools.product is used inside run_all_experiments)
        results = run_all_experiments(
            args.design,
            args.fabric_cells,
            args.netlist,
            args.pins,
            alpha_values,
            moves_values,
            p_refine_values,
            seed=args.seed,
            output_csv=args.csv
        )
    
    # Generate plot
    plot_knob_analysis(results, args.output)
    
    # Analyze and print results
    analysis = analyze_results(results)
    print_analysis(analysis)
    
    # Save analysis to JSON
    analysis_json_path = args.output.replace('.png', '_analysis.json')
    with open(analysis_json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_json_path}")


if __name__ == '__main__':
    main()

