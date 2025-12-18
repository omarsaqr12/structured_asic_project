#!/usr/bin/env python3
"""
Script to run all needed variations of the placer ensuring they don't collide.

This script runs:
1. Greedy-only placement (no SA) for all designs
2. SA placement with various parameter combinations for all designs

Output structure:
  build/
    {design}/
      greedy/
        {design}.map
        {design}_placement.json
      sa_alpha{alpha}_moves{moves}_Tfinal{Tfinal}/
        {design}.map
        {design}_placement.json

Based on requirements from PHASE2_ISSUES.md Issue #9:
- Cooling rate (alpha): 0.80, 0.85, 0.90, 0.95, 0.99
- Moves per temperature (N): 50, 100, 200, 500
- Final temperature (T_final): 0.1, 0.01, 0.001
"""

import os
import sys
import subprocess
import time
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import argparse


# Design files available
DESIGNS = [
    'designs/6502_mapped.json',
    'designs/aes_128_mapped.json',
    'designs/arith_mapped.json',
    'designs/z80_mapped.json'
]

# SA Parameter variations (from PHASE2_ISSUES.md Issue #9)
SA_ALPHA_VALUES = [0.80, 0.85, 0.90, 0.95, 0.99]
SA_MOVES_VALUES = [50, 100, 200, 500]
SA_T_FINAL_VALUES = [0.1, 0.01, 0.001]

# Default paths
FABRIC_CELLS = 'fabric/fabric_cells.yaml'
PINS = 'fabric/pins.yaml'


def extract_design_name(design_path: str) -> str:
    """Extract design name from design file path."""
    basename = os.path.basename(design_path)
    if basename.endswith('_mapped.json'):
        return basename[:-12]  # Remove '_mapped.json'
    elif basename.endswith('.json'):
        return basename[:-5]  # Remove '.json'
    else:
        return basename


def get_output_dir(design_path: str, config: Dict) -> str:
    """
    Generate unique output directory for a configuration.
    
    Args:
        design_path: Path to design file
        config: Configuration dict with keys: 'no_sa', 'alpha', 'moves', 't_final'
    
    Returns:
        Output directory path
    """
    design_name = extract_design_name(design_path)
    base_dir = f'build/{design_name}'
    
    if config.get('no_sa', False):
        return f'{base_dir}/greedy'
    else:
        alpha = config.get('alpha', 0.95)
        moves = config.get('moves', 100)
        t_final = config.get('t_final', 0.1)
        # Format: sa_alpha0.95_moves100_Tfinal0.1
        return f'{base_dir}/sa_alpha{alpha}_moves{moves}_Tfinal{t_final}'


def extract_hpwl_from_output(output: str) -> Optional[float]:
    """
    Extract Final Total HPWL from placer output.
    
    Looks for: "Final Total HPWL: X.XX um"
    
    Args:
        output: Placer stdout/stderr text
    
    Returns:
        HPWL value in microns, or None if not found
    """
    # Pattern: "Final Total HPWL: 12345.67 um"
    pattern = r'Final Total HPWL:\s+([\d.]+)\s+um'
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    return None


def calculate_hpwl_from_placement(placement_json_path: str, design_path: str) -> Optional[float]:
    """
    Calculate HPWL from placement JSON file.
    
    Args:
        placement_json_path: Path to placement JSON file
        design_path: Path to design file
    
    Returns:
        HPWL value in microns, or None if calculation fails
    """
    try:
        # Import placer functions
        from placer import calculate_total_hpwl, extract_nets
        from parse_fabric import parse_fabric_cells, parse_pins
        from parse_design import parse_design
        from placer import get_port_to_net_mapping
        
        # Load placement
        with open(placement_json_path, 'r') as f:
            placement = json.load(f)
        
        # Load required data
        fabric_db = parse_fabric_cells(FABRIC_CELLS)
        _, netlist_graph = parse_design(design_path)
        pins_db = parse_pins(PINS)
        nets_dict = extract_nets(netlist_graph)
        port_to_nets = get_port_to_net_mapping(design_path)
        
        # Calculate HPWL
        hpwl = calculate_total_hpwl(placement, nets_dict, fabric_db, pins_db, port_to_nets)
        return hpwl
    except Exception as e:
        print(f"WARNING: Could not calculate HPWL from placement: {e}")
        return None


def run_placer(design_path: str, config: Dict, log_file: Optional[str] = None) -> Tuple[bool, float, str, Optional[float]]:
    """
    Run placer with given configuration.
    
    Args:
        design_path: Path to design file
        config: Configuration dict
        log_file: Optional path to log file
    
    Returns:
        (success, runtime_seconds, output_dir, hpwl)
    """
    output_dir = get_output_dir(design_path, config)
    
    # Build command
    cmd = [
        sys.executable, 'placer.py',
        '--design', design_path,
        '--fabric-cells', FABRIC_CELLS,
        '--pins', PINS,
        '--output', output_dir
    ]
    
    # Add SA parameters if not greedy-only
    if not config.get('no_sa', False):
        cmd.extend(['--sa-alpha', str(config.get('alpha', 0.95))])
        cmd.extend(['--sa-moves', str(config.get('moves', 100))])
        cmd.extend(['--sa-T-final', str(config.get('t_final', 0.1))])
    else:
        cmd.append('--no-sa')
    
    # If greedy placement is pre-computed, use it to skip greedy
    if not config.get('no_sa', False) and config.get('_greedy_placement'):
        greedy_placement_path = config.get('_greedy_placement')
        if os.path.exists(greedy_placement_path):
            cmd.extend(['--initial-placement', greedy_placement_path])
    
    # Add seed for reproducibility (use config hash as seed)
    # This ensures same config always gets same seed, but different configs get different seeds
    import hashlib
    # Exclude internal fields from seed calculation
    config_for_seed = {k: v for k, v in config.items() if not k.startswith('_')}
    config_str = json.dumps(config_for_seed, sort_keys=True)
    seed = int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16) % (2**31)
    cmd.extend(['--seed', str(seed)])
    
    # Run command
    start_time = time.time()
    output_text = ""
    hpwl = None
    
    # Log start of run
    design_name = extract_design_name(design_path)
    config_str = config_to_str(config)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {design_name} {config_str}...", flush=True)
    
    try:
        # Always capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=25200  # 7 hours timeout per run
        )
        output_text = result.stdout
        
        # Always save to log file if provided
        if log_file:
            try:
                # Create directory if it doesn't exist
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                # Write output to log file with header
                with open(log_file, 'w') as f:
                    f.write(f"# Log file for: {design_path}\n")
                    f.write(f"# Configuration: {config}\n")
                    f.write(f"# Start time: {datetime.now().isoformat()}\n")
                    f.write(f"# Command: {' '.join(cmd)}\n")
                    f.write("="*80 + "\n\n")
                    f.write(output_text)
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"# End time: {datetime.now().isoformat()}\n")
                    f.write(f"# Return code: {result.returncode}\n")
            except Exception as log_error:
                print(f"WARNING: Could not write to log file {log_file}: {log_error}")
        
        runtime = time.time() - start_time
        success = (result.returncode == 0)
        
        # Log completion
        runtime_str = f"{runtime:.1f}s" if runtime < 60 else f"{runtime/60:.1f}m" if runtime < 3600 else f"{runtime/3600:.1f}h"
        status_str = "COMPLETED" if success else "FAILED"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status_str}: {design_name} {config_str} ({runtime_str})", flush=True)
        
        # Extract HPWL from output
        if success:
            hpwl = extract_hpwl_from_output(output_text)
            
            # If not found in output, try calculating from placement file
            if hpwl is None:
                design_name = extract_design_name(design_path)
                placement_json = f'{output_dir}/{design_name}_placement.json'
                if os.path.exists(placement_json):
                    hpwl = calculate_hpwl_from_placement(placement_json, design_path)
        else:
            # Placer failed - print error details
            print(f"\n  ERROR: Placer returned exit code {result.returncode}")
            if output_text:
                # Print last 10 lines of output for debugging
                lines = output_text.strip().split('\n')
                if len(lines) > 10:
                    print(f"  Last 10 lines of output:")
                    for line in lines[-10:]:
                        print(f"    {line}")
                else:
                    print(f"  Output:")
                    for line in lines:
                        print(f"    {line}")
        
        return success, runtime, output_dir, hpwl
        
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        error_msg = f"TIMEOUT: {design_path} exceeded 7 hour limit"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] TIMEOUT: {design_name} {config_str} (after {runtime/3600:.1f}h)", flush=True)
        
        # Save timeout error to log file
        if log_file:
            try:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                with open(log_file, 'w') as f:
                    f.write(f"# Log file for: {design_path}\n")
                    f.write(f"# Configuration: {config}\n")
                    f.write(f"# Start time: {datetime.now().isoformat()}\n")
                    f.write(f"# Command: {' '.join(cmd)}\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"ERROR: {error_msg}\n")
                    f.write(f"Runtime: {runtime:.1f} seconds\n")
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"# End time: {datetime.now().isoformat()}\n")
            except Exception:
                pass  # Ignore log write errors for timeout
        
        return False, runtime, output_dir, None
    except FileNotFoundError as e:
        runtime = time.time() - start_time
        print(f"FILE NOT FOUND: {e}")
        print(f"  Design: {design_path}")
        print(f"  Check that the design file exists and is readable")
        return False, runtime, output_dir, None
    except PermissionError as e:
        runtime = time.time() - start_time
        print(f"PERMISSION ERROR: {e}")
        print(f"  Check file permissions for: {design_path}")
        return False, runtime, output_dir, None
    except Exception as e:
        runtime = time.time() - start_time
        error_msg = f"ERROR running {design_path} with config {config}: {type(e).__name__}: {e}"
        print(error_msg)
        if output_text:
            # Print last few lines of output for debugging
            lines = output_text.strip().split('\n')
            if len(lines) > 5:
                print(f"  Last 5 lines of output:")
                for line in lines[-5:]:
                    print(f"    {line}")
        
        # Save error to log file
        if log_file:
            try:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                with open(log_file, 'w') as f:
                    f.write(f"# Log file for: {design_path}\n")
                    f.write(f"# Configuration: {config}\n")
                    f.write(f"# Start time: {datetime.now().isoformat()}\n")
                    f.write(f"# Command: {' '.join(cmd)}\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"ERROR: {error_msg}\n")
                    f.write(f"Runtime: {runtime:.1f} seconds\n")
                    if output_text:
                        f.write("\nOutput:\n")
                        f.write(output_text)
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"# End time: {datetime.now().isoformat()}\n")
            except Exception:
                pass  # Ignore log write errors
        
        return False, runtime, output_dir, None


def generate_sa_knob_configs() -> List[Dict]:
    """
    Generate SA knob exploration configurations (20 configs for Task 1.D).
    
    Based on requirements:
    - Use ONE design (6502)
    - 20 different knob settings
    - Vary alpha from 0.80 to 0.99 AND vary Moves per Temp (N)
    - Also vary T_final for more diversity
    
    Returns:
        List of SA configuration dicts (no greedy config)
    """
    configs = []
    
    # Start with recommended combinations from sa_parameter_recommendations.md
    recommended = [
        {'alpha': 0.95, 'moves': 100, 't_final': 0.1},  # Default
        {'alpha': 0.98, 'moves': 300, 't_final': 0.01},  # Recommended
        {'alpha': 0.97, 'moves': 200, 't_final': 0.01},  # Option 2
        {'alpha': 0.99, 'moves': 500, 't_final': 0.001},  # Aggressive
        {'alpha': 0.90, 'moves': 200, 't_final': 0.1},  # Faster cooling
        {'alpha': 0.85, 'moves': 100, 't_final': 0.1},  # Fast cooling
        {'alpha': 0.80, 'moves': 50, 't_final': 0.1},   # Very fast cooling
        {'alpha': 0.92, 'moves': 150, 't_final': 0.05}, # Moderate
        {'alpha': 0.96, 'moves': 250, 't_final': 0.01}, # Slow cooling
        {'alpha': 0.88, 'moves': 400, 't_final': 0.1},  # Many moves
    ]
    
    # Track seen configurations
    seen = set()
    configs = []
    
    # Add recommended configs first
    for config in recommended:
        key = (config.get('alpha'), config.get('moves'), config.get('t_final'))
        if key not in seen:
            seen.add(key)
            configs.append(config)
    
    # Generate diverse combinations to reach 20 configs
    # Strategy: Sample evenly across parameter space to ensure diversity
    alphas = [0.80, 0.85, 0.90, 0.95, 0.99]
    moves_list = [50, 100, 200, 500]
    t_finals = [0.1, 0.01, 0.001]
    
    # Create a diverse set by interleaving alpha values
    # Instead of exhausting one alpha before moving to next, cycle through them
    alpha_idx = 0
    moves_idx = 0
    t_final_idx = 0
    
    while len(configs) < 20:
        alpha = alphas[alpha_idx % len(alphas)]
        moves = moves_list[moves_idx % len(moves_list)]
        t_final = t_finals[t_final_idx % len(t_finals)]
        
        key = (alpha, moves, t_final)
        if key not in seen:
            seen.add(key)
            configs.append({'alpha': alpha, 'moves': moves, 't_final': t_final})
        
        # Cycle through parameters to ensure diversity
        # Move to next combination
        alpha_idx += 1
        if alpha_idx % len(alphas) == 0:
            moves_idx += 1
        if moves_idx % len(moves_list) == 0 and alpha_idx % len(alphas) == 0:
            t_final_idx += 1
    
    # If we still need more, add intermediate values
    if len(configs) < 20:
        intermediate_alphas = [0.82, 0.87, 0.93, 0.96]
        intermediate_moves = [75, 150, 300, 400]
        intermediate_t_finals = [0.05, 0.02]
        
        for alpha in intermediate_alphas:
            if len(configs) >= 20:
                break
            for moves in intermediate_moves:
                if len(configs) >= 20:
                    break
                for t_final in intermediate_t_finals:
                    if len(configs) >= 20:
                        break
                    key = (alpha, moves, t_final)
                    if key not in seen:
                        seen.add(key)
                        configs.append({'alpha': alpha, 'moves': moves, 't_final': t_final})
    
    # Ensure we have exactly 20 by adding more if needed
    # Use a more systematic approach: sample evenly across parameter space
    if len(configs) < 20:
        # Create a grid of combinations
        all_combinations = []
        for alpha in [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95, 0.96, 0.98, 0.99]:
            for moves in [50, 75, 100, 150, 200, 300, 400, 500]:
                for t_final in [0.001, 0.01, 0.02, 0.05, 0.1]:
                    key = (alpha, moves, t_final)
                    if key not in seen:
                        all_combinations.append({'alpha': alpha, 'moves': moves, 't_final': t_final})
        
        # Add remaining configs needed
        for config in all_combinations:
            if len(configs) >= 20:
                break
            key = (config['alpha'], config['moves'], config['t_final'])
            if key not in seen:
                seen.add(key)
                configs.append(config)
    
    return configs[:20]  # Return exactly 20 configs


def generate_heavy_mode_configs() -> List[Dict]:
    """
    Generate heavy mode configurations (20 configs).
    
    Heavy mode requirements:
    - Start from alpha=0.9, moves=300
    - Cover area before successful config (alpha=0.99, moves=500, T_final=0.001)
    - Cover area after successful config (higher moves, alpha=0.99)
    - Don't start small (use larger parameter values)
    
    Returns:
        List of SA configuration dicts (20 configs)
    """
    # Heavy mode: Explore around the successful config (alpha=0.99, moves=500, T_final=0.001)
    # Strategy: Systematically cover the parameter space
    # - Before: lower alpha (0.9-0.98), lower moves (300-450)
    # - At: the successful config (0.99, 500, 0.001) and nearby
    # - After: alpha=0.99, higher moves (600-1000+)
    
    configs = [
        # AREA BEFORE: Lower alpha, lower moves (starting from 0.9, 300)
        # Starting point and progression towards successful config
        {'alpha': 0.9, 'moves': 300, 't_final': 0.1},      # Base: start from 0.9, 300
        {'alpha': 0.9, 'moves': 300, 't_final': 0.01},     # Base with different T_final
        {'alpha': 0.9, 'moves': 300, 't_final': 0.001},   # Base with T_final=0.001 (like successful)
        {'alpha': 0.92, 'moves': 350, 't_final': 0.01},   # Progress: higher alpha, higher moves
        {'alpha': 0.94, 'moves': 400, 't_final': 0.001},  # Progress: getting closer
        {'alpha': 0.96, 'moves': 450, 't_final': 0.01},   # Progress: closer to successful
        {'alpha': 0.98, 'moves': 500, 't_final': 0.001},  # Almost at successful (lower alpha)
        
        # AT SUCCESSFUL CONFIG AREA: Around alpha=0.99, moves=500
        {'alpha': 0.99, 'moves': 500, 't_final': 0.001},  # THE SUCCESSFUL CONFIG
        {'alpha': 0.99, 'moves': 500, 't_final': 0.01},   # Same alpha/moves, different T_final
        {'alpha': 0.99, 'moves': 500, 't_final': 0.1},    # Same alpha/moves, different T_final
        {'alpha': 0.99, 'moves': 450, 't_final': 0.001},  # Slightly lower moves
        {'alpha': 0.99, 'moves': 400, 't_final': 0.001},  # Lower moves, same alpha
        
        # AREA AFTER: Higher moves with alpha=0.99 (exploring beyond successful)
        {'alpha': 0.99, 'moves': 600, 't_final': 0.001},  # Higher moves, same config
        {'alpha': 0.99, 'moves': 600, 't_final': 0.01},   # Higher moves, different T_final
        {'alpha': 0.99, 'moves': 700, 't_final': 0.001},  # Even higher moves
        {'alpha': 0.99, 'moves': 800, 't_final': 0.001},  # Even higher moves
        {'alpha': 0.99, 'moves': 1000, 't_final': 0.001},  # Much higher moves
        
        # Additional exploration: High alpha with intermediate moves
        {'alpha': 0.98, 'moves': 600, 't_final': 0.001},  # High moves, slightly lower alpha
        {'alpha': 0.97, 'moves': 700, 't_final': 0.001},  # High moves, lower alpha
        {'alpha': 0.96, 'moves': 800, 't_final': 0.001},  # High moves, even lower alpha
    ]
    
    return configs[:20]  # Return exactly 20 configs


def generate_all_configs(max_sa_configs: Optional[int] = None) -> List[Dict]:
    """
    Generate all configuration combinations.
    
    Args:
        max_sa_configs: Maximum number of SA configurations to generate.
                        If None, generates ALL combinations (60 SA configs).
    
    Returns:
        List of configuration dicts
    """
    configs = []
    
    # 1. Greedy-only (no SA)
    configs.append({'no_sa': True})
    
    # 2. SA with parameter combinations
    # Full combination: 5 alphas × 4 moves × 3 T_finals = 60 SA configs
    
    all_sa_configs = []
    seen = set()
    
    # Generate ALL combinations from Issue #9 parameter ranges
    for alpha in SA_ALPHA_VALUES:
        for moves in SA_MOVES_VALUES:
            for t_final in SA_T_FINAL_VALUES:
                config = {'alpha': alpha, 'moves': moves, 't_final': t_final}
                key = (config.get('alpha'), config.get('moves'), config.get('t_final'))
                if key not in seen:
                    seen.add(key)
                    all_sa_configs.append(config)
    
    # If max_sa_configs is specified, limit the number
    if max_sa_configs is not None and max_sa_configs < len(all_sa_configs):
        # Prioritize recommended configs first
        recommended_configs = [
            {'alpha': 0.95, 'moves': 100, 't_final': 0.1},  # Default
            {'alpha': 0.98, 'moves': 300, 't_final': 0.01},  # Recommended
            {'alpha': 0.97, 'moves': 200, 't_final': 0.01},  # Option 2
            {'alpha': 0.99, 'moves': 500, 't_final': 0.001},  # Aggressive
            {'alpha': 0.90, 'moves': 200, 't_final': 0.1},  # Faster cooling
            {'alpha': 0.85, 'moves': 100, 't_final': 0.1},  # Fast cooling
        ]
        
        # Rebuild with recommended first
        prioritized = []
        prioritized_seen = set()
        
        # Add recommended configs first
        for config in recommended_configs:
            key = (config.get('alpha'), config.get('moves'), config.get('t_final'))
            if key not in prioritized_seen:
                prioritized_seen.add(key)
                prioritized.append(config)
        
        # Add remaining configs up to limit
        for config in all_sa_configs:
            if len(prioritized) >= max_sa_configs:
                break
            key = (config.get('alpha'), config.get('moves'), config.get('t_final'))
            if key not in prioritized_seen:
                prioritized_seen.add(key)
                prioritized.append(config)
        
        configs.extend(prioritized)
    else:
        # Generate all combinations
        configs.extend(all_sa_configs)
    
    return configs


def run_single_placer(args: Tuple[str, str, Dict, str]) -> Tuple[str, str, Dict, bool, float, str, Optional[float]]:
    """
    Wrapper function for running a single placer configuration.
    Must be at module level for multiprocessing pickling.
    
    Args:
        args: Tuple of (design_path, design_name, config, log_dir)
    
    Returns:
        Tuple of (design_path, design_name, config, success, runtime, output_dir, hpwl)
    """
    design_path, design_name, config, log_dir = args
    # Import here to avoid pickling issues
    from run_all_placer_variations import config_to_str, run_placer
    log_file = f'{log_dir}/{design_name}_{config_to_str(config)}.log'
    success, runtime, output_dir, hpwl = run_placer(design_path, config, log_file)
    return (design_path, design_name, config, success, runtime, output_dir, hpwl)


def run_all_variations(designs: Optional[List[str]] = None,
                      configs: Optional[List[Dict]] = None,
                      parallel: bool = False,
                      max_parallel: int = 4,
                      log_dir: str = 'logs',
                      design_greedy_placements: Optional[Dict[str, str]] = None) -> Dict:
    """
    Run all placer variations.
    
    Args:
        designs: List of design paths (default: all designs)
        configs: List of config dicts (default: generate_all_configs())
        parallel: Whether to run in parallel
        max_parallel: Maximum parallel jobs
        log_dir: Directory for log files
    
    Returns:
        Results dictionary
    """
    if designs is None:
        designs = DESIGNS
    
    if configs is None:
        configs = generate_all_configs()
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create consolidated log file
    consolidated_log = os.path.join(log_dir, 'all_runs.log')
    consolidated_log_fp = open(consolidated_log, 'w')
    consolidated_log_fp.write(f"# Consolidated log for all placer runs\n")
    consolidated_log_fp.write(f"# Start time: {datetime.now().isoformat()}\n")
    consolidated_log_fp.write(f"# Designs: {[extract_design_name(d) for d in designs]}\n")
    consolidated_log_fp.write(f"# Configurations: {len(configs)}\n")
    consolidated_log_fp.write("="*80 + "\n\n")
    
    # Results storage
    results = {
        'start_time': datetime.now().isoformat(),
        'designs': {},
        'summary': {
            'total_runs': 0,
            'successful': 0,
            'failed': 0,
            'total_runtime': 0.0
        }
    }
    
    # Generate all run combinations
    runs = []
    for design_path in designs:
        design_name = extract_design_name(design_path)
        for config in configs:
            # Create a copy of config to avoid modifying the original
            config_copy = config.copy()
            
            # If this config should use greedy placement and we have greedy placements
            if config.get('_use_greedy_placement') and design_greedy_placements and design_path in design_greedy_placements:
                config_copy['_greedy_placement'] = design_greedy_placements[design_path]
            
            runs.append((design_path, design_name, config_copy))
    
    results['summary']['total_runs'] = len(runs)
    
    print("="*80)
    print(f"Running {len(runs)} placer variations")
    print(f"Designs: {len(designs)}")
    print(f"Configurations: {len(configs)}")
    print("="*80)
    print()
    
    # Run all variations
    if parallel:
        # Parallel execution (requires multiprocessing)
        try:
            from multiprocessing import Pool
            
            # Prepare runs with log_dir included (needed for pickling)
            runs_with_log_dir = [(design_path, design_name, config, log_dir) 
                                for design_path, design_name, config in runs]
            
            with Pool(processes=max_parallel) as pool:
                # Log start of parallel execution
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting parallel execution of {len(runs)} runs with {max_parallel} workers...", flush=True)
                print()
                
                # Use imap to get results as they complete (for real-time progress)
                run_results_iter = pool.imap(run_single_placer, runs_with_log_dir)
                
                for i, (design_path, design_name, config, success, runtime, output_dir, hpwl) in enumerate(run_results_iter, 1):
                    if design_name not in results['designs']:
                        results['designs'][design_name] = {}
                    
                    config_str = config_to_str(config)
                    results['designs'][design_name][config_str] = {
                        'success': success,
                        'runtime': runtime,
                        'output_dir': output_dir,
                        'hpwl': hpwl
                    }
                    
                    # Write to consolidated log
                    status = "✓" if success else "✗"
                    hpwl_str = f", HPWL={hpwl:.2f} um" if hpwl else ", HPWL=N/A"
                    consolidated_log_fp.write(f"{status} [{i}/{len(runs)}] {design_name} {config_str}: {runtime:.1f}s{hpwl_str}\n")
                    consolidated_log_fp.flush()
                    
                    # Print result with HPWL immediately as it completes
                    if success:
                        results['summary']['successful'] += 1
                        print(f"[{i}/{len(runs)}] ✓ {design_name} {config_str}: {runtime:.1f}s{hpwl_str}")
                    else:
                        results['summary']['failed'] += 1
                        print(f"[{i}/{len(runs)}] ✗ {design_name} {config_str}: {runtime:.1f}s{hpwl_str}")
                    
                    results['summary']['total_runtime'] += runtime
                
        except ImportError:
            print("WARNING: multiprocessing not available, running sequentially")
            parallel = False
    
    if not parallel:
        # Sequential execution
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting sequential execution of {len(runs)} runs...", flush=True)
        print()
        
        for i, (design_path, design_name, config) in enumerate(runs, 1):
            config_str = config_to_str(config)
            print(f"[{i}/{len(runs)}] Running {design_name} with {config_str}...", end=' ', flush=True)
            
            log_file = f'{log_dir}/{design_name}_{config_str}.log'
            success, runtime, output_dir, hpwl = run_placer(design_path, config, log_file)
            
            # Write to consolidated log
            status = "✓" if success else "✗"
            hpwl_str = f", HPWL={hpwl:.2f} um" if hpwl else ", HPWL=N/A"
            consolidated_log_fp.write(f"{status} [{i}/{len(runs)}] {design_name} {config_str}: {runtime:.1f}s{hpwl_str}\n")
            consolidated_log_fp.flush()  # Ensure it's written immediately
            
            if design_name not in results['designs']:
                results['designs'][design_name] = {}
            
            results['designs'][design_name][config_str] = {
                'success': success,
                'runtime': runtime,
                'output_dir': output_dir,
                'hpwl': hpwl
            }
            
            if success:
                results['summary']['successful'] += 1
                print(f"✓ ({runtime:.1f}s{hpwl_str})")
            else:
                results['summary']['failed'] += 1
                print(f"✗ ({runtime:.1f}s{hpwl_str})")
            
            results['summary']['total_runtime'] += runtime
    
    results['end_time'] = datetime.now().isoformat()
    
    # Close consolidated log
    consolidated_log_fp.write("\n" + "="*80 + "\n")
    consolidated_log_fp.write(f"# End time: {datetime.now().isoformat()}\n")
    consolidated_log_fp.write(f"# Total runs: {results['summary']['total_runs']}\n")
    consolidated_log_fp.write(f"# Successful: {results['summary']['successful']}\n")
    consolidated_log_fp.write(f"# Failed: {results['summary']['failed']}\n")
    consolidated_log_fp.write(f"# Total runtime: {results['summary']['total_runtime']:.1f} seconds\n")
    consolidated_log_fp.close()
    print(f"\nConsolidated log saved to: {consolidated_log}")
    
    return results


def config_to_str(config: Dict) -> str:
    """Convert config dict to string identifier."""
    if config.get('no_sa', False):
        return 'greedy'
    else:
        alpha = config.get('alpha', 0.95)
        moves = config.get('moves', 100)
        t_final = config.get('t_final', 0.1)
        return f'sa_a{alpha}_m{moves}_T{t_final}'


def print_summary(results: Dict):
    """Print summary of results."""
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total runs: {results['summary']['total_runs']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Total runtime: {results['summary']['total_runtime']:.1f} seconds ({results['summary']['total_runtime']/60:.1f} minutes)")
    print()
    
    # Per-design summary
    for design_name, design_results in results['designs'].items():
        print(f"\n{design_name}:")
        for config_str, result in design_results.items():
            status = "✓" if result['success'] else "✗"
            hpwl_str = f", HPWL={result.get('hpwl', 'N/A'):.2f} um" if result.get('hpwl') else ""
            print(f"  {status} {config_str}: {result['runtime']:.1f}s{hpwl_str} -> {result['output_dir']}")


def calculate_pareto_frontier(runtimes: List[float], hpwls: List[float]) -> List[int]:
    """
    Calculate Pareto-optimal points.
    
    A point is Pareto-optimal if no other point has both lower runtime and lower HPWL.
    
    Args:
        runtimes: List of runtime values
        hpwls: List of HPWL values
    
    Returns:
        List of indices of Pareto-optimal points
    """
    if len(runtimes) != len(hpwls):
        return []
    
    pareto_indices = []
    n = len(runtimes)
    
    for i in range(n):
        is_pareto = True
        for j in range(n):
            if i != j:
                # Point j dominates point i if it has both lower runtime AND lower HPWL
                if runtimes[j] < runtimes[i] and hpwls[j] < hpwls[i]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    # Sort by runtime for plotting
    pareto_indices.sort(key=lambda i: runtimes[i])
    return pareto_indices


def plot_sa_knob_analysis(results: Dict, output_file: str = 'sa_knob_analysis.png'):
    """
    Generate scatter plot: Runtime vs HPWL with Pareto frontier.
    
    Args:
        results: Results dictionary from run_all_variations
        output_file: Output PNG file path
    """
    # Collect data points
    runtimes = []
    hpwls = []
    configs = []
    alphas = []
    
    for design_name, design_results in results['designs'].items():
        for config_str, result in design_results.items():
            if result.get('success') and result.get('hpwl') is not None:
                runtimes.append(result['runtime'])
                hpwls.append(result['hpwl'])
                configs.append(config_str)
                # Extract alpha from config string
                alpha_match = re.search(r'alpha([\d.]+)', config_str)
                if alpha_match:
                    alphas.append(float(alpha_match.group(1)))
                else:
                    alphas.append(0.95)  # Default
    
    if len(runtimes) == 0:
        print("WARNING: No valid data points for plot generation")
        return
    
    # Calculate Pareto frontier
    pareto_indices = calculate_pareto_frontier(runtimes, hpwls)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points, color-coded by alpha
    scatter = ax.scatter(runtimes, hpwls, c=alphas, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Highlight Pareto frontier
    if pareto_indices:
        pareto_runtimes = [runtimes[i] for i in pareto_indices]
        pareto_hpwls = [hpwls[i] for i in pareto_indices]
        ax.scatter(pareto_runtimes, pareto_hpwls, s=200, marker='*', 
                  c='red', edgecolors='black', linewidth=1.5, 
                  label='Pareto Frontier', zorder=5)
        
        # Draw line connecting Pareto points
        if len(pareto_indices) > 1:
            ax.plot(pareto_runtimes, pareto_hpwls, 'r--', alpha=0.5, linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cooling Rate (alpha)', rotation=270, labelpad=20)
    
    # Labels and title
    ax.set_xlabel('Runtime (seconds)', fontsize=12)
    ax.set_ylabel('Final HPWL (microns)', fontsize=12)
    ax.set_title('SA Knob Analysis: Runtime vs Final HPWL\n(Pareto Frontier Highlighted)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add text annotation for Pareto frontier
    if pareto_indices:
        ax.text(0.02, 0.98, 
               f'Pareto-optimal points: {len(pareto_indices)}/{len(runtimes)}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nScatter plot saved to {output_file}")
    plt.close()


def save_results(results: Dict, output_file: str = 'placer_variations_results.json'):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run all placer variations ensuring no collisions'
    )
    parser.add_argument('--designs', nargs='+',
                       help='Design files to run (default: 6502 only, use --all-designs for all)')
    parser.add_argument('--all-designs', action='store_true',
                       help='Run on all designs (6502, aes_128, arith, z80) instead of just 6502')
    parser.add_argument('--greedy-only', action='store_true',
                       help='Run only greedy (no SA) variations')
    parser.add_argument('--sa-only', action='store_true',
                       help='Run only SA variations (skip greedy)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run in parallel (requires multiprocessing)')
    parser.add_argument('--max-parallel', type=int, default=4,
                       help='Maximum parallel jobs (default: 4)')
    parser.add_argument('--log-dir', default='logs',
                       help='Directory for log files (default: logs)')
    parser.add_argument('--results-file', default='placer_variations_results.json',
                       help='Output file for results JSON (default: placer_variations_results.json)')
    parser.add_argument('--configs', type=int, default=None,
                       help='Maximum number of SA configs to run (default: None = all 60 combinations)')
    parser.add_argument('--sa-knob-exploration', action='store_true',
                       help='Run SA knob exploration (Task 1.D): ONE design (6502) with 20 configs, generate plot')
    parser.add_argument('--plot-output', default='sa_knob_analysis.png',
                       help='Output file for scatter plot (default: sa_knob_analysis.png)')
    parser.add_argument('--heavy-mode', action='store_true',
                       help='Run heavy mode: 20 configs starting from alpha=0.9, moves=300, don\'t start small')
    
    args = parser.parse_args()
    
    # Initialize design_greedy_placements (used in heavy mode)
    design_greedy_placements = None
    
    # Handle SA knob exploration mode (Task 1.D)
    if args.sa_knob_exploration:
        # Task 1.D: Use ONE design (6502) with 20 SA configs
        designs = ['designs/6502_mapped.json']
        configs = generate_sa_knob_configs()  # 20 SA configs, no greedy
        
        # First, run greedy once and save it
        print("="*80)
        print("SA Knob Exploration Mode (Task 1.D)")
        print(f"Design: 6502")
        print(f"Step 1: Running greedy placement once (will be reused for all SA runs)...")
        print("="*80)
        print()
        
        design_path = designs[0]
        design_name = extract_design_name(design_path)
        greedy_output_dir = f'build/{design_name}/greedy'
        greedy_placement_json = f'{greedy_output_dir}/{design_name}_placement.json'
        
        # Run greedy once
        greedy_config = {'no_sa': True}
        greedy_log = f'{args.log_dir}/{design_name}_greedy_initial.log'
        success, runtime, _, _ = run_placer(design_path, greedy_config, greedy_log)
        
        if not success or not os.path.exists(greedy_placement_json):
            print(f"ERROR: Failed to generate greedy placement. Cannot proceed with SA knob exploration.")
            sys.exit(1)
        
        print(f"\n✓ Greedy placement completed and saved to: {greedy_placement_json}")
        print(f"  This will be reused as initial state for all {len(configs)} SA configurations")
        print("="*80)
        print(f"\nStep 2: Running {len(configs)} SA configurations...")
        print("="*80)
        print()
        
        # Store greedy placement path for SA runs
        for config in configs:
            config['_greedy_placement'] = greedy_placement_json
    elif args.heavy_mode:
        # Heavy mode: 20 configs starting from alpha=0.9, moves=300, don't start small
        # Determine designs
        if args.designs:
            designs = args.designs
        elif args.all_designs:
            designs = DESIGNS  # All designs
        else:
            # Default: 6502 only
            designs = ['designs/6502_mapped.json']
        
        configs = generate_heavy_mode_configs()  # 20 heavy mode SA configs
        
        # First, run greedy once for each design and save it
        print("="*80)
        print("Heavy Mode")
        print(f"Designs: {[extract_design_name(d) for d in designs]}")
        print(f"Step 1: Running greedy placement once for each design (will be reused for all SA runs)...")
        print("="*80)
        print()
        
        # Run greedy for each design and store placement paths
        design_greedy_placements = {}
        for design_path in designs:
            design_name = extract_design_name(design_path)
            greedy_output_dir = f'build/{design_name}/greedy'
            greedy_placement_json = f'{greedy_output_dir}/{design_name}_placement.json'
            
            # Run greedy once
            greedy_config = {'no_sa': True}
            greedy_log = f'{args.log_dir}/{design_name}_greedy_initial.log'
            success, runtime, _, _ = run_placer(design_path, greedy_config, greedy_log)
            
            if not success or not os.path.exists(greedy_placement_json):
                print(f"ERROR: Failed to generate greedy placement for {design_name}. Cannot proceed with heavy mode.")
                sys.exit(1)
            
            design_greedy_placements[design_path] = greedy_placement_json
            print(f"✓ Greedy placement for {design_name} completed and saved to: {greedy_placement_json}")
        
        print(f"\n  All greedy placements completed. They will be reused as initial state for all {len(configs)} SA configurations per design")
        print("="*80)
        print(f"\nStep 2: Running {len(configs)} SA configurations for each design...")
        print("="*80)
        print()
        
        # Store greedy placement paths - will be set per design when creating runs
        # We'll pass this to run_all_variations via a custom parameter
        # For now, mark configs to use greedy placement
        for config in configs:
            config['_use_greedy_placement'] = True
    else:
        # Regular mode: Determine designs
        if args.designs:
            designs = args.designs
        elif args.all_designs:
            designs = DESIGNS  # All designs
        else:
            # Default: 6502 only
            designs = ['designs/6502_mapped.json']
        
        # Generate configs (includes greedy + SA)
        # If configs is None, generate ALL combinations (60 SA configs)
        all_configs = generate_all_configs(max_sa_configs=args.configs)
        
        if args.greedy_only:
            # Only greedy placement for all designs
            configs = [c for c in all_configs if c.get('no_sa', False)]
            print("="*80)
            print("Greedy-Only Mode")
            print(f"Designs: {[extract_design_name(d) for d in designs]}")
            print(f"Configurations: {len(configs)} (greedy only)")
            print("="*80)
            print()
        elif args.sa_only:
            # Only SA placement for all designs (skip greedy)
            configs = [c for c in all_configs if not c.get('no_sa', False)]
            print("="*80)
            print("SA-Only Mode")
            print(f"Designs: {[extract_design_name(d) for d in designs]}")
            print(f"Configurations: {len(configs)} (SA only, no greedy)")
            print("="*80)
            print()
        else:
            # Default: Use all configs (greedy + SA) for all designs
            configs = all_configs
            greedy_count = sum(1 for c in configs if c.get('no_sa', False))
            sa_count = len(configs) - greedy_count
            print("="*80)
            print("Regular Mode (Greedy + SA)")
            print(f"Designs: {[extract_design_name(d) for d in designs]}")
            print(f"Configurations: {len(configs)} total ({greedy_count} greedy + {sa_count} SA)")
            if args.configs is None:
                print("  Generating ALL SA parameter combinations (60 SA configs)")
            print("="*80)
            print("Note: Greedy will run on ALL designs, then SA on ALL designs")
            print("="*80)
            print()
    
    # Run all variations
    results = run_all_variations(
        designs=designs,
        configs=configs,
        parallel=args.parallel,
        max_parallel=args.max_parallel,
        log_dir=args.log_dir,
        design_greedy_placements=design_greedy_placements
    )
    
    # Print summary
    print_summary(results)
    
    # Generate scatter plot for SA knob exploration
    if args.sa_knob_exploration:
        plot_sa_knob_analysis(results, args.plot_output)
        print("\n" + "="*80)
        print("SA Knob Exploration Complete")
        print("="*80)
        print(f"Plot saved to: {args.plot_output}")
        print("\nNext steps:")
        print("1. Review the Pareto frontier in the plot")
        print("2. Identify best HPWL and fastest runtime configurations")
        print("3. Add analysis to README.md")
        print("="*80)
    
    # Save results
    save_results(results, args.results_file)
    
    # Exit with error if any failed
    if results['summary']['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
