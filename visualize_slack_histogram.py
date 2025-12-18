#!/usr/bin/env python3
"""
Slack Histogram Visualization Script
Parse _setup.rpt and plot a 1D histogram of all endpoint slacks.
"""

import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def parse_setup_report_for_slacks(setup_rpt_path: str) -> List[float]:
    """
    Parse setup timing report to extract all endpoint slacks.
    
    Returns:
        List of slack values (in ns)
    """
    all_slacks = []
    current_slack = None
    
    with open(setup_rpt_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for "No paths found"
        if line == "No paths found.":
            print("  ⚠ WARNING: Report contains 'No paths found'")
            break
        
        # Check for slack value
        # Format: "58.36   slack (MET)" or "-0.15   slack (VIOLATED)"
        if 'slack' in line.lower():
            slack_match = re.search(r'([+-]?\d+\.?\d*)\s+slack', line)
            if slack_match:
                slack_value = float(slack_match.group(1))
                all_slacks.append(slack_value)
                current_slack = slack_value
        
        i += 1
    
    return all_slacks


def plot_slack_histogram(slacks: List[float], output_path: str, design_name: str):
    """Plot 1D histogram of endpoint slacks."""
    if not slacks:
        print("  ✗ ERROR: No slack values found. Cannot plot histogram.")
        return
    
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(slacks, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Color bars based on slack value (red for negative, yellow for low positive, green for good)
    for i, (bar, slack_val) in enumerate(zip(patches, bins[:-1])):
        if slack_val < 0:
            bar.set_facecolor('red')
            bar.set_alpha(0.8)
        elif slack_val < 1.0:
            bar.set_facecolor('orange')
            bar.set_alpha(0.7)
        else:
            bar.set_facecolor('green')
            bar.set_alpha(0.6)
    
    plt.xlabel('Slack (ns)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Endpoints', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of Setup Timing Slack - {design_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics
    mean_slack = np.mean(slacks)
    min_slack = np.min(slacks)
    max_slack = np.max(slacks)
    median_slack = np.median(slacks)
    std_slack = np.std(slacks)
    
    # Count violations (negative slack)
    violations = sum(1 for s in slacks if s < 0)
    violation_percent = (violations / len(slacks)) * 100 if slacks else 0
    
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean_slack:.3f} ns\n'
    stats_text += f'Min: {min_slack:.3f} ns\n'
    stats_text += f'Max: {max_slack:.3f} ns\n'
    stats_text += f'Median: {median_slack:.3f} ns\n'
    stats_text += f'Std Dev: {std_slack:.3f} ns\n'
    stats_text += f'Total Paths: {len(slacks)}\n'
    if violations > 0:
        stats_text += f'⚠ Violations: {violations} ({violation_percent:.1f}%)'
    else:
        stats_text += f'✓ No violations'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, family='monospace')
    
    # Add vertical line at slack = 0
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Slack')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Histogram saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot histogram of setup timing slack values from STA report'
    )
    parser.add_argument('--design', required=True, help='Design name (e.g., arith, 6502)')
    parser.add_argument('--build-dir', default='build', help='Build directory (default: build)')
    parser.add_argument('--setup-rpt', default=None,
                       help='Path to setup report (default: build/{design}/{design}_setup.rpt)')
    parser.add_argument('--output', default=None,
                       help='Output PNG file (default: build/{design}/{design}_slack_histogram.png)')
    
    args = parser.parse_args()
    
    # Build paths
    build_dir = args.build_dir
    design_name = args.design
    
    if args.setup_rpt:
        setup_rpt_path = args.setup_rpt
    else:
        setup_rpt_path = os.path.join(build_dir, design_name, f"{design_name}_setup.rpt")
    
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(build_dir, design_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{design_name}_slack_histogram.png")
    
    print("=" * 60)
    print(f"Slack Histogram Visualization for {design_name}")
    print("=" * 60)
    print(f"Setup Report: {setup_rpt_path}")
    print(f"Output: {output_path}")
    print()
    
    # Parse setup report
    print("Parsing setup report...")
    if not os.path.exists(setup_rpt_path):
        print(f"  ✗ ERROR: Setup report not found: {setup_rpt_path}")
        return
    
    all_slacks = parse_setup_report_for_slacks(setup_rpt_path)
    
    if not all_slacks:
        print("  ✗ ERROR: No slack values found in report.")
        print("  This may indicate:")
        print("    - Report contains 'No paths found'")
        print("    - Report format is different than expected")
        print("    - Timing analysis did not complete successfully")
        return
    
    print(f"  ✓ Found {len(all_slacks)} timing paths with slack values")
    print(f"    Slack range: [{min(all_slacks):.3f}, {max(all_slacks):.3f}] ns")
    print()
    
    # Plot histogram
    print("Plotting histogram...")
    plot_slack_histogram(all_slacks, output_path, design_name)
    
    print()
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

