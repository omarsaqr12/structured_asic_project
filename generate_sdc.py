#!/usr/bin/env python3
"""
Generate SDC (Synopsys Design Constraints) files for each design.
Creates [design_name].sdc with clock, input delay, and output delay constraints.
"""

import argparse
import os
import json
from typing import Dict, List, Tuple, Optional


# Design frequency specifications (MHz -> period in ns)
# Based on design complexity, netlist size, and path characteristics
# Recommended starting points for SDC generation and first STA runs
DESIGN_FREQUENCIES = {
    '6502': {'freq_mhz': 25, 'period_ns': 40.0},      # 1.7 MB - Long decode/control paths; 50 MHz is unlikely to pass
    'aes_128': {'freq_mhz': 50, 'period_ns': 20.0},  # 50 MB - Big netlist; 50 MHz is a balanced starting point for optimization
    'arith': {'freq_mhz': 50, 'period_ns': 20.0},     # 303 KB - Small datapath; 50 MHz safe (can go to 100 MHz / 10 ns)
    'z80': {'freq_mhz': 25, 'period_ns': 40.0},       # 5.6 MB - Control-heavy, similar to 6502 — safe default
}

# Clock port name variations to check
CLOCK_PORT_NAMES = ['clk', 'clock', 'CLK', 'Clock', 'CLOCK']


def get_io_ports(design_json_path: str) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Extract input and output ports from design JSON file.
    Also identifies the clock port.
    
    Returns:
        (input_ports, output_ports, clock_port_name)
    """
    if not os.path.exists(design_json_path):
        raise FileNotFoundError(f"Design JSON not found: {design_json_path}")
    
    with open(design_json_path, 'r') as f:
        design_data = json.load(f)
    
    # Find the top module
    modules = design_data.get('modules', {})
    top_module = None
    
    for module_name, module_data in modules.items():
        if module_data.get('attributes', {}).get('top') == '00000000000000000000000000000001':
            top_module = module_data
            break
    
    if top_module is None and modules:
        # Fallback: use first module
        top_module = list(modules.values())[0]
    
    if top_module is None:
        return [], [], None
    
    ports = top_module.get('ports', {})
    input_ports = []
    output_ports = []
    clock_port = None
    
    for port_name, port_data in ports.items():
        direction = port_data.get('direction', '')
        
        # Check if this is a clock port
        if port_name in CLOCK_PORT_NAMES or port_name.lower() in [c.lower() for c in CLOCK_PORT_NAMES]:
            clock_port = port_name
        
        if direction == 'input':
            input_ports.append(port_name)
        elif direction == 'output':
            output_ports.append(port_name)
    
    return input_ports, output_ports, clock_port


def generate_sdc_content(
    design_name: str,
    input_ports: List[str],
    output_ports: List[str],
    clock_port: Optional[str],
    period_ns: float
) -> str:
    """
    Generate SDC file content following the mandatory format:
    1. create_clock
    2. set_input_delay (using remove_from_collection)
    3. set_output_delay (using all_outputs)
    
    Args:
        design_name: Name of the design
        input_ports: List of input port names
        output_ports: List of output port names
        clock_port: Name of clock port (or None)
        period_ns: Clock period in nanoseconds
    
    Returns:
        SDC file content as string
    """
    lines = []
    
    # Header comment
    lines.append(f"# SDC Constraints for {design_name}")
    lines.append(f"# Clock period: {period_ns} ns")
    lines.append("")
    
    # 1. Clock definition (MANDATORY)
    if clock_port:
        lines.append("# 1. Clock definition")
        lines.append(f"create_clock -name clk -period {period_ns:.1f} [get_ports {clock_port}]")
        lines.append("")
    else:
        raise ValueError(f"No clock port found for design {design_name}. Cannot generate valid SDC file.")
    
    # 2. Input delay (MANDATORY) - exclude clock port using remove_from_collection
    # Delay value: 2.0 ns (as specified)
    delay_ns = 2.0
    
    if input_ports:
        lines.append("# 2. Input delay (exclude the clock port)")
        lines.append(f"set_input_delay {delay_ns:.1f} -clock clk \\")
        lines.append(f"    [remove_from_collection [all_inputs] [get_ports {clock_port}]]")
        lines.append("")
    
    # 3. Output delay (MANDATORY) - use all_outputs
    if output_ports:
        lines.append("# 3. Output delay")
        lines.append(f"set_output_delay {delay_ns:.1f} -clock clk [all_outputs]")
        lines.append("")
    
    return '\n'.join(lines)


def generate_sdc_file(design_name: str, design_json_path: str, output_path: Optional[str] = None) -> str:
    """
    Generate SDC file for a design.
    
    Args:
        design_name: Name of the design (e.g., '6502', 'aes_128')
        design_json_path: Path to the mapped JSON file
        output_path: Optional output path (default: [design_name].sdc in current directory)
    
    Returns:
        Path to generated SDC file
    """
    # Get design frequency specification
    if design_name not in DESIGN_FREQUENCIES:
        raise ValueError(f"Unknown design: {design_name}. Known designs: {list(DESIGN_FREQUENCIES.keys())}")
    
    freq_spec = DESIGN_FREQUENCIES[design_name]
    period_ns = freq_spec['period_ns']
    
    # Extract ports
    print(f"Extracting ports from {design_json_path}...")
    input_ports, output_ports, clock_port = get_io_ports(design_json_path)
    
    print(f"  Found {len(input_ports)} input ports")
    print(f"  Found {len(output_ports)} output ports")
    if clock_port:
        print(f"  Clock port: {clock_port}")
    else:
        print(f"  WARNING: No clock port detected!")
    
    # Filter out clock port from input ports for delay constraints
    input_ports_for_delays = [p for p in input_ports if p != clock_port]
    
    # Generate SDC content
    sdc_content = generate_sdc_content(
        design_name,
        input_ports,
        output_ports,
        clock_port,
        period_ns
    )
    
    # Determine output path
    if output_path is None:
        output_path = f"{design_name}.sdc"
    
    # Write SDC file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sdc_content)
    
    print(f"✓ SDC file generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate SDC constraint files for designs'
    )
    parser.add_argument(
        '--design',
        type=str,
        choices=['6502', 'aes_128', 'arith', 'z80', 'all'],
        default='all',
        help='Design name to generate SDC for (default: all)'
    )
    parser.add_argument(
        '--designs-dir',
        type=str,
        default='designs',
        help='Directory containing mapped JSON files (default: designs)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for SDC files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Determine which designs to process
    if args.design == 'all':
        designs = ['6502', 'aes_128', 'arith', 'z80']
    else:
        designs = [args.design]
    
    # Generate SDC files
    print("=" * 60)
    print("SDC File Generation")
    print("=" * 60)
    print()
    
    for design_name in designs:
        design_json_path = os.path.join(args.designs_dir, f"{design_name}_mapped.json")
        
        if not os.path.exists(design_json_path):
            print(f"ERROR: Design file not found: {design_json_path}")
            continue
        
        output_path = os.path.join(args.output_dir, f"{design_name}.sdc")
        
        try:
            generate_sdc_file(design_name, design_json_path, output_path)
            print()
        except Exception as e:
            print(f"ERROR: Failed to generate SDC for {design_name}: {e}")
            print()
    
    print("=" * 60)
    print("SDC generation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
