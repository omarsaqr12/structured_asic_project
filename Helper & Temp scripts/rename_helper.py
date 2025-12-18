#!/usr/bin/env python3
"""
Rename Helper: Replace logical instance names with physical slot names.

Reads [design]_final.v and placement.map, then generates [design]_renamed.v
with physical slot names instead of logical names.
"""

import re
import argparse
from pathlib import Path
from typing import Dict


def parse_placement_map(map_path: str) -> Dict[str, str]:
    """
    Parse placement map file.
    
    Args:
        map_path: Path to placement.map file
        
    Returns:
        Dictionary mapping logical_name -> slot_name
    """
    placement_map = {}
    
    with open(map_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                logical_name = parts[0]
                slot_name = parts[1]
                placement_map[logical_name] = slot_name
    
    return placement_map


def rename_instances(verilog_path: str, map_path: str, output_path: str):
    """
    Rename instances in Verilog file from logical names to physical slot names.
    
    Args:
        verilog_path: Path to input Verilog file ([design]_final.v)
        map_path: Path to placement.map file
        output_path: Path to output Verilog file ([design]_renamed.v)
    """
    # Parse placement map
    placement_map = parse_placement_map(map_path)
    
    if not placement_map:
        raise ValueError(f"No mappings found in placement map: {map_path}")
    
    # Read Verilog file
    with open(verilog_path, 'r') as f:
        verilog_content = f.read()
    
    # Create reverse mapping for special cases (CTS buffers, unused cells)
    # These should keep their names (but still need sanitization if they contain $)
    special_prefixes = ['cts_buffer_', 'eco_unused_']
    
    def sanitize_identifier(name: str) -> str:
        """
        Sanitize Verilog identifier by replacing invalid characters.
        Verilog identifiers can contain letters, digits, underscores, and $.
        Colons (:), dots (.), and backslashes (\) are not allowed in identifiers.
        """
        sanitized = name
        # Replace invalid characters with underscore
        if '\\' in sanitized or ':' in sanitized or '.' in sanitized or '$' in sanitized:
            sanitized = sanitized.replace('\\', '_')  # Backslash
            sanitized = sanitized.replace(':', '_')   # Colon
            sanitized = sanitized.replace('.', '_')   # Dot
            sanitized = sanitized.replace('$', '_')   # Dollar sign
        return sanitized
    
    # Replace instance names
    # Pattern: cell_type instance_name (
    # We need to be careful to only replace instance names, not other identifiers
    
    lines = verilog_content.split('\n')
    output_lines = []
    
    for line in lines:
        # Check if this line contains a cell instance declaration
        # Pattern: sky130_fd_sc_hd__<type> <instance_name> (
        # Updated to match instance names with $, :, and other characters
        instance_match = re.search(r'(sky130_fd_sc_hd__\w+)\s+([^\s(]+)\s*\(', line)
        
        if instance_match:
            cell_type = instance_match.group(1)
            instance_name = instance_match.group(2)
            
            # Sanitize instance name if it contains invalid characters (\, $, :, .)
            original_name = instance_name
            if '\\' in instance_name or ':' in instance_name or '.' in instance_name or '$' in instance_name:
                instance_name = sanitize_identifier(instance_name)
                # Replace the invalid name in the line
                line = re.sub(
                    rf'\b{re.escape(original_name)}\b',
                    instance_name,
                    line
                )
            
            # Check if this is a special case (CTS buffers, unused cells, etc.)
            should_rename = True
            for prefix in special_prefixes:
                if instance_name.startswith(prefix):
                    should_rename = False
                    break
            
            # Rename if in placement map
            if should_rename and instance_name in placement_map:
                new_name = placement_map[instance_name]
                # Replace instance name in this line
                line = re.sub(
                    rf'\b{re.escape(instance_name)}\b',
                    new_name,
                    line
                )
        
        output_lines.append(line)
    
    # Write output file
    output_content = '\n'.join(output_lines)
    
    # Add header comment
    header = f"// Renamed Verilog netlist\n"
    header += f"// Logical instance names replaced with physical slot names from placement.map\n"
    header += f"// Generated from: {Path(verilog_path).name}\n\n"
    
    with open(output_path, 'w') as f:
        f.write(header)
        f.write(output_content)
    
    print(f"Renamed Verilog saved to: {output_path}")
    print(f"  Mapped {len(placement_map)} logical names to physical slots")


def main():
    parser = argparse.ArgumentParser(
        description='Rename Verilog instances from logical names to physical slot names'
    )
    parser.add_argument('--design', 
                        default='6502',
                        help='Design name (default: 6502)')
    parser.add_argument('--verilog',
                        help='Path to input Verilog file ([design]_final.v). '
                             'If not specified, uses test_eco_[design].v')
    parser.add_argument('--map',
                        help='Path to placement.map file. '
                             'If not specified, uses build/[design]/[design].map')
    parser.add_argument('--output',
                        help='Path to output Verilog file ([design]_renamed.v). '
                             'If not specified, uses build/[design]/[design]_renamed.v')
    
    args = parser.parse_args()
    
    design = args.design
    
    # Determine input Verilog path
    if args.verilog:
        verilog_path = args.verilog
    else:
        verilog_path = f"test_eco_{design}.v"
    
    # Determine placement map path
    if args.map:
        map_path = args.map
    else:
        map_path = f"build/{design}/{design}.map"
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"build/{design}/{design}_renamed.v"
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Validate files exist
    if not Path(verilog_path).exists():
        print(f"ERROR: Verilog file not found: {verilog_path}")
        return 1
    
    if not Path(map_path).exists():
        print(f"ERROR: Placement map not found: {map_path}")
        return 1
    
    try:
        rename_instances(verilog_path, map_path, output_path)
        print(f"\n✓ Renaming complete: {output_path}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == '__main__':
    exit(main())

