#!/usr/bin/env python3
"""
Instance Renaming Script for Phase 4 & 5.

Parse _final.v and .map files. Rename every Verilog instance to match its 
assigned physical slot name. Output build/[design_name]/[design_name]_renamed.v.
"""

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def parse_placement_map(map_path: str) -> Dict[str, str]:
    """
    Parse placement map file to get logical_name -> physical_slot_name mapping.
    
    Format: logical_name physical_slot_name
    
    Args:
        map_path: Path to placement.map file
        
    Returns:
        Dictionary mapping logical_name -> physical_slot_name
    """
    placement_map = {}
    
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Placement map not found: {map_path}")
    
    with open(map_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                logical_name = parts[0]
                physical_slot_name = parts[1]
                placement_map[logical_name] = physical_slot_name
    
    return placement_map


def find_instance_declarations(verilog_content: str) -> List[Tuple[int, str, str, str]]:
    """
    Find all instance declarations in Verilog content.
    
    Pattern: cell_type instance_name (
    Excludes: module declarations (module name ( )
    
    Returns:
        List of tuples: (line_number, full_line, cell_type, instance_name)
    """
    instances = []
    lines = verilog_content.split('\n')
    
    # Pattern to match: cell_type instance_name (
    # Handles various cell types: sky130_fd_sc_hd__*, conb_1, etc.
    # Excludes module declarations by checking that first word is not "module"
    instance_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(')
    
    for line_num, line in enumerate(lines, 1):
        match = instance_pattern.search(line)
        if match:
            cell_type = match.group(1)
            instance_name = match.group(2)
            # Skip module declarations
            if cell_type.lower() == 'module':
                continue
            instances.append((line_num, line, cell_type, instance_name))
    
    return instances


def is_buffer_cell(cell_type: str) -> bool:
    """
    Check if a cell type is a buffer (clkbuf, buf, etc.).
    
    Args:
        cell_type: Cell type name (e.g., 'sky130_fd_sc_hd__clkbuf_4')
        
    Returns:
        True if cell is a buffer, False otherwise
    """
    cell_lower = cell_type.lower()
    return 'buf' in cell_lower or 'clkbuf' in cell_lower


def remove_buffer_output_pins(instance_lines: List[str]) -> List[str]:
    """
    Remove output pin connections (typically .Y(...)) from buffer instances.
    Keeps only input pin connections (typically .A(...)).
    
    Args:
        instance_lines: List of lines making up the instance declaration
        
    Returns:
        Modified list of lines with output pins removed
    """
    output_lines = []
    
    for line in instance_lines:
        # Skip lines that are output pin connections
        # Common output pins: .Y(, .Q(, .Z(, etc.
        # For buffers, typically only .Y( is the output
        stripped = line.strip()
        
        # Skip output pin connections
        # Pattern: .Y(...) or .Y(...),
        if re.match(r'^\s*\.Y\s*\(', stripped):
            continue  # Skip .Y(...) line
        
        # Keep all other lines (including .A(...), closing );, etc.)
        output_lines.append(line)
    
    # If we removed .Y pin, we need to remove the trailing comma from the last pin connection
    # This handles the case: .A(n109), .Y(n504) -> .A(n109)
    if len(output_lines) >= 2:
        # Find the last pin connection line (before closing );)
        # Look for the last line that contains a pin connection (starts with .)
        for i in range(len(output_lines) - 2, -1, -1):
            line = output_lines[i].strip()
            # Check if this is a pin connection line (starts with .)
            if re.match(r'^\s*\.\w+\s*\(', line):
                # This is a pin connection, check if it ends with comma
                line_rstrip = output_lines[i].rstrip()
                if line_rstrip.endswith(','):
                    # Remove trailing comma
                    output_lines[i] = line_rstrip[:-1]
                break  # Found the last pin connection, done
    
    return output_lines


def rename_instances_in_verilog(
    verilog_content: str,
    placement_map: Dict[str, str],
    design_name: str
) -> Tuple[str, Dict[str, int]]:
    """
    Rename instances in Verilog content using placement map.
    Also removes output pin connections from buffer instances.
    
    Args:
        verilog_content: Original Verilog file content
        placement_map: Dictionary mapping logical_name -> physical_slot_name
        design_name: Design name for error reporting
        
    Returns:
        Tuple of (renamed_content, statistics_dict)
    """
    lines = verilog_content.split('\n')
    output_lines = []
    
    # Statistics
    stats = {
        'total_instances': 0,
        'renamed_instances': 0,
        'skipped_instances': 0,
        'missing_in_map': [],
        'buffer_instances_processed': 0
    }
    
    # Track if we've added the line at the end of wire definitions
    wire_section_ended = False
    
    # All instances should be renamed to their physical slot names from the placement map
    # No special handling - all instances are treated the same way
    special_prefixes = []  # No special prefixes - all instances should be renamed normally
    
    # Pattern to match instance declarations: cell_type instance_name (
    # Exclude module declarations
    # Updated pattern to handle instance names with $, \, :, . (which need to be sanitized)
    # Pattern matches: cell_type followed by whitespace, then instance_name (can contain $, \, letters, digits, underscore, dot, colon)
    # Note: \S+ matches any non-whitespace characters, but we'll validate it's a valid instance name
    instance_pattern = re.compile(r'(\w+)\s+([\w\$\.:\\]+)\s*\(')
    
    def sanitize_identifier(name: str) -> str:
        """
        Sanitize Verilog identifier by replacing invalid characters.
        Verilog identifiers can contain letters, digits, underscores, and $.
        However, to ensure compatibility and avoid issues, we replace
        all special characters ($, \, :, ., etc.) with underscores.
        """
        if not name:
            return name
        # Replace all special characters with underscore
        # Valid Verilog identifier characters: letters, digits, underscore, $
        # But we sanitize $, \, :, . to ensure clean names
        sanitized = name
        # Replace $, \, :, . with underscore
        sanitized = sanitized.replace('$', '_')
        sanitized = sanitized.replace('\\', '_')  # Backslash
        sanitized = sanitized.replace(':', '_')
        sanitized = sanitized.replace('.', '_')
        # Ensure it doesn't start with a digit (Verilog requirement)
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line contains an instance declaration
        match = instance_pattern.search(line)
        
        # Skip module declarations
        if match and match.group(1).lower() == 'module':
            output_lines.append(line)
            i += 1
            continue
        
        if match:
            # If this is the first instance we encounter, add the line at the end of wire definitions
            if not wire_section_ended:
                output_lines.append("    assign n2 = clk;")
                wire_section_ended = True
            
            cell_type = match.group(1)
            instance_name = match.group(2)
            stats['total_instances'] += 1
            
            # Check if this is a buffer instance
            is_buffer = is_buffer_cell(cell_type)
            
            # Keep original name for placement map lookup
            original_name = instance_name
            
            # All instances should be renamed - no special cases
            is_special = False
            
            # Look up instance name in placement map
            # Try original name first (with $, :, .), then try sanitized version
            new_name = None
            lookup_name = original_name  # Use original name for replacement in line
            
            # First try original name (as it appears in Verilog and placement map)
            if original_name in placement_map:
                new_name = placement_map[original_name]
            else:
                # Try sanitized version (in case Verilog already sanitized it)
                sanitized_name = sanitize_identifier(original_name)
                if sanitized_name != original_name and sanitized_name in placement_map:
                    new_name = placement_map[sanitized_name]
                    lookup_name = sanitized_name
                elif 'auto_insbuf' in instance_name or 'auto$insbuf' in instance_name:
                    # Verilog might have sanitized name, try with $ format
                    # Pattern: _auto_insbuf.cc:97:execute_11909 -> $auto$insbuf.cc:97:execute$11909
                    if instance_name.startswith('_auto_insbuf'):
                        dollar_name = instance_name.replace('_auto_insbuf', '$auto$insbuf', 1)
                    elif instance_name.startswith('$auto$insbuf'):
                        dollar_name = instance_name
                    else:
                        dollar_name = instance_name.replace('_auto_insbuf', '$auto$insbuf')
                    
                    # Replace execute_ with execute$ (no underscore before execute)
                    dollar_name = re.sub(r'execute_(\d+)', lambda m: f'execute${m.group(1)}', dollar_name)
                    if dollar_name in placement_map:
                        new_name = placement_map[dollar_name]
                        lookup_name = instance_name
            
            # Collect all lines for this instance (until closing );)
            instance_lines = [line]
            i += 1
            paren_count = 1  # We've seen the opening (
            
            while i < len(lines) and paren_count > 0:
                current_line = lines[i]
                instance_lines.append(current_line)
                # Count parentheses to find the closing );
                paren_count += current_line.count('(') - current_line.count(')')
                i += 1
            
            # Now process the instance
            if new_name:
                # Sanitize the physical slot name to ensure no special characters
                sanitized_slot_name = sanitize_identifier(new_name)
                
                # Replace instance name in the first line with sanitized physical slot name
                first_line = instance_lines[0]
                first_line = re.sub(
                    rf'({re.escape(cell_type)}\s+){re.escape(lookup_name)}(\s*\()',
                    rf'\1{sanitized_slot_name}\2',
                    first_line
                )
                instance_lines[0] = first_line
                stats['renamed_instances'] += 1
            else:
                # Instance not found in map - this is a problem
                stats['missing_in_map'].append(instance_name)
                # Warn but don't fail - might be a test case or special handling
                print(f"Warning: Instance '{instance_name}' not found in placement map")
                # Still sanitize the name to remove invalid characters even if not in map
                sanitized_name = sanitize_identifier(original_name)
                if sanitized_name != original_name:
                    # Match the full declaration structure: cell_type instance_name (
                    first_line = instance_lines[0]
                    first_line = re.sub(
                        rf'({re.escape(cell_type)}\s+){re.escape(original_name)}(\s*\()',
                        rf'\1{sanitized_name}\2',
                        first_line
                    )
                    instance_lines[0] = first_line
                stats['skipped_instances'] += 1
            
            # If this is a buffer, remove output pin connections
            if is_buffer:
                instance_lines = remove_buffer_output_pins(instance_lines)
                stats['buffer_instances_processed'] += 1
            
            # Add all instance lines to output
            output_lines.extend(instance_lines)
        else:
            # Not an instance declaration, just add the line
            output_lines.append(line)
            i += 1
    
    renamed_content = '\n'.join(output_lines)
    return renamed_content, stats


def validate_renaming(
    verilog_content: str,
    placement_map: Dict[str, str],
    design_name: str
) -> bool:
    """
    Validate that all instances in Verilog have corresponding entries in placement map.
    
    Args:
        verilog_content: Verilog file content
        placement_map: Placement map dictionary
        design_name: Design name for error reporting
        
    Returns:
        True if validation passes, False otherwise
    """
    instances = find_instance_declarations(verilog_content)
    missing = []
    
    # All instances should be in the placement map - no special cases
    for line_num, line, cell_type, instance_name in instances:
        if instance_name not in placement_map:
            missing.append((line_num, instance_name))
    
    if missing:
        print(f"\nERROR: Found {len(missing)} instances in Verilog that are not in placement map:")
        for line_num, instance_name in missing[:10]:  # Show first 10
            print(f"  Line {line_num}: {instance_name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Rename Verilog instances to match physical slot names from placement map'
    )
    parser.add_argument(
        '--design',
        required=True,
        help='Design name (e.g., "6502")'
    )
    parser.add_argument(
        '--build-dir',
        default='build',
        help='Build directory (default: build)'
    )
    parser.add_argument(
        '--final-v',
        help='Path to input Verilog file ([design]_final.v). '
             'If not specified, uses build/[design]/[design]_final.v'
    )
    parser.add_argument(
        '--map',
        help='Path to placement map file ([design].map). '
             'If not specified, uses build/[design]/[design].map'
    )
    parser.add_argument(
        '--output',
        help='Path to output Verilog file ([design]_renamed.v). '
             'If not specified, uses build/[design]/[design]_renamed.v'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation of instances in placement map'
    )
    
    args = parser.parse_args()
    
    # Build paths
    design_name = args.design
    build_dir = args.build_dir
    design_dir = os.path.join(build_dir, design_name)
    
    # Determine input Verilog path
    if args.final_v:
        final_v_path = args.final_v
    else:
        # Try multiple possible locations
        possible_paths = [
            os.path.join(design_dir, f"{design_name}_final.v"),
            f"test_eco_{design_name}.v",  # Fallback to test_eco format
            os.path.join(design_dir, f"test_eco_{design_name}.v")
        ]
        final_v_path = None
        for path in possible_paths:
            if os.path.exists(path):
                final_v_path = path
                break
        
        if not final_v_path:
            raise FileNotFoundError(
                f"Verilog file not found. Tried: {', '.join(possible_paths)}"
            )
    
    # Determine placement map path
    if args.map:
        map_path = args.map
    else:
        # Try _with_eco.map first (if ECO was run), then fall back to .map
        eco_map_path = os.path.join(design_dir, f"{design_name}_eco.map")
        regular_map_path = os.path.join(design_dir, f"{design_name}.map")
        
        if os.path.exists(eco_map_path):
            map_path = eco_map_path
            print(f"  Using ECO placement map: {map_path}")
        else:
            map_path = regular_map_path
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(design_dir, f"{design_name}_renamed.v")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Renaming instances for design: {design_name}")
    print(f"  Input Verilog: {final_v_path}")
    print(f"  Placement map: {map_path}")
    print(f"  Output: {output_path}")
    print()
    
    # Parse placement map
    print("Parsing placement map...")
    placement_map = parse_placement_map(map_path)
    print(f"  Found {len(placement_map)} mappings")
    
    # Read Verilog file
    print("Reading Verilog file...")
    with open(final_v_path, 'r', encoding='utf-8') as f:
        verilog_content = f.read()
    
    # Find instances for reporting
    instances = find_instance_declarations(verilog_content)
    print(f"  Found {len(instances)} instance declarations")
    
    # Validate (optional)
    if not args.skip_validation:
        print("Validating instances...")
        if not validate_renaming(verilog_content, placement_map, design_name):
            print("\nWARNING: Validation failed. Some instances are missing from placement map.")
            print("Use --skip-validation to proceed anyway.")
            return 1
    
    # Rename instances
    print("Renaming instances...")
    renamed_content, stats = rename_instances_in_verilog(
        verilog_content,
        placement_map,
        design_name
    )
    
    # Add header comment
    header = f"// Renamed Verilog netlist for {design_name}\n"
    header += f"// Logical instance names replaced with physical slot names from placement.map\n"
    header += f"// Generated from: {os.path.basename(final_v_path)}\n"
    header += f"// Total instances: {stats['total_instances']}, Renamed: {stats['renamed_instances']}\n\n"
    
    # Write output file
    print("Writing output file...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(renamed_content)
    
    # Print statistics
    print()
    print("=" * 60)
    print("Renaming Statistics")
    print("=" * 60)
    print(f"Total instances found: {stats['total_instances']}")
    print(f"Successfully renamed: {stats['renamed_instances']}")
    print(f"Skipped (total): {stats['skipped_instances']}")
    print(f"  - Missing from placement map: {len(stats['missing_in_map'])}")
    print(f"Buffer instances processed (output pins removed): {stats['buffer_instances_processed']}")
    if stats['missing_in_map']:
        if len(stats['missing_in_map']) <= 10:
            print(f"  Missing instances:")
            for name in stats['missing_in_map']:
                print(f"    - {name}")
        else:
            print(f"  Missing instances (showing first 10 of {len(stats['missing_in_map'])}):")
            for name in stats['missing_in_map'][:10]:
                print(f"    - {name}")
    print("=" * 60)
    print()
    print(f"✓ Renamed Verilog saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())