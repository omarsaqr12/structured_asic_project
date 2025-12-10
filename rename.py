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


def rename_instances_in_verilog(
    verilog_content: str,
    placement_map: Dict[str, str],
    design_name: str
) -> Tuple[str, Dict[str, int]]:
    """
    Rename instances in Verilog content using placement map.
    
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
        'missing_in_map': []
    }
    
    # Special prefixes that should be handled differently
    # These are typically not in the original placement map but may be added by ECO
    # Note: eco_conb_1 should be in the _with_eco.map file and will be renamed normally
    special_prefixes = ['cts_buffer_', 'tie_low_cell', 'eco_unused_']
    
    # Pattern to match instance declarations: cell_type instance_name (
    # Exclude module declarations
    # Updated pattern to handle instance names with $ (which need to be sanitized)
    # Pattern matches: cell_type followed by whitespace, then instance_name (can contain $, letters, digits, underscore, dot, colon)
    # Note: \S+ matches any non-whitespace characters, but we'll validate it's a valid instance name
    instance_pattern = re.compile(r'(\w+)\s+([\w\$\.:]+)\s*\(')
    
    def sanitize_identifier(name: str) -> str:
        """
        Sanitize Verilog identifier by replacing invalid characters.
        Verilog identifiers cannot start with $, so we replace $ with _.
        """
        if name.startswith('$'):
            # Replace $ with _ to make it a valid Verilog identifier
            sanitized = name.replace('$', '_')
            return sanitized
        return name
    
    for line in lines:
        # Check if this line contains an instance declaration
        match = instance_pattern.search(line)
        
        # Skip module declarations
        if match and match.group(1).lower() == 'module':
            output_lines.append(line)
            continue
        
        if match:
            cell_type = match.group(1)
            instance_name = match.group(2)
            stats['total_instances'] += 1
            
            # Sanitize instance name if it contains invalid characters
            original_name = instance_name
            if '$' in instance_name:
                instance_name = sanitize_identifier(instance_name)
                # Replace the invalid name in the line using regex to avoid partial matches
                # Match the instance name as a whole word (not part of another identifier)
                line = re.sub(
                    rf'\b{re.escape(original_name)}\b',
                    instance_name,
                    line
                )
            
            # Check if this instance should be renamed
            should_rename = True
            is_special = False
            
            # Check if it's a special case
            for prefix in special_prefixes:
                if instance_name.startswith(prefix):
                    is_special = True
                    break
            
            # Look up instance name in placement map (use sanitized name)
            if instance_name in placement_map:
                new_name = placement_map[instance_name]
                # Replace instance name in the line
                # Use word boundary to avoid partial matches
                line = re.sub(
                    rf'\b{re.escape(instance_name)}\b',
                    new_name,
                    line
                )
                stats['renamed_instances'] += 1
            else:
                # Instance not found in map
                if not is_special:
                    # This is a problem - regular instances should be in the map
                    stats['missing_in_map'].append(instance_name)
                    # Warn but don't fail - might be a test case or special handling
                    print(f"Warning: Instance '{instance_name}' not found in placement map")
                stats['skipped_instances'] += 1
        
        output_lines.append(line)
    
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
    
    special_prefixes = ['cts_buffer_', 'tie_low_cell', 'eco_unused_']
    
    for line_num, line, cell_type, instance_name in instances:
        # Skip special cases
        is_special = any(instance_name.startswith(prefix) for prefix in special_prefixes)
        
        if not is_special and instance_name not in placement_map:
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
        eco_map_path = os.path.join(design_dir, f"{design_name}_with_eco.map")
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
    print(f"Skipped (special cases): {stats['skipped_instances']}")
    if stats['missing_in_map']:
        print(f"Missing in map: {len(stats['missing_in_map'])}")
        if len(stats['missing_in_map']) <= 10:
            for name in stats['missing_in_map']:
                print(f"  - {name}")
    print("=" * 60)
    print()
    print(f"✓ Renamed Verilog saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())

