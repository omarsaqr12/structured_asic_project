#!/usr/bin/env python3
"""
rename_spef.py - Fix net names in SPEF file

This script fixes net names in SPEF files by replacing 'net_<number>' with 'n<number>'
to match the Verilog netlist naming convention.

Usage:
    python3 rename_spef.py input.spef output.spef
    python3 rename_spef.py build/arith/arith.spef build/arith/arith_fixed.spef
"""

import argparse
import re
import sys
import os


def rename_nets_in_spef(input_spef_path: str, output_spef_path: str) -> dict:
    """
    Rename nets in SPEF file from 'net_<number>' to 'n<number>'.
    
    Args:
        input_spef_path: Path to input SPEF file
        output_spef_path: Path to output SPEF file
        
    Returns:
        Dictionary with statistics about the renaming
    """
    stats = {
        'total_lines': 0,
        'name_map_entries': 0,
        'renamed_entries': 0,
        'other_net_references': 0,
        'renamed_references': 0
    }
    
    if not os.path.exists(input_spef_path):
        raise FileNotFoundError(f"Input SPEF file not found: {input_spef_path}")
    
    # Pattern to match NAME_MAP entries: *123 net_10
    name_map_pattern = re.compile(r'^(\*\d+)\s+(net_)(\d+)(\s*)$')
    
    # Pattern to match other references to net_<number> (in comments, etc.)
    # But be careful not to match things like "net_10:12" which might be coordinates
    # We'll focus on NAME_MAP entries first, then check if there are other references
    net_ref_pattern = re.compile(r'\bnet_(\d+)\b')
    
    print(f"Reading SPEF file: {input_spef_path}")
    with open(input_spef_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    stats['total_lines'] = len(lines)
    output_lines = []
    in_name_map = False
    
    for line_num, line in enumerate(lines, 1):
        original_line = line
        
        # Check if we're entering NAME_MAP section
        if line.strip() == '*NAME_MAP':
            in_name_map = True
            output_lines.append(line)
            continue
        
        # Check if we're leaving NAME_MAP section (next section starts with *)
        # NAME_MAP ends when we hit a line that starts with * but is not a NAME_MAP entry
        if in_name_map:
            # NAME_MAP entries are like: *123 net_10
            # Other sections start with *D_NET, *CONN, etc.
            if line.startswith('*') and not re.match(r'^\*\d+\s+', line):
                in_name_map = False
                # Don't skip this line, process it below
        
        # Process NAME_MAP entries
        if in_name_map:
            match = name_map_pattern.match(line)
            if match:
                stats['name_map_entries'] += 1
                map_id = match.group(1)  # e.g., *123
                prefix = match.group(2)  # "net_"
                number = match.group(3)  # e.g., "10"
                trailing = match.group(4)  # whitespace
                
                # Replace net_<number> with n<number>
                new_line = f"{map_id} n{number}{trailing}\n"
                output_lines.append(new_line)
                stats['renamed_entries'] += 1
                continue
        
        # Check for other references to net_<number> in the file
        # (though SPEF typically uses mapped IDs, not direct names)
        if 'net_' in line:
            # Replace net_<number> with n<number>
            new_line = net_ref_pattern.sub(r'n\1', line)
            if new_line != line:
                stats['other_net_references'] += 1
                stats['renamed_references'] += 1
                output_lines.append(new_line)
                continue
        
        # Keep line as-is
        output_lines.append(line)
    
    # Write output file
    print(f"Writing corrected SPEF file: {output_spef_path}")
    os.makedirs(os.path.dirname(output_spef_path) if os.path.dirname(output_spef_path) else '.', exist_ok=True)
    
    with open(output_spef_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Fix net names in SPEF file (net_<number> -> n<number>)'
    )
    parser.add_argument(
        'input_spef',
        help='Input SPEF file path'
    )
    parser.add_argument(
        'output_spef',
        nargs='?',
        help='Output SPEF file path (default: input_spef with _fixed suffix)'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify input file in place (overwrites original)'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.in_place:
        output_path = args.input_spef
        print("WARNING: Modifying file in place. Original will be overwritten.")
    elif args.output_spef:
        output_path = args.output_spef
    else:
        # Default: add _fixed suffix
        base, ext = os.path.splitext(args.input_spef)
        output_path = f"{base}_fixed{ext}"
    
    try:
        stats = rename_nets_in_spef(args.input_spef, output_path)
        
        print()
        print("=" * 60)
        print("SPEF Net Renaming Summary")
        print("=" * 60)
        print(f"Total lines processed: {stats['total_lines']}")
        print(f"NAME_MAP entries found: {stats['name_map_entries']}")
        print(f"NAME_MAP entries renamed: {stats['renamed_entries']}")
        if stats['other_net_references'] > 0:
            print(f"Other net references found: {stats['other_net_references']}")
            print(f"Other net references renamed: {stats['renamed_references']}")
        print()
        print(f"✓ Corrected SPEF file saved to: {output_path}")
        print("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

