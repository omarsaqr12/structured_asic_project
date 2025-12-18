#!/usr/bin/env python3
"""
Critical Path Overlay Visualization Script
Parse the worst path from _setup.rpt, get the cell locations, and draw a bright red line on the layout connecting them.
"""

import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict, deque


def parse_setup_report_for_worst_path(setup_rpt_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse setup timing report to extract the worst path (lowest slack).
    Extracts startpoint, endpoint, and nets in the path.
    
    Returns:
        Dict with: {slack, startpoint, endpoint, nets, cells}
        Returns None if no paths found
    """
    worst_path = None
    worst_slack = float('inf')
    
    current_path = None
    in_data_path = False  # Track if we're in the data path section
    
    with open(setup_rpt_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for "No paths found"
        if line == "No paths found.":
            print("  ⚠ WARNING: Report contains 'No paths found'")
            return None
        
        # Check if we've reached the end of data path section
        if 'data arrival time' in line.lower():
            in_data_path = False
        
        # Check for startpoint
        if line.startswith('Startpoint:'):
            # Save previous path if exists
            if current_path is not None and current_path['slack'] is not None:
                if current_path['slack'] < worst_slack:
                    worst_slack = current_path['slack']
                    worst_path = current_path.copy()
            
            # Start new path
            match = re.match(r'Startpoint:\s+(.+)', line)
            if match:
                current_startpoint = match.group(1).split()[0]  # Get instance name
                current_path = {
                    'startpoint': current_startpoint,
                    'startpoint_pin': None,  # Will be extracted from path
                    'endpoint': None,
                    'endpoint_pin': None,  # Will be extracted from path
                    'slack': None,
                    'nets': [],  # List of nets in the path
                    'cells': []   # Will be populated by tracing nets
                }
                in_data_path = False  # Reset flag, will be set when Description section starts
        
        # Check for endpoint
        elif line.startswith('Endpoint:'):
            match = re.match(r'Endpoint:\s+(.+)', line)
            if match:
                current_endpoint = match.group(1).split()[0]  # Get instance name
                if current_path:
                    current_path['endpoint'] = current_endpoint
        
        # Check if we're entering the path description section (starts with "Description" or separator line)
        if current_path and ('Description' in line or line.startswith('---')):
            in_data_path = True
        
        # Check for slack value
        # Format: "                                 58.36   slack (MET)" or "-0.15   slack (VIOLATED)"
        if 'slack' in line.lower() and current_path:
            slack_match = re.search(r'([+-]?\d+\.?\d*)\s+slack', line)
            if slack_match:
                slack_value = float(slack_match.group(1))
                current_path['slack'] = slack_value
        
        # Extract cells and nets from path description lines
        # Format: "                  0.19    0.00    0.58 ^ T14Y58__R1_DFBBP_0/Q (sky130_fd_sc_hd__dfbbp_1)"
        # Format: "                                         n125 (net)"
        elif current_path and in_data_path:
            # Extract cell names with pin information (format: T14Y58__R1_DFBBP_0/Q)
            cell_pin_match = re.search(r'(T\d+Y\d+__R\d+_[^/\s]+)/(\w+)', line)
            if cell_pin_match:
                cell_name = cell_pin_match.group(1).rstrip()
                pin_name = cell_pin_match.group(2)
                
                # Store startpoint pin (first cell's output pin)
                if cell_name == current_path['startpoint'] and current_path['startpoint_pin'] is None:
                    # Check if this is an output pin
                    if pin_name in ['Q', 'Q_N', 'Y', 'X', 'Z']:
                        current_path['startpoint_pin'] = pin_name
                
                # Store endpoint pin (last cell's input pin)
                if cell_name == current_path['endpoint']:
                    # Check if this is an input pin
                    if pin_name in ['D', 'A', 'B', 'C', 'CLK']:
                        current_path['endpoint_pin'] = pin_name
                
                # Add cells in order as they appear in the path
                if cell_name not in current_path['cells']:
                    current_path['cells'].append(cell_name)
            
            # Extract net names (format: n### (net) or net_###)
            net_match = re.search(r'\b(n\d+|net_\d+)\s*\(net\)', line)
            if net_match:
                net_name = net_match.group(1)
                if net_name not in current_path['nets']:
                    current_path['nets'].append(net_name)
    
        i += 1
    
    # Don't forget the last path
    if current_path is not None and current_path['slack'] is not None:
        if current_path['slack'] < worst_slack:
            worst_slack = current_path['slack']
            worst_path = current_path.copy()
    
    # Debug: if no path found, print what we have
    if worst_path is None and current_path is not None:
        print(f"  Debug: Found path but slack is None. Path has {len(current_path.get('nets', []))} nets")
        print(f"  Debug: Startpoint: {current_path.get('startpoint')}, Endpoint: {current_path.get('endpoint')}")
    
    return worst_path


def parse_verilog_netlist(verilog_path: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[Tuple[str, str]]]]:
    """
    Parse Verilog netlist to build connectivity graph.
    
    Returns:
        cell_connections: Dict mapping cell_name -> {pin_name: net_name}
        net_connections: Dict mapping net_name -> list of (cell_name, pin_name) tuples
    """
    cell_connections = {}
    net_connections = defaultdict(list)
    
    if not os.path.exists(verilog_path):
        print(f"  ⚠ WARNING: Verilog file not found: {verilog_path}")
        return cell_connections, dict(net_connections)
    
    current_cell = None
    current_pins = {}
    
    with open(verilog_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('//') or line.startswith('module') or line.startswith('endmodule'):
                continue
            
            # Match cell instantiation: sky130_fd_sc_hd__dfbbp_1 T14Y58__R1_DFBBP_0 (
            cell_match = re.match(r'(\S+)\s+(\S+)\s*\(', line)
            if cell_match:
                # Save previous cell if exists
                if current_cell and current_pins:
                    cell_connections[current_cell] = current_pins
                    # Add to net connections
                    for pin_name, net_name in current_pins.items():
                        net_connections[net_name].append((current_cell, pin_name))
                
                current_cell = cell_match.group(2)
                current_pins = {}
                continue
            
            # Match pin connection: .PIN_NAME(net_name)
            pin_match = re.search(r'\.(\w+)\s*\((\S+)\)', line)
            if pin_match and current_cell:
                pin_name = pin_match.group(1)
                net_name = pin_match.group(2)
                current_pins[pin_name] = net_name
    
    # Don't forget the last cell
    if current_cell and current_pins:
        cell_connections[current_cell] = current_pins
        for pin_name, net_name in current_pins.items():
            net_connections[net_name].append((current_cell, pin_name))
    
    return cell_connections, dict(net_connections)


def trace_path_in_netlist(startpoint_cell: str, startpoint_pin: str,
                          endpoint_cell: str, endpoint_pin: str,
                          cell_connections: Dict[str, Dict[str, str]],
                          net_connections: Dict[str, List[Tuple[str, str]]],
                          cell_locations: Dict[str, Tuple[float, float]]) -> List[str]:
    """
    Trace the complete logical path from startpoint to endpoint through the netlist.
    Uses BFS to find all cells in the path following signal flow (outputs -> inputs).
    
    Returns ordered list of cells in the path.
    """
    path_cells = []
    
    # Get the net connected to startpoint output pin
    if startpoint_cell not in cell_connections:
        return path_cells
    
    start_net = cell_connections[startpoint_cell].get(startpoint_pin)
    if not start_net:
        return path_cells
    
    # Get the net connected to endpoint input pin
    if endpoint_cell not in cell_connections:
        return path_cells
    
    end_net = cell_connections[endpoint_cell].get(endpoint_pin)
    if not end_net:
        return path_cells
    
    # Build connectivity: net -> cells that drive it (output pins)
    # and net -> cells that receive it (input pins)
    net_drivers = defaultdict(list)  # net -> [(cell, pin)]
    net_receivers = defaultdict(list)  # net -> [(cell, pin)]
    
    for cell_name, pins in cell_connections.items():
        if cell_name not in cell_locations:
            continue
        for pin_name, net_name in pins.items():
            # Determine if pin is input or output based on common patterns
            if pin_name in ['Q', 'Q_N', 'Y', 'X', 'Z']:
                net_drivers[net_name].append((cell_name, pin_name))
            elif pin_name in ['A', 'B', 'C', 'D', 'CLK', 'RESET_B', 'SET_B']:
                net_receivers[net_name].append((cell_name, pin_name))
    
    # BFS from startpoint to endpoint following signal flow
    visited_cells = set()
    visited_nets = set()
    queue = deque([(startpoint_cell, [startpoint_cell])])
    visited_cells.add(startpoint_cell)
    
    while queue:
        current_cell, current_path = queue.popleft()
        
        # If we reached the endpoint, return the path
        if current_cell == endpoint_cell:
            # Filter to only cells that exist in cell_locations
            path_cells = [c for c in current_path if c in cell_locations]
            return path_cells
        
        # Get all nets driven by current cell (output pins)
        if current_cell not in cell_connections:
            continue
        
        for pin_name, net_name in cell_connections[current_cell].items():
            # Only follow output pins
            if pin_name not in ['Q', 'Q_N', 'Y', 'X', 'Z']:
                continue
            
            if net_name in visited_nets:
                continue
            
            visited_nets.add(net_name)
            
            # Find all cells that receive this net (input pins)
            for receiver_cell, receiver_pin in net_receivers.get(net_name, []):
                if receiver_cell in visited_cells:
                    continue
                
                if receiver_cell not in cell_locations:
                    continue
                
                # Add to path and continue BFS
                new_path = current_path + [receiver_cell]
                visited_cells.add(receiver_cell)
                queue.append((receiver_cell, new_path))
    
    # If BFS didn't find path, return empty list
    return path_cells


def parse_def_routing(def_path: str, target_nets: Set[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse DEF file to extract routing information for specific nets.
    
    Format: + ROUTED layer ( x y ) via_type ( x y ) ...
            NEW layer ( x y ) via_type ...
    
    Returns:
        Dict mapping net_name -> list of routing segments
        Each segment is: {'layer': 'met1', 'points': [(x, y), ...], 'via_type': 'M1M2_PR'}
    """
    routing_paths = {}
    
    if not os.path.exists(def_path):
        return routing_paths
    
    in_nets = False
    current_net = None
    current_segments = []
    current_layer = None
    current_points = []
    
    # Normalize net names (n125 -> net_125, net_125 -> n125)
    net_name_map = {}
    for net in target_nets:
        if net.startswith('n') and net[1:].isdigit():
            net_name_map[f"net_{net[1:]}"] = net
            net_name_map[net] = net
        elif net.startswith('net_'):
            num = net.split('_')[1]
            net_name_map[f"n{num}"] = net
            net_name_map[net] = net
    
    with open(def_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('NETS'):
                in_nets = True
                continue
            elif line.startswith('END NETS'):
                if current_net and current_segments:
                    routing_paths[current_net] = current_segments
                break
            
            if in_nets:
                # Check for new net definition: - NET_NAME
                if line.startswith('- ') and not line.startswith('- (') and not line.startswith('- +'):
                    # Save previous net if exists
                    if current_net and current_segments:
                        routing_paths[current_net] = current_segments
                    
                    # Extract net name
                    match = re.match(r'-\s+(\S+)', line)
                    if match:
                        net_name = match.group(1)
                        # Check if this net is in our target set
                        if net_name in net_name_map:
                            current_net = net_name_map[net_name]
                        elif net_name in target_nets:
                            current_net = net_name
                        else:
                            current_net = None
                        current_segments = []
                        current_layer = None
                        current_points = []
                    continue
                
                # Only process routing if this is a target net
                if current_net is None:
                    continue
                
                # Check for ROUTED line: + ROUTED layer ( x y ) ...
                routed_match = re.search(r'\+ ROUTED\s+(\w+)', line)
                if routed_match:
                    current_layer = routed_match.group(1)
                    current_points = []
                    # Extract initial coordinates - look for first coordinate pair
                    coord_match = re.search(r'\(\s*([\d*]+)\s+([\d*]+)\s*\)', line)
                    if coord_match:
                        x_str, y_str = coord_match.groups()
                        x = None if x_str == '*' else float(x_str)
                        y = None if y_str == '*' else float(y_str)
                        if x is not None and y is not None:
                            current_points.append((x, y))
                    # Continue parsing coordinates from this line
                    coords = re.findall(r'\(\s*([\d*]+)\s+([\d*]+)\s*\)', line)
                    last_x, last_y = current_points[-1] if current_points else (None, None)
                    for i, (x_str, y_str) in enumerate(coords):
                        if i == 0 and current_points:
                            continue  # Already added first coordinate
                        x = last_x if x_str == '*' else float(x_str)
                        y = last_y if y_str == '*' else float(y_str)
                        if x is not None and y is not None:
                            current_points.append((x, y))
                            last_x, last_y = x, y
                    continue
                
                # Check for NEW layer line: NEW layer ( x y ) ...
                new_match = re.search(r'NEW\s+(\w+)', line)
                if new_match:
                    # Save previous segment
                    if current_layer and current_points:
                        current_segments.append({
                            'layer': current_layer,
                            'points': current_points.copy()
                        })
                    current_layer = new_match.group(1)
                    current_points = []
                    # Extract initial coordinates
                    coord_match = re.search(r'\(\s*([\d*]+)\s+([\d*]+)\s*\)', line)
                    if coord_match:
                        x_str, y_str = coord_match.groups()
                        x = None if x_str == '*' else float(x_str)
                        y = None if y_str == '*' else float(y_str)
                        if x is not None and y is not None:
                            current_points.append((x, y))
                    # Continue parsing coordinates from this line
                    coords = re.findall(r'\(\s*([\d*]+)\s+([\d*]+)\s*\)', line)
                    last_x, last_y = current_points[-1] if current_points else (None, None)
                    for i, (x_str, y_str) in enumerate(coords):
                        if i == 0 and current_points:
                            continue  # Already added first coordinate
                        x = last_x if x_str == '*' else float(x_str)
                        y = last_y if y_str == '*' else float(y_str)
                        if x is not None and y is not None:
                            current_points.append((x, y))
                            last_x, last_y = x, y
                    continue
                
                # Extract coordinates and via types from current line
                if current_layer:
                    # Pattern: ( x y ) or ( * y ) or ( x * ) followed by via types like L1M1_PR, M1M2_PR
                    # Extract all coordinates first
                    coords = re.findall(r'\(\s*([\d*]+)\s+([\d*]+)\s*\)', line)
                    last_x, last_y = None, None
                    if current_points:
                        last_x, last_y = current_points[-1]
                    
                    for x_str, y_str in coords:
                        x = last_x if x_str == '*' else float(x_str)
                        y = last_y if y_str == '*' else float(y_str)
                        if x is not None and y is not None:
                            current_points.append((x, y))
                            last_x, last_y = x, y
    
    # Don't forget the last segment
    if current_net and current_layer and current_points:
        if current_net not in routing_paths:
            routing_paths[current_net] = []
        routing_paths[current_net].append({
            'layer': current_layer,
            'points': current_points
        })
    
    return routing_paths


def parse_def_nets(def_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse DEF file to extract net connectivity.
    
    Format: - NET_NAME ( CELL_NAME PIN_NAME ) ( CELL_NAME PIN_NAME ) ... + USE SIGNAL ;
    
    Returns:
        Dict mapping net_name -> list of (cell_name, pin_name) tuples
    """
    net_connections = {}
    
    if not os.path.exists(def_path):
        print(f"  ⚠ WARNING: DEF file not found: {def_path}")
        return net_connections
    
    in_nets = False
    current_net = None
    current_connections = []
    
    with open(def_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('NETS'):
                in_nets = True
                continue
            elif line.startswith('END NETS'):
                if current_net and current_connections:
                    net_connections[current_net] = current_connections
                break
            
            if in_nets:
                # Check for new net definition: - NET_NAME
                if line.startswith('- ') and not line.startswith('- (') and not line.startswith('- +'):
                    # Save previous net if exists
                    if current_net and current_connections:
                        net_connections[current_net] = current_connections
                    
                    # Extract net name
                    match = re.match(r'-\s+(\S+)', line)
                    if match:
                        current_net = match.group(1)
                        current_connections = []
                
                # Extract connections: ( CELL_NAME PIN_NAME )
                # Handle multi-line net definitions
                connections = re.findall(r'\(\s*(\S+)\s+(\S+)\s*\)', line)
                for cell_name, pin_name in connections:
                    # Skip PIN connections (they're I/O ports, not cells)
                    if cell_name != 'PIN':
                        current_connections.append((cell_name, pin_name))
    
    return net_connections


def trace_path_through_nets(timing_cells: List[str], nets: List[str], 
                            net_connections: Dict[str, List[Tuple[str, str]]],
                            cell_locations: Dict[str, Tuple[float, float]]) -> List[str]:
    """
    Trace the path through the given nets, merging timing report cells with 
    additional cells found on the nets. Preserves order from timing report.
    
    Returns ordered list of all cells in the path.
    """
    path_cells = []
    visited_cells = set()
    timing_cells_set = set(timing_cells)
    
    # Build a mapping of which cells are on which nets
    net_to_cells = {}
    for net_name in nets:
        # Try to find the net in DEF (handle name format differences)
        def_net_name = None
        if net_name in net_connections:
            def_net_name = net_name
        else:
            # Try alternative formats: n125 -> net_125, net_125 -> n125
            alt_names = [
                f"net_{net_name[1:]}" if net_name.startswith('n') and net_name[1:].isdigit() else None,
                f"n{net_name.split('_')[1]}" if net_name.startswith('net_') else None,
            ]
            for alt_name in alt_names:
                if alt_name and alt_name in net_connections:
                    def_net_name = alt_name
                    break
        
        if def_net_name:
            # Get all cells on this net
            cells_on_net = []
            for cell_name, pin_name in net_connections[def_net_name]:
                if cell_name in cell_locations:
                    cells_on_net.append(cell_name)
            net_to_cells[net_name] = cells_on_net
    
    # Start with cells from timing report (they're in the correct order)
    for cell in timing_cells:
        if cell in cell_locations:
            path_cells.append(cell)
            visited_cells.add(cell)
    
    # For each net, add any additional cells that are on the net but not in timing report
    # Insert them near the timing report cells that are also on this net
    for net_name in nets:
        if net_name not in net_to_cells:
            continue
        
        net_cells = net_to_cells[net_name]
        
        # Find positions of timing report cells on this net
        timing_positions = []
        for i, cell in enumerate(path_cells):
            if cell in net_cells:
                timing_positions.append(i)
        
        # Add new cells from this net near the timing report cells
        # Insert them after the last timing report cell on this net
        if timing_positions:
            insert_pos = timing_positions[-1] + 1
            for cell in net_cells:
                if cell not in visited_cells:
                    path_cells.insert(insert_pos, cell)
                    visited_cells.add(cell)
                    insert_pos += 1
        else:
            # No timing report cells on this net, append at end
            for cell in net_cells:
                if cell not in visited_cells:
                    path_cells.append(cell)
                    visited_cells.add(cell)
    
    return path_cells


def parse_def_file(def_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Parse DEF file to extract cell locations.
    
    Format: - INST_NAME CELL_TYPE + FIXED ( X Y ) ORIENT ;
    
    Returns:
        Dict mapping instance_name -> (x, y) in microns
    """
    cell_locations = {}
    
    if not os.path.exists(def_path):
        print(f"  ⚠ WARNING: DEF file not found: {def_path}")
        return cell_locations
    
    in_components = False
    
    with open(def_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('COMPONENTS'):
                in_components = True
                continue
            elif line.startswith('END COMPONENTS'):
                in_components = False
                continue
            
            if in_components and line.startswith('-'):
                # Parse: - INST_NAME CELL_TYPE + FIXED ( X Y ) ORIENT ;
                # Example: - T14Y58__R1_DFBBP_0 sky130_fd_sc_hd__dfbbp_1 + FIXED ( 14200 5000 ) N ;
                match = re.match(r'-\s+(\S+)\s+\S+\s+\+\s+FIXED\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', line)
                if match:
                    inst_name = match.group(1)
                    x = float(match.group(2))
                    y = float(match.group(3))
                    cell_locations[inst_name] = (x, y)
    
    return cell_locations


def plot_worst_path_on_layout(worst_path: Dict[str, Any], 
                               cell_locations: Dict[str, Tuple[float, float]],
                               def_path: str,
                               output_path: str,
                               design_name: str,
                               routing_paths: Optional[Dict[str, List[Dict[str, Any]]]] = None):
    """
    Plot the worst timing path on the layout.
    Draws routing paths through metal layers and tracks, not just straight lines.
    """
    if not worst_path or not worst_path.get('cells'):
        print("  ✗ ERROR: No worst path data available. Cannot plot path on layout.")
        return
    
    # Get die area from DEF file
    die_area = None
    with open(def_path, 'r') as f:
        for line in f:
            if line.startswith('DIEAREA'):
                # Format: DIEAREA ( 0 0 ) ( X Y ) ;
                match = re.search(r'\(\s*[\d.]+\s+[\d.]+\s+\)\s+\(\s*([\d.]+)\s+([\d.]+)\s+\)', line)
                if match:
                    die_area = (float(match.group(1)), float(match.group(2)))
                    break
    
    if not die_area:
        print("  ⚠ WARNING: Could not find DIEAREA in DEF file. Using cell bounds.")
        # Fallback: calculate from cell locations
        if cell_locations:
            xs = [loc[0] for loc in cell_locations.values()]
            ys = [loc[1] for loc in cell_locations.values()]
            die_area = (max(xs) + 1000, max(ys) + 1000)
        else:
            print("  ✗ ERROR: No cell locations available.")
            return
    
    # Get path cell locations
    path_cells = worst_path['cells']
    path_coords = []
    missing_cells = []
    
    for cell_name in path_cells:
        if cell_name in cell_locations:
            path_coords.append(cell_locations[cell_name])
        else:
            missing_cells.append(cell_name)
    
    if missing_cells:
        print(f"  ⚠ WARNING: {len(missing_cells)} cells not found in DEF file:")
        for cell in missing_cells[:5]:  # Show first 5
            print(f"    - {cell}")
        if len(missing_cells) > 5:
            print(f"    ... and {len(missing_cells) - 5} more")
    
    if not path_coords:
        print("  ✗ ERROR: No path cell locations found. Cannot plot path.")
        print(f"    Path had {len(path_cells)} cells, but {len(missing_cells)} were missing from DEF file.")
        return
    
    print(f"  ✓ Found {len(path_coords)} path cell locations (out of {len(path_cells)} cells in path)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot all cells as small dots (light gray)
    all_x = [loc[0] for loc in cell_locations.values()]
    all_y = [loc[1] for loc in cell_locations.values()]
    if all_x and all_y:
        ax.scatter(all_x, all_y, c='lightgray', s=1, alpha=0.3, label='All Cells')
    
    # Plot path cells as larger dots (blue)
    path_x = [coord[0] for coord in path_coords]
    path_y = [coord[1] for coord in path_coords]
    ax.scatter(path_x, path_y, c='blue', s=100, alpha=0.8, zorder=5, 
              edgecolors='black', linewidths=1, label='Path Cells')
    
    # Draw routing paths if available
    if routing_paths:
        # Layer colors (different colors for different metal layers)
        layer_colors = {
            'li1': '#FF6B6B',      # Light red (local interconnect)
            'met1': '#4ECDC4',     # Teal (metal 1)
            'met2': '#45B7D1',     # Blue (metal 2)
            'met3': '#96CEB4',     # Green (metal 3)
            'met4': '#FFEAA7',     # Yellow (metal 4)
            'met5': '#DDA15E',     # Orange (metal 5)
        }
        layer_widths = {
            'li1': 1.5,
            'met1': 2.0,
            'met2': 2.5,
            'met3': 3.0,
            'met4': 3.5,
            'met5': 4.0,
        }
        
        # Track which layers we've drawn for legend
        layers_drawn = set()
        nets_drawn = 0
        segments_drawn = 0
        
        for net_name, segments in routing_paths.items():
            for segment in segments:
                layer = segment.get('layer', 'met1')
                points = segment.get('points', [])
                
                if len(points) < 2:
                    continue
                
                seg_x = [p[0] for p in points]
                seg_y = [p[1] for p in points]
                
                color = layer_colors.get(layer, '#FF0000')
                width = layer_widths.get(layer, 2.0)
                
                # Only add to legend if this is the first time we see this layer
                label = f'{layer.upper()}' if layer not in layers_drawn else ''
                if layer not in layers_drawn:
                    layers_drawn.add(layer)
                
                ax.plot(seg_x, seg_y, color=color, linewidth=width, alpha=0.8, 
                       zorder=3, label=label)
                segments_drawn += 1
        
        if segments_drawn > 0:
            print(f"  ✓ Drew {segments_drawn} routing segments across {len(layers_drawn)} metal layers")
    else:
        # Fallback: Draw straight lines connecting path cells
        if len(path_coords) > 1:
            # Draw main path line in bright red
            ax.plot(path_x, path_y, color='#FF0000', linewidth=4, alpha=0.9, 
                   zorder=4, label='Critical Path')
            # Add a slightly thicker line for visibility
            ax.plot(path_x, path_y, color='#FF0000', linewidth=2, alpha=1.0, zorder=4)
    
    # Mark startpoint and endpoint
    if path_coords:
        # Startpoint (first cell) - green square
        ax.scatter(path_x[0], path_y[0], c='green', s=300, marker='s', 
                  edgecolors='black', linewidths=3, zorder=6, label='Startpoint')
        # Endpoint (last cell) - red square
        ax.scatter(path_x[-1], path_y[-1], c='red', s=300, marker='s', 
                  edgecolors='black', linewidths=3, zorder=6, label='Endpoint')
    
    # Set labels and title
    ax.set_xlabel('X (microns)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (microns)', fontsize=12, fontweight='bold')
    slack_value = worst_path.get('slack', 'N/A')
    if isinstance(slack_value, (int, float)):
        title = f'Critical Path Overlay - {design_name}\n'
        title += f'Worst Setup Path (Slack: {slack_value:.3f} ns)'
        if slack_value < 0:
            title += ' [VIOLATED]'
    else:
        title = f'Critical Path Overlay - {design_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    # Set reasonable axis limits
    if all_x and all_y:
        margin = 2000
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Critical path visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize worst timing path on layout with bright red line'
    )
    parser.add_argument('--design', required=True, help='Design name (e.g., arith, 6502)')
    parser.add_argument('--build-dir', default='build', help='Build directory (default: build)')
    parser.add_argument('--setup-rpt', default=None,
                       help='Path to setup report (default: build/{design}/{design}_setup.rpt)')
    parser.add_argument('--def-file', '--def', dest='def_file', default=None,
                       help='Path to DEF file (default: build/{design}/{design}_routed.def)')
    parser.add_argument('--output', default=None,
                       help='Output PNG file (default: build/{design}/{design}_critical_path.png)')
    
    args = parser.parse_args()
    
    # Build paths
    build_dir = args.build_dir
    design_name = args.design
    
    if args.setup_rpt:
        setup_rpt_path = args.setup_rpt
    else:
        setup_rpt_path = os.path.join(build_dir, design_name, f"{design_name}_setup.rpt")
    
    if args.def_file:
        def_path = args.def_file
    else:
        def_path = os.path.join(build_dir, design_name, f"{design_name}_routed.def")
    
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(build_dir, design_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{design_name}_critical_path.png")
    
    print("=" * 60)
    print(f"Critical Path Visualization for {design_name}")
    print("=" * 60)
    print(f"Setup Report: {setup_rpt_path}")
    print(f"DEF File: {def_path}")
    print(f"Output: {output_path}")
    print()
    
    # Parse setup report for worst path
    print("Step 1: Parsing setup report for worst path...")
    if not os.path.exists(setup_rpt_path):
        print(f"  ✗ ERROR: Setup report not found: {setup_rpt_path}")
        return
    
    worst_path = parse_setup_report_for_worst_path(setup_rpt_path)
    
    if not worst_path:
        print("  ✗ ERROR: No worst path found in report.")
        print("  This may indicate:")
        print("    - Report contains 'No paths found'")
        print("    - Report format is different than expected")
        print("    - Timing analysis did not complete successfully")
        return
    
    print(f"  ✓ Worst path found:")
    print(f"    Startpoint: {worst_path['startpoint']}")
    print(f"    Endpoint: {worst_path['endpoint']}")
    print(f"    Slack: {worst_path['slack']:.3f} ns")
    print(f"    Nets in path: {len(worst_path.get('nets', []))}")
    if worst_path.get('nets'):
        print(f"    Nets: {', '.join(worst_path['nets'][:10])}" + 
              (f" ... ({len(worst_path['nets']) - 10} more)" if len(worst_path['nets']) > 10 else ""))
    print()
    
    # Parse DEF file for cell locations
    print("Step 2: Parsing DEF file for cell locations...")
    cell_locations = parse_def_file(def_path)
    print(f"  ✓ Found {len(cell_locations)} cell locations in DEF file")
    print()
    
    # Parse Verilog netlist for complete path tracing
    verilog_path = os.path.join(build_dir, design_name, f"{design_name}_renamed.v")
    print("Step 3: Parsing Verilog netlist for path tracing...")
    cell_connections, net_connections_verilog = parse_verilog_netlist(verilog_path)
    print(f"  ✓ Parsed {len(cell_connections)} cells and {len(net_connections_verilog)} nets from Verilog")
    print()
    
    # Use ONLY the cells explicitly listed in the timing report
    # The timing report already shows the exact critical path - we should use it directly
    print("Step 4: Using exact critical path from timing report...")
    timing_cells = worst_path.get('cells', [])
    
    # Ensure startpoint and endpoint are included in correct order
    if worst_path['startpoint'] not in timing_cells:
        timing_cells.insert(0, worst_path['startpoint'])
    if worst_path['endpoint'] not in timing_cells:
        timing_cells.append(worst_path['endpoint'])
    
    # Filter to only cells that exist in cell_locations (preserve order)
    path_cells = []
    for cell in timing_cells:
        if cell in cell_locations:
            path_cells.append(cell)
    
    print(f"  ✓ Using {len(path_cells)} cells from timing report (exact critical path)")
    worst_path['cells'] = path_cells
    if len(path_cells) > 0:
        print(f"    First few cells: {', '.join(path_cells[:min(5, len(path_cells))])}")
        if len(path_cells) > 5:
            print(f"    ... and {len(path_cells) - 5} more cells")
    print()
    
    # Parse routing information for critical path nets
    print("Step 5: Parsing routing information for critical path nets...")
    critical_nets = set(worst_path.get('nets', []))
    routing_paths = parse_def_routing(def_path, critical_nets)
    print(f"  ✓ Found routing information for {len(routing_paths)} nets")
    if routing_paths:
        total_segments = sum(len(segments) for segments in routing_paths.values())
        print(f"    Total routing segments: {total_segments}")
    print()
    
    # Plot worst path
    print("Step 6: Plotting worst path on layout with routing...")
    plot_worst_path_on_layout(worst_path, cell_locations, def_path, output_path, design_name, routing_paths)
    
    print()
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

