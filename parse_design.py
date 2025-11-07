#!/usr/bin/env python3
"""
Parser for design netlist files ([design_name]_mapped.json).
Creates a logical_db and netlist_graph.
"""

import json
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple


def parse_design(design_path: str) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Parse a Yosys-generated JSON netlist.
    
    Returns:
        logical_db: Dict mapping cell_type -> List of instance names
        netlist_graph: Dict mapping instance_name -> {type, connections}
    """
    with open(design_path, 'r') as f:
        data = json.load(f)
    
    logical_db = defaultdict(list)
    netlist_graph = {}
    
    # Find the top module
    modules = data.get('modules', {})
    top_module = None
    for mod_name, mod_data in modules.items():
        if mod_data.get('attributes', {}).get('top') == '00000000000000000000000000000001':
            top_module = mod_name
            break
    
    if not top_module:
        # If no top attribute, use the first module with cells
        for mod_name, mod_data in modules.items():
            if 'cells' in mod_data and len(mod_data['cells']) > 0:
                top_module = mod_name
                break
    
    if not top_module:
        raise ValueError("No top module found in netlist")
    
    module_data = modules[top_module]
    cells = module_data.get('cells', {})
    
    # Parse each cell instance
    for inst_name, inst_data in cells.items():
        cell_type = inst_data.get('type', 'unknown')
        
        # Store in logical_db
        logical_db[cell_type].append(inst_name)
        
        # Store in netlist_graph
        netlist_graph[inst_name] = {
            'type': cell_type,
            'connections': inst_data.get('connections', {})
        }
    
    return dict(logical_db), netlist_graph


if __name__ == '__main__':
    # Test the parser
    logical_db, netlist_graph = parse_design('designs/6502_mapped.json')
    
    print("Logical DB Summary:")
    for cell_type, instances in sorted(logical_db.items()):
        print(f"  {cell_type}: {len(instances)} instances")
    
    print(f"\nTotal instances: {len(netlist_graph)}")

