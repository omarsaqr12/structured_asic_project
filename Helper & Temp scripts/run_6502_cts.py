#!/usr/bin/env python3
"""Run CTS for 6502 design - both greedy and best SA placements"""
import os
import sys
import shutil
from cts_htree import (synthesize_clock_tree, save_cts_tree, save_buffer_mapping,
                       update_placement_map, update_placement_json, update_netlist_connections)

# Configuration
design_file = "designs/6502_mapped.json"
fabric_cells = "fabric/fabric_cells.yaml"

# Placement configurations
placements = {
    'greedy': {
        'placement_json': 'build/6502/greedy/6502_placement.json',
        'placement_map': 'build/6502/greedy/6502.map',
        'output_dir': 'build/6502/cts/greedy'
    },
    'best': {
        'placement_json': 'build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/6502_placement.json',
        'placement_map': 'build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/6502.map',
        'output_dir': 'build/6502/cts/best'
    }
}

def run_cts_for_placement(name: str, config: dict):
    """Run CTS for a specific placement configuration."""
    print(f"\n{'=' * 60}")
    print(f"Running H-Tree CTS for {name.upper()} placement")
    print(f"{'=' * 60}")
    
    placement_json = config['placement_json']
    placement_map = config['placement_map']
    output_dir = config['output_dir']
    
    # Check if placement files exist
    if not os.path.exists(placement_json):
        print(f"ERROR: Placement file {placement_json} not found!")
        return False
    
    if not os.path.exists(placement_map):
        print(f"WARNING: Placement map file {placement_map} not found!")
        print("  (Will skip placement.map update)")
        placement_map = None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Synthesize clock tree
    root, used_buffers = synthesize_clock_tree(
        placement_json,
        design_file,
        fabric_cells,
        threshold=4
    )
    
    if root is None:
        print(f"ERROR: CTS synthesis failed for {name} placement")
        return False
    
    # Save outputs
    save_cts_tree(root, f"{output_dir}/6502_cts_tree.json")
    save_buffer_mapping(used_buffers, f"{output_dir}/6502_cts_buffers.map")
    
    # Update placement.map if it exists
    if placement_map:
        # Create a copy to update (don't modify original)
        placement_map_updated = f"{output_dir}/6502_with_cts.map"
        shutil.copy(placement_map, placement_map_updated)
        update_placement_map(placement_map_updated, used_buffers, fabric_cells)
        print(f"✓ Created updated placement.map: {placement_map_updated}")
    
    # Update placement JSON (create updated version)
    placement_json_updated = f"{output_dir}/6502_placement_with_cts.json"
    shutil.copy(placement_json, placement_json_updated)
    update_placement_json(placement_json_updated, used_buffers, fabric_cells)
    print(f"✓ Created updated placement JSON: {placement_json_updated}")
    
    # Update netlist with clock connections
    netlist_updated = f"{output_dir}/6502_mapped_with_cts.json"
    update_netlist_connections(design_file, root, used_buffers, netlist_updated)
    print(f"✓ Created updated netlist: {netlist_updated}")
    
    print(f"\n✓ {name.upper()} placement CTS completed successfully!")
    print(f"  Output directory: {output_dir}")
    return True

# Run CTS for both placements
print("=" * 60)
print("Running CTS for 6502 design - Both Placements")
print("=" * 60)

results = {}
for name, config in placements.items():
    results[name] = run_cts_for_placement(name, config)

# Summary
print("\n" + "=" * 60)
print("CTS Synthesis Summary")
print("=" * 60)
for name, success in results.items():
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"  {name.upper()}: {status}")

if all(results.values()):
    print("\n✓ All CTS runs completed successfully!")
    sys.exit(0)
else:
    print("\n✗ Some CTS runs failed!")
    sys.exit(1)

