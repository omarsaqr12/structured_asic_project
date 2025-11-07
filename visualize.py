#!/usr/bin/env python3
"""
Visualization script for Phase 1.
Generates fabric layout plots showing die, core, pins, and fabric slots.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Tuple
from parse_fabric import parse_fabric_cells, parse_pins


# Color mapping for different cell types
CELL_COLORS = {
    'sky130_fd_sc_hd__nand2_2': '#FFD700',      # Gold/Yellow
    'sky130_fd_sc_hd__or2_2': '#FFA500',        # Orange
    'sky130_fd_sc_hd__clkinv_2': '#87CEEB',     # Sky Blue
    'sky130_fd_sc_hd__clkbuf_4': '#4169E1',     # Royal Blue
    'sky130_fd_sc_hd__dfbbp_1': '#FF69B4',      # Hot Pink
    'sky130_fd_sc_hd__conb_1': '#32CD32',       # Lime Green
    'sky130_fd_sc_hd__tapvpwrvgnd_1': '#808080', # Gray
    'sky130_fd_sc_hd__decap_3': '#DDA0DD',      # Plum
    'sky130_fd_sc_hd__decap_4': '#DDA0DD',      # Plum
    'unknown': '#000000'                        # Black
}

# Cell dimensions from fabric.yaml (site dimensions)
SITE_WIDTH = 0.46  # um (from fabric.yaml)
SITE_HEIGHT = 2.72  # um (from fabric.yaml)

# Cell widths in sites (from fabric.yaml cell_definitions)
CELL_WIDTH_SITES = {
    'sky130_fd_sc_hd__nand2_2': 5,
    'sky130_fd_sc_hd__or2_2': 5,
    'sky130_fd_sc_hd__clkinv_2': 4,
    'sky130_fd_sc_hd__clkbuf_4': 6,
    'sky130_fd_sc_hd__dfbbp_1': 26,
    'sky130_fd_sc_hd__conb_1': 3,
    'sky130_fd_sc_hd__tapvpwrvgnd_1': 1,
    'sky130_fd_sc_hd__decap_3': 3,
    'sky130_fd_sc_hd__decap_4': 4,
}

def get_cell_dimensions(cell_type: str) -> Tuple[float, float]:
    """Get cell width and height in microns."""
    width_sites = CELL_WIDTH_SITES.get(cell_type, 5)  # Default to 5 sites
    width_um = width_sites * SITE_WIDTH
    height_um = SITE_HEIGHT
    return width_um, height_um


def plot_fabric_layout(fabric_cells_path: str, pins_path: str, output_path: str):
    """
    Generate the ground-truth plot of the die, core, pins, and fabric slots.
    """
    # Parse fabric and pins
    fabric_db = parse_fabric_cells(fabric_cells_path)
    pins_db = parse_pins(pins_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    # Get die and core dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    
    # Calculate core position (centered in die with margin)
    core_x = core_margin
    core_y = core_margin
    
    # Draw die boundary
    die_rect = patches.Rectangle(
        (0, 0), die_width, die_height,
        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
    )
    ax.add_patch(die_rect)
    
    # Draw core boundary
    core_rect = patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
    )
    ax.add_patch(core_rect)
    
    # Draw all fabric slots as color-coded rectangles
    for cell_type, slots in fabric_db.items():
        color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
        width_um, height_um = get_cell_dimensions(cell_type)
        
        for slot in slots:
            x = slot['x']
            y = slot['y']
            
            # Draw semi-transparent rectangle for each slot
            # Center the rectangle at the slot coordinates
            slot_rect = patches.Rectangle(
                (x - width_um/2, y - height_um/2),
                width_um, height_um,
                linewidth=0.1, edgecolor='black', facecolor=color, alpha=0.5
            )
            ax.add_patch(slot_rect)
    
    # Draw pins
    for pin in pins_db['pins']:
        x = pin['x_um']
        y = pin['y_um']
        direction = pin['direction']
        
        # Color pins by direction
        pin_color = 'green' if direction == 'INPUT' else 'red'
        
        # Draw pin as a small circle
        pin_circle = plt.Circle((x, y), radius=2, color=pin_color, zorder=10)
        ax.add_patch(pin_circle)
        
        # Optionally label pins (can be commented out if too cluttered)
        # ax.text(x, y, pin['name'], fontsize=4, ha='center', va='center')
    
    # Set axis properties
    ax.set_xlim(-50, die_width + 50)
    ax.set_ylim(-50, die_height + 50)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    ax.set_title('Fabric Layout: Die, Core, Pins, and Fabric Slots', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend for cell types
    legend_elements = []
    for cell_type, color in CELL_COLORS.items():
        if cell_type != 'unknown' and any(cell_type in ct for ct in fabric_db.keys()):
            # Extract short name for legend
            short_name = cell_type.replace('sky130_fd_sc_hd__', '')
            legend_elements.append(patches.Patch(facecolor=color, label=short_name, alpha=0.6))
    
    # Add pin legend
    legend_elements.append(patches.Patch(facecolor='green', label='Input Pin', alpha=1.0))
    legend_elements.append(patches.Patch(facecolor='red', label='Output Pin', alpha=1.0))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Fabric layout saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize fabric layout')
    parser.add_argument('command', choices=['init'], help='Visualization command')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                        help='Path to pins.yaml')
    parser.add_argument('--output', default='fabric_layout.png',
                        help='Output PNG file path')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        plot_fabric_layout(args.fabric_cells, args.pins, args.output)


if __name__ == '__main__':
    main()

