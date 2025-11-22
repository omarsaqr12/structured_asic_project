#!/usr/bin/env python3
"""
Visualization script for placement results.
Generates heatmaps showing placement density and quality metrics overlaid on fabric layout.
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Any, List
from parse_fabric import parse_fabric_cells, parse_pins


# Color mapping for different cell types (same as original)
CELL_COLORS = {
    'sky130_fd_sc_hd__nand2_2': '#FFD700',
    'sky130_fd_sc_hd__or2_2': '#FFA500',
    'sky130_fd_sc_hd__clkinv_2': '#87CEEB',
    'sky130_fd_sc_hd__clkbuf_4': '#4169E1',
    'sky130_fd_sc_hd__dfbbp_1': '#FF69B4',
    'sky130_fd_sc_hd__conb_1': '#32CD32',
    'sky130_fd_sc_hd__tapvpwrvgnd_1': '#808080',
    'sky130_fd_sc_hd__decap_3': '#DDA0DD',
    'sky130_fd_sc_hd__decap_4': '#DDA0DD',
    'unknown': '#000000'
}

SITE_WIDTH = 0.46
SITE_HEIGHT = 2.72

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
    width_sites = CELL_WIDTH_SITES.get(cell_type, 5)
    width_um = width_sites * SITE_WIDTH
    height_um = SITE_HEIGHT
    return width_um, height_um


def load_placement(placement_path: str) -> Dict[str, Dict[str, Any]]:
    """Load placement results from JSON file."""
    with open(placement_path, 'r') as f:
        return json.load(f)


def create_density_heatmap(placement: Dict[str, Dict[str, Any]],
                           die_width: float,
                           die_height: float,
                           resolution: float = 5.0,
                           sigma: float = 10.0) -> Tuple[np.ndarray, float, float]:
    """
    Create a 2D density heatmap from placement data.
    
    Args:
        placement: Placement dictionary
        die_width: Die width in microns
        die_height: Die height in microns
        resolution: Grid resolution in microns (smaller = finer)
        sigma: Gaussian smoothing parameter (higher = smoother)
    
    Returns:
        heatmap: 2D numpy array of density values
        grid_width: Width of each grid cell
        grid_height: Height of each grid cell
    """
    # Create grid
    nx = int(np.ceil(die_width / resolution))
    ny = int(np.ceil(die_height / resolution))
    
    density_grid = np.zeros((ny, nx))
    
    # Accumulate cell placements
    for cell_name, cell_placement in placement.items():
        x = cell_placement['x']
        y = cell_placement['y']
        
        # Convert to grid indices
        grid_x = int(x / resolution)
        grid_y = int(y / resolution)
        
        if 0 <= grid_x < nx and 0 <= grid_y < ny:
            density_grid[grid_y, grid_x] += 1
    
    # Apply Gaussian smoothing for better visualization
    if sigma > 0:
        density_grid = gaussian_filter(density_grid, sigma=sigma)
    
    return density_grid, resolution, resolution


def plot_placement_heatmap(fabric_cells_path: str,
                           pins_path: str,
                           placement_path: str,
                           output_path: str,
                           show_slots: bool = True,
                           show_pins: bool = True,
                           heatmap_alpha: float = 0.6):
    """
    Generate placement heatmap overlaid on fabric layout.
    
    Args:
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        placement_path: Path to placement.json
        output_path: Output image path
        show_slots: Whether to show fabric slots underneath
        show_pins: Whether to show I/O pins
        heatmap_alpha: Transparency of heatmap (0=transparent, 1=opaque)
    """
    # Parse data
    fabric_db = parse_fabric_cells(fabric_cells_path)
    pins_db = parse_pins(pins_path)
    placement = load_placement(placement_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    # Get dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    
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
    
    # Optionally draw fabric slots
    if show_slots:
        for cell_type, slots in fabric_db.items():
            color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
            width_um, height_um = get_cell_dimensions(cell_type)
            
            for slot in slots:
                x = slot['x']
                y = slot['y']
                
                slot_rect = patches.Rectangle(
                    (x, y), width_um, height_um,
                    linewidth=0.1, edgecolor='gray', facecolor=color, alpha=0.2
                )
                ax.add_patch(slot_rect)
    
    # Create and overlay density heatmap
    density_grid, grid_width, grid_height = create_density_heatmap(
        placement, die_width, die_height, resolution=5.0, sigma=3.0
    )
    
    # Create custom colormap (white to red)
    colors = ['white', 'yellow', 'orange', 'red', 'darkred']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('density', colors, N=n_bins)
    
    # Plot heatmap
    extent = [0, die_width, 0, die_height]
    im = ax.imshow(
        density_grid,
        extent=extent,
        origin='lower',
        cmap=cmap,
        alpha=heatmap_alpha,
        interpolation='bilinear'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Placement Density (cells)', rotation=270, labelpad=20, fontsize=12)
    
    # Optionally draw pins
    if show_pins:
        for pin in pins_db['pins']:
            x = pin['x_um']
            y = pin['y_um']
            direction = pin['direction']
            
            pin_color = 'green' if direction == 'INPUT' else 'red'
            pin_circle = plt.Circle((x, y), radius=2, color=pin_color, zorder=10,
                                   edgecolor='black', linewidth=0.5)
            ax.add_patch(pin_circle)
    
    # Set axis properties
    ax.set_xlim(-50, die_width + 50)
    ax.set_ylim(-50, die_height + 50)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    ax.set_title(f'Placement Density Heatmap ({len(placement)} cells placed)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Placement heatmap saved to {output_path}")
    plt.close()


def plot_placement_detailed(fabric_cells_path: str,
                            pins_path: str,
                            placement_path: str,
                            output_path: str,
                            design_path: str = None):
    """
    Generate detailed placement visualization with actual cell rectangles colored by type.
    
    Args:
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        placement_path: Path to placement.json
        output_path: Output image path
        design_path: Optional path to design JSON to get cell types
    """
    # Parse data
    fabric_db = parse_fabric_cells(fabric_cells_path)
    pins_db = parse_pins(pins_path)
    placement = load_placement(placement_path)
    
    # Load cell types from design if available
    cell_types = {}
    if design_path:
        with open(design_path, 'r') as f:
            design_data = json.load(f)
        
        # Extract cell types from netlist
        modules = design_data.get('modules', {})
        for mod_name, mod_data in modules.items():
            cells = mod_data.get('cells', {})
            for cell_name, cell_data in cells.items():
                cell_types[cell_name] = cell_data.get('type', 'unknown')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    # Get dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    
    core_x = core_margin
    core_y = core_margin
    
    # Draw die boundary
    die_rect = patches.Rectangle(
        (0, 0), die_width, die_height,
        linewidth=2, edgecolor='black', facecolor='white', alpha=1.0
    )
    ax.add_patch(die_rect)
    
    # Draw core boundary
    core_rect = patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
    )
    ax.add_patch(core_rect)
    
    # Draw placed cells as colored rectangles
    for cell_name, cell_placement in placement.items():
        x = cell_placement['x']
        y = cell_placement['y']
        
        # Get cell type
        cell_type = cell_types.get(cell_name, 'unknown')
        color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
        width_um, height_um = get_cell_dimensions(cell_type)
        
        # Draw cell
        cell_rect = patches.Rectangle(
            (x, y), width_um, height_um,
            linewidth=0.3, edgecolor='black', facecolor=color, alpha=0.7
        )
        ax.add_patch(cell_rect)
    
    # Draw pins
    for pin in pins_db['pins']:
        x = pin['x_um']
        y = pin['y_um']
        direction = pin['direction']
        
        pin_color = 'green' if direction == 'INPUT' else 'red'
        pin_circle = plt.Circle((x, y), radius=2, color=pin_color, zorder=10,
                               edgecolor='black', linewidth=0.5)
        ax.add_patch(pin_circle)
    
    # Set axis properties
    ax.set_xlim(-50, die_width + 50)
    ax.set_ylim(-50, die_height + 50)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    ax.set_title(f'Detailed Placement View ({len(placement)} cells)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = []
    used_types = set(cell_types.values()) if cell_types else set()
    for cell_type, color in CELL_COLORS.items():
        if cell_type != 'unknown' and cell_type in used_types:
            short_name = cell_type.replace('sky130_fd_sc_hd__', '')
            legend_elements.append(patches.Patch(facecolor=color, label=short_name, alpha=0.7))
    
    legend_elements.append(patches.Patch(facecolor='green', label='Input Pin', alpha=1.0))
    legend_elements.append(patches.Patch(facecolor='red', label='Output Pin', alpha=1.0))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed placement saved to {output_path}")
    plt.close()


def plot_layered_comparison(fabric_cells_path: str,
                            pins_path: str,
                            placement_path: str,
                            output_path: str):
    """
    Generate a 3-panel comparison: fabric only, heatmap only, and overlay.
    
    Args:
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        placement_path: Path to placement.json
        output_path: Output image path
    """
    # Parse data
    fabric_db = parse_fabric_cells(fabric_cells_path)
    pins_db = parse_pins(pins_path)
    placement = load_placement(placement_path)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Get dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    
    core_x = core_margin
    core_y = core_margin
    
    # Create density heatmap once
    density_grid, _, _ = create_density_heatmap(
        placement, die_width, die_height, resolution=5.0, sigma=3.0
    )
    
    colors = ['white', 'yellow', 'orange', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('density', colors, N=100)
    extent = [0, die_width, 0, die_height]
    
    for idx, ax in enumerate(axes):
        # Draw die and core for all
        die_rect = patches.Rectangle(
            (0, 0), die_width, die_height,
            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
        )
        ax.add_patch(die_rect)
        
        core_rect = patches.Rectangle(
            (core_x, core_y), core_width, core_height,
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
        )
        ax.add_patch(core_rect)
        
        if idx == 0:  # Fabric only
            for cell_type, slots in fabric_db.items():
                color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
                width_um, height_um = get_cell_dimensions(cell_type)
                
                for slot in slots:
                    slot_rect = patches.Rectangle(
                        (slot['x'], slot['y']), width_um, height_um,
                        linewidth=0.1, edgecolor='black', facecolor=color, alpha=0.5
                    )
                    ax.add_patch(slot_rect)
            
            ax.set_title('Fabric Layout (Available Slots)', fontsize=12, fontweight='bold')
        
        elif idx == 1:  # Heatmap only
            im = ax.imshow(
                density_grid, extent=extent, origin='lower',
                cmap=cmap, alpha=0.9, interpolation='bilinear'
            )
            ax.set_title('Placement Density Heatmap', fontsize=12, fontweight='bold')
        
        else:  # Overlay
            for cell_type, slots in fabric_db.items():
                color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
                width_um, height_um = get_cell_dimensions(cell_type)
                
                for slot in slots:
                    slot_rect = patches.Rectangle(
                        (slot['x'], slot['y']), width_um, height_um,
                        linewidth=0.1, edgecolor='gray', facecolor=color, alpha=0.2
                    )
                    ax.add_patch(slot_rect)
            
            im = ax.imshow(
                density_grid, extent=extent, origin='lower',
                cmap=cmap, alpha=0.6, interpolation='bilinear'
            )
            ax.set_title('Overlay: Fabric + Placement Heatmap', fontsize=12, fontweight='bold')
        
        # Draw pins
        for pin in pins_db['pins']:
            pin_color = 'green' if pin['direction'] == 'INPUT' else 'red'
            pin_circle = plt.Circle(
                (pin['x_um'], pin['y_um']), radius=2,
                color=pin_color, zorder=10, edgecolor='black', linewidth=0.5
            )
            ax.add_patch(pin_circle)
        
        ax.set_xlim(-50, die_width + 50)
        ax.set_ylim(-50, die_height + 50)
        ax.set_aspect('equal')
        ax.set_xlabel('X (microns)', fontsize=10)
        ax.set_ylabel('Y (microns)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Layered comparison saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize placement results')
    parser.add_argument('command', default='heatmap',
                       choices=['heatmap', 'detailed', 'comparison'],
                       help='Visualization type')
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                        help='Path to pins.yaml')
    parser.add_argument('--placement', default='placement.json',
                        help='Path to placement.json')
    parser.add_argument('--design', default='designs/6502_mapped.json',
                        help='Path to design JSON (for detailed view)')
    parser.add_argument('--output', default='placement_heatmap.png',
                        help='Output PNG file path')
    parser.add_argument('--no-slots', default=False, action='store_true',
                        help='Hide fabric slots in heatmap view')
    parser.add_argument('--no-pins', default=False, action='store_true',
                        help='Hide I/O pins')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Heatmap transparency (0-1)')
    
    args = parser.parse_args()
    
    if args.command == 'heatmap':
        plot_placement_heatmap(
            args.fabric_cells, args.pins, args.placement, args.output,
            show_slots=not args.no_slots,
            show_pins=not args.no_pins,
            heatmap_alpha=args.alpha
        )
    elif args.command == 'detailed':
        plot_placement_detailed(
            args.fabric_cells, args.pins, args.placement, output_path='placement_detailed.png',
            design_path=args.design
        )
    elif args.command == 'comparison':
        plot_layered_comparison(
            args.fabric_cells, args.pins, args.placement, output_path='comparison_detailed.png'
        )


if __name__ == '__main__':
    main()