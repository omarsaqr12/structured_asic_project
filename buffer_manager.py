#!/usr/bin/env python3
"""
Buffer Manager for Clock Tree Synthesis (CTS)

Manages buffer and inverter slot assignments for CTS algorithms.
Tracks available vs claimed buffers and updates placement files.
"""

import json
import math
import os
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from parse_fabric import parse_fabric_cells

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Falling back to linear search for buffers.")


def parse_placement_map(map_file_path: str) -> Dict[str, str]:
    """
    Parse placement map file (.map format).
    
    Format: One line per logical instance
    Format: logical_instance_name physical_slot_name
    
    Args:
        map_file_path: Path to .map file
    
    Returns:
        placement_map: Dict mapping logical_name -> physical_slot_name
    """
    placement_map = {}
    
    if not os.path.exists(map_file_path):
        return placement_map
    
    try:
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    logical_name = parts[0]
                    physical_slot_name = parts[1]
                    placement_map[logical_name] = physical_slot_name
    except Exception as e:
        raise ValueError(f"Error parsing placement map file: {e}")
    
    return placement_map


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class BufferManager:
    """
    Manages buffer and inverter slot assignments for CTS.
    
    Tracks available vs claimed buffers, finds nearest buffers,
    and updates placement files.
    """
    
    def __init__(self, fabric_cells_path: str, placement_map_path: Optional[str] = None):
        """
        Initialize BufferManager with fabric database and placement map.
        
        Args:
            fabric_cells_path: Path to fabric_cells.yaml
            placement_map_path: Optional path to placement.map file
        """
        # Parse fabric database
        fabric_db = parse_fabric_cells(fabric_cells_path)
        
        # Buffer and inverter types
        buffer_type = 'sky130_fd_sc_hd__clkbuf_4'  # BUF
        inv_type = 'sky130_fd_sc_hd__clkinv_2'     # INV
        
        # Get all buffer/inverter slots from fabric
        self.available = {
            'BUF': [],
            'INV': []
        }
        
        if buffer_type in fabric_db:
            self.available['BUF'] = [
                {
                    'name': slot['name'],
                    'x': slot['x'],
                    'y': slot['y'],
                    'orient': slot.get('orient', 'N'),
                    'type': 'BUF'
                }
                for slot in fabric_db[buffer_type]
            ]
        
        if inv_type in fabric_db:
            self.available['INV'] = [
                {
                    'name': slot['name'],
                    'x': slot['x'],
                    'y': slot['y'],
                    'orient': slot.get('orient', 'N'),
                    'type': 'INV'
                }
                for slot in fabric_db[inv_type]
            ]
        
        # Parse placement map to get already-placed buffers
        placed_slots = set()
        if placement_map_path:
            placement_map = parse_placement_map(placement_map_path)
            placed_slots = set(placement_map.values())
        
        # Remove already-placed buffers from available
        self.available['BUF'] = [
            s for s in self.available['BUF'] 
            if s['name'] not in placed_slots
        ]
        self.available['INV'] = [
            s for s in self.available['INV'] 
            if s['name'] not in placed_slots
        ]
        
        # Track claimed buffers: {logical_name: {slot, x, y, level, children, ...}}
        self.claimed = {}
        
        # Counter for logical buffer names
        self.buffer_counter = 0
        
        # Build spatial indices for fast lookup
        self._build_spatial_indices()
    
    def _build_spatial_indices(self):
        """Build KD-trees for fast nearest buffer lookup."""
        self.buf_kdtree = None
        self.buf_coords = None
        self.inv_kdtree = None
        self.inv_coords = None
        
        if HAS_SCIPY:
            if self.available['BUF']:
                self.buf_coords = [(b['x'], b['y']) for b in self.available['BUF']]
                self.buf_kdtree = KDTree(self.buf_coords)
            
            if self.available['INV']:
                self.inv_coords = [(b['x'], b['y']) for b in self.available['INV']]
                self.inv_kdtree = KDTree(self.inv_coords)
    
    def _find_nearest_buffer(self, target_x: float, target_y: float,
                            preferred_type: str = 'BUF') -> Optional[Dict[str, Any]]:
        """
        Find nearest available buffer to target position.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            preferred_type: 'BUF' or 'INV' (default: 'BUF')
        
        Returns:
            Nearest buffer dict or None if no buffers available
        """
        # Try preferred type first
        preferred_list = self.available.get(preferred_type, [])
        
        # Fallback to other type if preferred not available
        other_type = 'INV' if preferred_type == 'BUF' else 'BUF'
        other_list = self.available.get(other_type, [])
        
        candidates = preferred_list if preferred_list else other_list
        
        if not candidates:
            return None
        
        # Use KD-tree if available for preferred type
        if HAS_SCIPY and preferred_list:
            kdtree = self.buf_kdtree if preferred_type == 'BUF' else self.inv_kdtree
            
            if kdtree is not None:
                k = min(10, len(preferred_list))
                result = kdtree.query([target_x, target_y], k=k)
                
                # Handle both single result and array results
                if k == 1:
                    distances, indices = result
                    indices = [int(indices)] if not isinstance(indices, (list, tuple)) else [int(i) for i in indices]
                else:
                    distances, indices = result
                    if hasattr(indices, '__iter__') and not isinstance(indices, (str, bytes)):
                        indices = [int(i) for i in indices]
                    else:
                        indices = [int(indices)]
                
                # Return first valid index from preferred list
                for idx in indices:
                    if 0 <= idx < len(preferred_list):
                        return preferred_list[idx]
        
        # Fallback to linear search
        nearest = None
        min_distance = float('inf')
        
        for buffer in candidates:
            distance = calculate_distance(target_x, target_y, buffer['x'], buffer['y'])
            if distance < min_distance:
                min_distance = distance
                nearest = buffer
        
        return nearest
    
    def claim_buffer(self, target_x: float, target_y: float,
                    preferred_type: str = 'BUF',
                    level: int = 0,
                    children: Optional[List[str]] = None) -> Optional[str]:
        """
        Claim nearest available buffer to target position.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            preferred_type: 'BUF' or 'INV' (default: 'BUF')
            level: Tree level for this buffer (default: 0)
            children: List of child buffer logical names (default: None)
        
        Returns:
            Logical buffer name (e.g., 'cts_buffer_0') or None if no buffers available
        
        Raises:
            ValueError: If insufficient buffers available
        """
        # Validate sufficient buffers remain
        total_available = len(self.available.get('BUF', [])) + len(self.available.get('INV', []))
        if total_available == 0:
            raise ValueError("No buffers available for claiming")
        
        # Find nearest buffer
        buffer = self._find_nearest_buffer(target_x, target_y, preferred_type)
        
        if buffer is None:
            raise ValueError(f"No {preferred_type} buffers available")
        
        # Generate logical name
        logical_name = f"cts_buffer_{self.buffer_counter}"
        self.buffer_counter += 1
        
        # Claim buffer
        self.claimed[logical_name] = {
            'slot': buffer['name'],
            'x': buffer['x'],
            'y': buffer['y'],
            'type': buffer['type'],
            'orient': buffer['orient'],
            'level': level,
            'children': children or [],
            'purpose': 'cts'
        }
        
        # Remove from available
        buffer_type = buffer['type']
        self.available[buffer_type] = [
            b for b in self.available[buffer_type] 
            if b['name'] != buffer['name']
        ]
        
        # Rebuild spatial index for this type
        if HAS_SCIPY:
            if buffer_type == 'BUF':
                if self.available['BUF']:
                    self.buf_coords = [(b['x'], b['y']) for b in self.available['BUF']]
                    self.buf_kdtree = KDTree(self.buf_coords)
                else:
                    self.buf_kdtree = None
                    self.buf_coords = None
            else:  # INV
                if self.available['INV']:
                    self.inv_coords = [(b['x'], b['y']) for b in self.available['INV']]
                    self.inv_kdtree = KDTree(self.inv_coords)
                else:
                    self.inv_kdtree = None
                    self.inv_coords = None
        
        return logical_name
    
    def update_placement_map(self, placement_map_path: str, append: bool = True):
        """
        Update placement.map file by appending claimed buffers.
        
        Args:
            placement_map_path: Path to placement.map file
            append: If True, append to existing file; if False, overwrite (default: True)
        """
        mode = 'a' if append else 'w'
        
        with open(placement_map_path, mode) as f:
            for logical_name in sorted(self.claimed.keys()):
                slot_name = self.claimed[logical_name]['slot']
                f.write(f"{logical_name} {slot_name}\n")
        
        print(f"Updated placement.map with {len(self.claimed)} CTS buffers")
    
    def export_claims(self, output_path: str):
        """
        Export claimed buffers to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        claims_dict = {}
        
        for logical_name, claim_data in self.claimed.items():
            claims_dict[logical_name] = {
                'slot': claim_data['slot'],
                'x': claim_data['x'],
                'y': claim_data['y'],
                'type': claim_data['type'],
                'level': claim_data['level'],
                'children': claim_data['children']
            }
        
        with open(output_path, 'w') as f:
            json.dump(claims_dict, f, indent=2)
        
        print(f"Exported {len(claims_dict)} claimed buffers to {output_path}")
    
    def get_available_count(self) -> Dict[str, int]:
        """
        Get count of available buffers by type.
        
        Returns:
            Dict mapping buffer type to count
        """
        return {
            'BUF': len(self.available.get('BUF', [])),
            'INV': len(self.available.get('INV', [])),
            'total': len(self.available.get('BUF', [])) + len(self.available.get('INV', []))
        }
    
    def get_claimed_count(self) -> int:
        """
        Get count of claimed buffers.
        
        Returns:
            Number of claimed buffers
        """
        return len(self.claimed)
    
    def validate_sufficient_buffers(self, required_count: int):
        """
        Validate that sufficient buffers are available.
        
        Args:
            required_count: Number of buffers required
        
        Raises:
            ValueError: If insufficient buffers available
        """
        available_count = self.get_available_count()['total']
        if available_count < required_count:
            raise ValueError(
                f"Insufficient buffers: required {required_count}, "
                f"available {available_count}"
            )


def main():
    """Test BufferManager functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Buffer Manager for Clock Tree Synthesis'
    )
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--placement-map', default=None,
                        help='Path to placement.map file (optional)')
    parser.add_argument('--output-claims', default='claimed_buffers.json',
                        help='Output file for claimed buffers JSON')
    parser.add_argument('--update-map', default=None,
                        help='Path to placement.map file to update (optional)')
    
    args = parser.parse_args()
    
    # Initialize buffer manager
    print("Initializing BufferManager...")
    manager = BufferManager(args.fabric_cells, args.placement_map)
    
    available = manager.get_available_count()
    print(f"Available buffers: {available['BUF']} BUF, {available['INV']} INV, "
          f"{available['total']} total")
    
    # Example: claim a few buffers
    print("\nClaiming test buffers...")
    try:
        buf1 = manager.claim_buffer(100.0, 100.0, preferred_type='BUF', level=0)
        print(f"Claimed: {buf1}")
        
        buf2 = manager.claim_buffer(200.0, 200.0, preferred_type='BUF', level=1, children=[buf1])
        print(f"Claimed: {buf2}")
        
        buf3 = manager.claim_buffer(150.0, 150.0, preferred_type='INV', level=1)
        print(f"Claimed: {buf3}")
        
        print(f"\nClaimed {manager.get_claimed_count()} buffers")
        available_after = manager.get_available_count()
        print(f"Remaining: {available_after['BUF']} BUF, {available_after['INV']} INV, "
              f"{available_after['total']} total")
        
        # Export claims
        manager.export_claims(args.output_claims)
        
        # Update placement map if requested
        if args.update_map:
            manager.update_placement_map(args.update_map)
        
    except ValueError as e:
        print(f"ERROR: {e}")


if __name__ == '__main__':
    main()

