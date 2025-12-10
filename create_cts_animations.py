#!/usr/bin/env python3
"""
Wrapper script to create both X-Tree and H-Tree animations.
"""

import sys
import subprocess
import os

def main():
    design = '6502'
    placement_path = f'build/{design}/{design}_placement.json'
    
    print("=" * 60)
    print(f"Creating CTS Tree Animations for {design}")
    print("=" * 60)
    
    # Create X-Tree animation
    print("\n[1/2] Creating X-Tree animation...")
    print("-" * 60)
    result_x = subprocess.run([
        sys.executable, 'animate_cts_tree.py',
        '--design', design,
        '--tree-type', 'x',
        '--placement', placement_path,
        '--output', f'build/{design}/{design}_cts_xtree_animation.mp4'
    ], capture_output=False, text=True)
    
    if result_x.returncode != 0:
        print(f"ERROR: X-Tree animation failed with return code {result_x.returncode}")
        return 1
    
    # Create H-Tree animation
    print("\n[2/2] Creating H-Tree animation...")
    print("-" * 60)
    result_h = subprocess.run([
        sys.executable, 'animate_cts_tree.py',
        '--design', design,
        '--tree-type', 'h',
        '--placement', placement_path,
        '--output', f'build/{design}/{design}_cts_htree_animation.mp4'
    ], capture_output=False, text=True)
    
    if result_h.returncode != 0:
        print(f"ERROR: H-Tree animation failed with return code {result_h.returncode}")
        return 1
    
    print("\n" + "=" * 60)
    print("Both animations created successfully!")
    print("=" * 60)
    print(f"\nX-Tree: build/{design}/{design}_cts_xtree_animation.mp4")
    print(f"H-Tree: build/{design}/{design}_cts_htree_animation.mp4")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
