#!/usr/bin/env python3
"""
Example usage of BufferManager for Clock Tree Synthesis

This demonstrates how to use BufferManager to claim buffers
during CTS tree construction.
"""

from buffer_manager import BufferManager
import json


def example_basic_usage():
    """Basic example of using BufferManager."""
    print("=" * 60)
    print("Example 1: Basic BufferManager Usage")
    print("=" * 60)
    
    # Initialize BufferManager
    fabric_cells = "fabric/fabric_cells.yaml"
    placement_map = "build/6502/greedy/6502.map"  # Optional
    
    manager = BufferManager(fabric_cells, placement_map)
    
    # Check available buffers
    available = manager.get_available_count()
    print(f"Available buffers: {available['BUF']} BUF, {available['INV']} INV")
    
    # Claim some buffers
    print("\nClaiming buffers...")
    buf1 = manager.claim_buffer(100.0, 100.0, preferred_type='BUF', level=0)
    print(f"Claimed buffer 1: {buf1} at (100, 100)")
    
    buf2 = manager.claim_buffer(200.0, 200.0, preferred_type='BUF', level=1, children=[buf1])
    print(f"Claimed buffer 2: {buf2} at (200, 200) with child {buf1}")
    
    buf3 = manager.claim_buffer(150.0, 150.0, preferred_type='INV', level=1)
    print(f"Claimed buffer 3: {buf3} at (150, 150)")
    
    # Export claims
    manager.export_claims("example_claimed_buffers.json")
    
    # Update placement map
    manager.update_placement_map("example_placement_with_cts.map", append=True)
    
    print(f"\nTotal claimed: {manager.get_claimed_count()}")
    remaining = manager.get_available_count()
    print(f"Remaining: {remaining['BUF']} BUF, {remaining['INV']} INV")


def example_integration_with_cts():
    """Example of integrating BufferManager with CTS tree building."""
    print("\n" + "=" * 60)
    print("Example 2: Integration with CTS Tree Building")
    print("=" * 60)
    
    from buffer_manager import BufferManager
    
    # Initialize BufferManager
    fabric_cells = "fabric/fabric_cells.yaml"
    placement_map = "build/6502/greedy/6502.map"
    
    manager = BufferManager(fabric_cells, placement_map)
    
    # Simulate tree building process
    print("Building clock tree structure...")
    
    # Root level buffer
    root_buf = manager.claim_buffer(
        target_x=250.0,
        target_y=250.0,
        preferred_type='BUF',
        level=0
    )
    print(f"Root buffer: {root_buf}")
    
    # Level 1 buffers (children of root)
    level1_bufs = []
    for i, (x, y) in enumerate([(100, 100), (400, 400)]):
        buf = manager.claim_buffer(
            target_x=x,
            target_y=y,
            preferred_type='BUF',
            level=1,
            children=[]
        )
        level1_bufs.append(buf)
        print(f"Level 1 buffer {i}: {buf} at ({x}, {y})")
    
    # Update root buffer with children
    manager.claimed[root_buf]['children'] = level1_bufs
    
    # Level 2 buffers
    level2_bufs = []
    for i, (x, y) in enumerate([(50, 50), (150, 150), (350, 350), (450, 450)]):
        buf = manager.claim_buffer(
            target_x=x,
            target_y=y,
            preferred_type='BUF',
            level=2,
            children=[]
        )
        level2_bufs.append(buf)
        print(f"Level 2 buffer {i}: {buf} at ({x}, {y})")
    
    # Update level 1 buffers with children
    for i, buf in enumerate(level1_bufs):
        manager.claimed[buf]['children'] = level2_bufs[i*2:(i+1)*2]
    
    # Export final tree structure
    manager.export_claims("example_cts_tree_buffers.json")
    
    print(f"\nTree structure:")
    print(f"  Root: {root_buf}")
    print(f"  Level 1: {len(level1_bufs)} buffers")
    print(f"  Level 2: {len(level2_bufs)} buffers")
    print(f"  Total: {manager.get_claimed_count()} buffers")


def example_error_handling():
    """Example of error handling with BufferManager."""
    print("\n" + "=" * 60)
    print("Example 3: Error Handling")
    print("=" * 60)
    
    from buffer_manager import BufferManager
    
    fabric_cells = "fabric/fabric_cells.yaml"
    manager = BufferManager(fabric_cells)
    
    # Validate sufficient buffers
    try:
        manager.validate_sufficient_buffers(required_count=1000)
        print("✓ Sufficient buffers available")
    except ValueError as e:
        print(f"✗ {e}")
    
    # Try to claim buffer when none available (edge case)
    available = manager.get_available_count()
    print(f"\nAvailable: {available['total']} buffers")
    
    if available['total'] > 0:
        # Claim all available buffers
        print("Claiming all available buffers...")
        for i in range(available['total']):
            try:
                buf = manager.claim_buffer(100.0 + i, 100.0 + i)
                if i < 5:  # Show first 5
                    print(f"  Claimed: {buf}")
            except ValueError as e:
                print(f"  Error: {e}")
                break
        
        # Try to claim one more (should fail)
        try:
            buf = manager.claim_buffer(1000.0, 1000.0)
            print(f"Unexpectedly claimed: {buf}")
        except ValueError as e:
            print(f"✓ Correctly raised error: {e}")


if __name__ == '__main__':
    try:
        example_basic_usage()
        example_integration_with_cts()
        example_error_handling()
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        print("Make sure fabric_cells.yaml and placement.map files exist")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

