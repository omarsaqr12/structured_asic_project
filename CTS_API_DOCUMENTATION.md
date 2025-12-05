# CTS API Documentation

## Overview

The CTS API (`cts_api.py`) provides a minimal, clean interface for generating Clock Tree Synthesis (CTS) trees. It integrates H-Tree and X-Tree algorithms with BufferManager for buffer claiming and placement map updates.

## Quick Start

```python
from cts_api import generate_cts_tree

# Generate H-Tree
cts_tree, updated_map = generate_cts_tree(
    placement_map_path="build/6502/greedy/6502.map",
    fabric_cells_path="fabric/fabric_cells.yaml",
    design_path="designs/6502_mapped.json",
    tree_type='h'
)
```

## API Reference

### `generate_cts_tree()`

Main function for generating CTS trees.

#### Signature

```python
def generate_cts_tree(
    placement_map_path: str,
    fabric_cells_path: str,
    design_path: str,
    tree_type: str = 'h',
    threshold: int = 4
) -> Tuple[Dict[str, Any], Dict[str, str]]
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `placement_map_path` | `str` | Yes | - | Path to placement map file (`.map` format) |
| `fabric_cells_path` | `str` | Yes | - | Path to `fabric_cells.yaml` file |
| `design_path` | `str` | Yes | - | Path to design mapped JSON file |
| `tree_type` | `str` | No | `'h'` | Tree algorithm: `'h'` for H-Tree, `'x'` for X-Tree |
| `threshold` | `int` | No | `4` | Maximum number of sinks per leaf node |

#### Returns

Returns a tuple of two dictionaries:

1. **`cts_tree_structure`** (`Dict[str, Any]`):
   - Dictionary keyed by buffer logical names (e.g., `"cts_buffer_0"`)
   - Each buffer entry contains:
     - `children`: List of child buffer names
     - `sinks`: List of DFF sink dictionaries
     - `x`, `y`: Buffer coordinates
     - `slot`: Physical slot name
     - `type`: Buffer type ('BUF' or 'INV')
     - `level`: Tree level
   - Also includes `_metadata` key with:
     - `tree_type`: 'H-Tree' or 'X-Tree'
     - `num_sinks`: Total number of sinks
     - `num_buffers`: Total number of buffers used
     - `root_buffer`: Root buffer name (or None if no buffers)

2. **`updated_placement_map`** (`Dict[str, str]`):
   - Dictionary mapping logical names to physical slot names
   - Includes original placement entries plus new CTS buffers

#### Raises

- `ValueError`: Invalid `tree_type`, no DFFs found, insufficient buffers
- `FileNotFoundError`: Missing input files

## Usage Examples

### Example 1: Generate H-Tree

```python
from cts_api import generate_cts_tree

# Generate H-Tree for 6502 design
cts_tree, updated_map = generate_cts_tree(
    placement_map_path="build/6502/greedy/6502.map",
    fabric_cells_path="fabric/fabric_cells.yaml",
    design_path="designs/6502_mapped.json",
    tree_type='h'
)

# Access tree structure
print(f"Tree type: {cts_tree['_metadata']['tree_type']}")
print(f"Number of buffers: {cts_tree['_metadata']['num_buffers']}")
print(f"Number of sinks: {cts_tree['_metadata']['num_sinks']}")

# Iterate through buffers
for buffer_name, buffer_info in cts_tree.items():
    if buffer_name.startswith('_'):
        continue  # Skip metadata
    
    print(f"\nBuffer: {buffer_name}")
    print(f"  Position: ({buffer_info['x']}, {buffer_info['y']})")
    print(f"  Slot: {buffer_info['slot']}")
    print(f"  Children: {buffer_info['children']}")
    print(f"  Direct sinks: {len(buffer_info['sinks'])}")
```

### Example 2: Generate X-Tree

```python
from cts_api import generate_cts_tree

# Generate X-Tree (diagonal partitioning)
cts_tree, updated_map = generate_cts_tree(
    placement_map_path="build/6502/greedy/6502.map",
    fabric_cells_path="fabric/fabric_cells.yaml",
    design_path="designs/6502_mapped.json",
    tree_type='x'  # X-Tree
)

# Get root buffer
root_buffer = cts_tree['_metadata']['root_buffer']
if root_buffer:
    print(f"Root buffer: {root_buffer}")
    root_info = cts_tree[root_buffer]
    print(f"  Position: ({root_info['x']}, {root_info['y']})")
    print(f"  Children: {root_info['children']}")
```

### Example 3: Save Updated Placement Map

```python
from cts_api import generate_cts_tree

cts_tree, updated_map = generate_cts_tree(
    placement_map_path="build/6502/greedy/6502.map",
    fabric_cells_path="fabric/fabric_cells.yaml",
    design_path="designs/6502_mapped.json",
    tree_type='h'
)

# Save updated placement map
with open("build/6502/cts/6502_with_cts.map", 'w') as f:
    for logical_name in sorted(updated_map.keys()):
        f.write(f"{logical_name} {updated_map[logical_name]}\n")

print(f"Updated placement map saved with {len(updated_map)} entries")
```

### Example 4: Error Handling

```python
from cts_api import generate_cts_tree

try:
    cts_tree, updated_map = generate_cts_tree(
        placement_map_path="build/6502/greedy/6502.map",
        fabric_cells_path="fabric/fabric_cells.yaml",
        design_path="designs/6502_mapped.json",
        tree_type='h'
    )
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Error: {e}")
    # Handle specific errors:
    # - Invalid tree_type
    # - No DFFs found
    # - Insufficient buffers
```

## Integration with eco_generator.py

### Basic Integration

```python
# In eco_generator.py
from cts_api import generate_cts_tree

def generate_eco_with_cts(placement_map_path, fabric_cells_path, design_path):
    """Generate ECO with CTS tree."""
    
    # Generate CTS tree
    cts_tree, updated_map = generate_cts_tree(
        placement_map_path=placement_map_path,
        fabric_cells_path=fabric_cells_path,
        design_path=design_path,
        tree_type='h'  # or 'x' for X-Tree
    )
    
    # Use updated placement map for ECO generation
    # updated_map now includes CTS buffers
    
    # Access tree structure if needed
    root_buffer = cts_tree['_metadata']['root_buffer']
    
    return updated_map, cts_tree
```

### Advanced Integration

```python
from cts_api import generate_cts_tree

def generate_eco_with_optional_cts(placement_map_path, fabric_cells_path, 
                                   design_path, enable_cts=True, tree_type='h'):
    """Generate ECO with optional CTS."""
    
    if enable_cts:
        try:
            cts_tree, updated_map = generate_cts_tree(
                placement_map_path=placement_map_path,
                fabric_cells_path=fabric_cells_path,
                design_path=design_path,
                tree_type=tree_type
            )
            
            # Log CTS information
            print(f"CTS generated: {cts_tree['_metadata']['num_buffers']} buffers, "
                  f"{cts_tree['_metadata']['num_sinks']} sinks")
            
            return updated_map, cts_tree
            
        except ValueError as e:
            print(f"Warning: CTS generation failed: {e}")
            print("Continuing without CTS...")
            # Fall back to original placement map
            from buffer_manager import parse_placement_map
            return parse_placement_map(placement_map_path), None
    else:
        # No CTS requested
        from buffer_manager import parse_placement_map
        return parse_placement_map(placement_map_path), None
```

## Command-Line Usage

The API also includes a command-line interface:

```bash
# Generate H-Tree
python cts_api.py \
    --placement-map build/6502/greedy/6502.map \
    --fabric-cells fabric/fabric_cells.yaml \
    --design designs/6502_mapped.json \
    --tree-type h \
    --output-tree cts_tree.json \
    --output-map placement_with_cts.map

# Generate X-Tree
python cts_api.py \
    --placement-map build/6502/greedy/6502.map \
    --fabric-cells fabric/fabric_cells.yaml \
    --design designs/6502_mapped.json \
    --tree-type x \
    --threshold 4
```

### Command-Line Options

- `--placement-map`: Path to placement.map file (required)
- `--fabric-cells`: Path to fabric_cells.yaml (default: `fabric/fabric_cells.yaml`)
- `--design`: Path to design mapped JSON file (required)
- `--tree-type`: Tree type: `h` for H-Tree, `x` for X-Tree (default: `h`)
- `--threshold`: Maximum sinks per leaf node (default: `4`)
- `--output-tree`: Output file for CTS tree JSON (default: `cts_tree.json`)
- `--output-map`: Output file for updated placement map (default: `placement_with_cts.map`)

## Tree Structure Format

### Buffer Entry Format

```json
{
  "cts_buffer_0": {
    "children": ["cts_buffer_1", "cts_buffer_2"],
    "sinks": [
      {
        "dff_name": "cpu.U_reg.1",
        "dff_slot": "T0Y0__R1_DFBBP_0",
        "x": 100.0,
        "y": 50.0
      }
    ],
    "x": 125.5,
    "y": 87.3,
    "slot": "T18Y4__R2_INV_0",
    "type": "BUF",
    "level": 0
  }
}
```

### Metadata Format

```json
{
  "_metadata": {
    "tree_type": "H-Tree",
    "num_sinks": 143,
    "num_buffers": 45,
    "root_buffer": "cts_buffer_0"
  }
}
```

## Error Handling

### Common Errors and Solutions

1. **`FileNotFoundError: Placement map file not found`**
   - **Solution**: Check that the placement map path is correct
   - **Example**: Ensure `build/6502/greedy/6502.map` exists

2. **`ValueError: Invalid tree_type 'x'. Must be 'h' (H-Tree) or 'x' (X-Tree)`**
   - **Solution**: Use `'h'` or `'x'` (lowercase) for tree_type
   - **Example**: `tree_type='h'` or `tree_type='x'`

3. **`ValueError: No DFF instances found in design`**
   - **Solution**: Ensure design file contains DFF cells
   - **Check**: Verify design file is correct and contains `sky130_fd_sc_hd__dfbbp_1` cells

4. **`ValueError: No DFFs found in placement map`**
   - **Solution**: Ensure all DFFs are placed before running CTS
   - **Check**: Verify placement map includes DFF entries

5. **`ValueError: Insufficient buffers: required at least X, available Y`**
   - **Solution**: Free up buffer slots or reduce threshold
   - **Options**: 
     - Reduce `threshold` parameter (allows more sinks per buffer)
     - Check if other cells are using buffer slots unnecessarily

## Dependencies

The CTS API requires:

- `buffer_manager.py` - Buffer claiming and management
- `parse_design.py` - Design file parsing
- `parse_fabric.py` - Fabric database parsing
- Python 3.6+
- Optional: `scipy` for faster buffer lookup (falls back to linear search if not available)

## Best Practices

1. **Always handle errors**: Wrap API calls in try-except blocks
2. **Validate inputs**: Check file paths before calling the API
3. **Save results**: Always save the updated placement map after CTS generation
4. **Check metadata**: Use `_metadata` to verify tree generation succeeded
5. **Choose tree type**: 
   - Use H-Tree (`'h'`) for standard clock distribution
   - Use X-Tree (`'x'`) for diagonal partitioning comparison

## Troubleshooting

### Issue: API returns empty tree structure

**Possible causes:**
- No DFFs in placement map
- All buffers already used
- Threshold too high

**Solution:**
```python
# Check metadata
if cts_tree['_metadata']['num_buffers'] == 0:
    print("Warning: No buffers used - check if DFFs are placed")
```

### Issue: Insufficient buffers error

**Solution:**
```python
# Reduce threshold to use fewer buffers
cts_tree, updated_map = generate_cts_tree(
    placement_map_path=placement_map_path,
    fabric_cells_path=fabric_cells_path,
    design_path=design_path,
    tree_type='h',
    threshold=8  # Increase from default 4
)
```

## See Also

- `buffer_manager.py` - Buffer management documentation
- `cts_htree.py` - H-Tree algorithm implementation
- `cts_xtree.py` - X-Tree algorithm implementation
- `run_6502_cts_with_manager.py` - Example usage script

