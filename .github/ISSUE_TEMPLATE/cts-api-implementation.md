# Implement Minimal CTS API for eco_generator.py

## Description
Create a clean, minimal API that Member B can call from `eco_generator.py` to generate CTS trees. This API wraps the existing H-Tree and X-Tree implementations with BufferManager integration.

## Objectives
- [x] Create `cts_api.py` with `generate_cts_tree()` function
- [x] Support both H-Tree ('h') and X-Tree ('x') tree types
- [x] Integrate with BufferManager for buffer claiming
- [x] Parse placement.map and fabric_cells.yaml
- [x] Identify DFFs from design file
- [x] Return tree structure in required format: `{buffer_name: {children, sinks, x, y, ...}}`
- [x] Return updated placement map with CTS buffers
- [x] Implement comprehensive error handling
- [ ] Test API integration with eco_generator.py
- [ ] Document API usage for Member B

## Technical Details

### API Function Signature
```python
def generate_cts_tree(placement_map_path: str,
                     fabric_cells_path: str,
                     design_path: str,
                     tree_type: str = 'h',
                     threshold: int = 4) -> Tuple[Dict[str, Any], Dict[str, str]]
```

### Parameters
- `placement_map_path`: Path to [design].map file
- `fabric_cells_path`: Path to fabric_cells.yaml
- `design_path`: Path to design mapped JSON file (needed to identify DFFs)
- `tree_type`: 'h' for H-Tree, 'x' for X-Tree (default: 'h')
- `threshold`: Maximum number of sinks per leaf node (default: 4)

### Returns
- `cts_tree_structure`: Dict keyed by buffer_name
  - Format: `{buffer_name: {children: [...], sinks: [...], x, y, slot, type, level, ...}}`
  - Includes `_metadata` key with tree info
- `updated_placement_map`: Dict mapping logical_name -> slot_name
  - Includes both original placement and new CTS buffers

### Error Handling
- `ValueError`: Invalid tree_type, no DFFs found, insufficient buffers
- `FileNotFoundError`: Missing input files
- Clear error messages for debugging

## Implementation Files
- `cts_api.py` - Minimal API implementation

## Dependencies
- `buffer_manager.py` - For buffer claiming
- `parse_design.py` - For DFF identification
- `parse_fabric.py` - For fabric database parsing

## Acceptance Criteria
- [x] API function accepts required parameters
- [x] Supports both H-Tree and X-Tree algorithms
- [x] Uses BufferManager for all buffer operations
- [x] Returns tree structure in correct format
- [x] Returns updated placement map with CTS buffers
- [x] Handles all error cases gracefully
- [x] Includes command-line interface for testing
- [ ] Integration tested with eco_generator.py

## Usage Example
```python
from cts_api import generate_cts_tree

cts_tree, updated_map = generate_cts_tree(
    placement_map_path="build/6502/greedy/6502.map",
    fabric_cells_path="fabric/fabric_cells.yaml",
    design_path="designs/6502_mapped.json",
    tree_type='h'
)
```

## Related
- Related to BufferManager: `buffer_manager.py`
- Related to H-Tree: `cts_htree.py`
- Related to X-Tree: `cts_xtree.py`
- Integration target: `eco_generator.py`

## Labels
- `enhancement`
- `api`
- `cts`
- `integration`

