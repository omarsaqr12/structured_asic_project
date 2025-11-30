# Pull Request: Implement Minimal CTS API for eco_generator.py

## Description
This PR implements a minimal, clean API (`cts_api.py`) that provides a simple interface for generating CTS trees. This API can be called by `eco_generator.py` to integrate CTS functionality.

## Related Issue
Closes #<issue_number>

## Changes Made
- ✅ Created `cts_api.py` with `generate_cts_tree()` function
  - Supports both H-Tree ('h') and X-Tree ('x') algorithms
  - Integrates with BufferManager for buffer claiming
  - Parses placement.map and fabric_cells.yaml
  - Identifies DFFs from design file
  - Returns tree structure in required format
  - Returns updated placement map with CTS buffers
  - Comprehensive error handling

## Implementation Details

### API Function
```python
def generate_cts_tree(placement_map_path: str,
                     fabric_cells_path: str,
                     design_path: str,
                     tree_type: str = 'h',
                     threshold: int = 4) -> Tuple[Dict[str, Any], Dict[str, str]]
```

### Return Format
- **cts_tree_structure**: Dict keyed by buffer_name
  - Format: `{buffer_name: {children: [...], sinks: [...], x, y, slot, type, level, ...}}`
  - Includes `_metadata` with tree information
- **updated_placement_map**: Dict mapping logical_name -> slot_name
  - Original placement entries preserved
  - CTS buffers appended

### Features
- Input validation (tree_type, file existence)
- DFF identification from design file
- Buffer availability validation
- Error handling with clear messages
- Command-line interface for testing

## Testing
- [x] API function compiles without errors
- [x] Function signature matches specification
- [x] Error handling works correctly
- [x] Command-line interface functional
- [ ] Integration tested with eco_generator.py
- [ ] Tested with both H-Tree and X-Tree

## Usage Example
```python
from cts_api import generate_cts_tree

# Generate H-Tree
cts_tree, updated_map = generate_cts_tree(
    placement_map_path="build/6502/greedy/6502.map",
    fabric_cells_path="fabric/fabric_cells.yaml",
    design_path="designs/6502_mapped.json",
    tree_type='h'
)

# Access tree structure
for buffer_name, buffer_info in cts_tree.items():
    if not buffer_name.startswith('_'):
        print(f"{buffer_name}: {len(buffer_info['children'])} children")
```

## Checklist
- [x] Code follows project style guidelines
- [x] API function matches specification
- [x] Error handling implemented
- [x] Documentation in docstrings
- [x] Command-line interface included
- [ ] Integration tested with eco_generator.py
- [ ] Unit tests added (if applicable)

## Reviewers
Please review:
- API function signature and return format
- Error handling completeness
- Integration with BufferManager
- Code clarity and documentation

## Next Steps
- Test integration with eco_generator.py
- Add unit tests if needed
- Update documentation for Member B

