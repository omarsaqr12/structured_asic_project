# Pull Request: Implement BufferManager for Clock Tree Synthesis

## Description
This PR implements a BufferManager system for managing buffer/inverter slot assignments during CTS. It provides centralized buffer claiming, validation, and placement map updates for both H-Tree and X-Tree algorithms.

## Related Issue
Closes #<issue_number>

## Changes Made
- ✅ Created `buffer_manager.py` with BufferManager class
  - Buffer inventory tracking (available vs claimed)
  - Nearest buffer lookup using KD-tree or linear search
  - Claim management with logical names (cts_buffer_0, etc.)
  - Tree structure maintenance (parent-child relationships)
  - Placement map updates (append mode)
  - JSON export for claimed buffers
- ✅ Created `run_6502_cts_with_manager.py` for running both H-Tree and X-Tree
  - Integrates BufferManager with both algorithms
  - Processes both greedy and best SA placements
  - Organizes outputs in `build/6502/cts_with_manager/`
- ✅ Generated CTS results for all 4 combinations:
  - `greedy_htree/` - H-Tree for greedy placement
  - `greedy_xtree/` - X-Tree for greedy placement
  - `best_htree/` - H-Tree for best SA placement
  - `best_xtree/` - X-Tree for best SA placement

## Implementation Details

### BufferManager API
- `__init__(fabric_cells_path, placement_map_path=None)` - Initialize with fabric DB
- `claim_buffer(x, y, preferred_type='BUF', level=0, children=None)` - Claim nearest buffer
- `update_placement_map(path, append=True)` - Update placement.map file
- `export_claims(output_path)` - Export to JSON
- `validate_sufficient_buffers(required_count)` - Validate availability
- `get_available_count()` - Get counts by type
- `get_claimed_count()` - Get number of claimed buffers

### Integration Pattern
- Replaced manual buffer selection with BufferManager.claim_buffer()
- Tree nodes store logical buffer names instead of physical slot names
- BufferManager maintains tree structure in claimed buffers
- Placement maps updated with CTS buffers appended

## Testing
- [x] BufferManager correctly tracks available buffers
- [x] Nearest buffer lookup works for both BUF and INV types
- [x] Placement map updates preserve original entries
- [x] H-Tree runs successfully with BufferManager
- [x] X-Tree runs successfully with BufferManager
- [x] All 4 combinations generate correct output files
- [ ] Metrics comparison with original CTS implementations

## Output Files
Each combination generates:
- `6502_cts_tree.json` - Tree structure
- `claimed_buffers.json` - Buffer assignments with tree structure
- `6502_with_cts.map` - Updated placement map

## Checklist
- [x] Code follows project style guidelines
- [x] BufferManager integrates with existing CTS code
- [x] Output directories organized correctly
- [x] All build files included
- [x] No example/test files included
- [ ] Tests pass (if applicable)
- [ ] Documentation updated (if applicable)

## Reviewers
Please review:
- BufferManager buffer claiming logic
- Integration with H-Tree and X-Tree algorithms
- Output file formats and organization
- Placement map update correctness

