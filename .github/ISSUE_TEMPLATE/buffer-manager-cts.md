# Implement BufferManager for Clock Tree Synthesis

## Description
Implement a BufferManager system to track and manage buffer/inverter slot assignments during Clock Tree Synthesis (CTS). This provides centralized buffer claiming, validation, and placement map updates.

## Objectives
- [x] Create `buffer_manager.py` with BufferManager class
- [x] Implement buffer inventory tracking (available vs claimed)
- [x] Implement `claim_buffer()` method with nearest buffer lookup
- [x] Implement `update_placement_map()` for appending CTS buffers
- [x] Implement `export_claims()` for JSON export
- [x] Integrate BufferManager with H-Tree and X-Tree CTS
- [x] Create `run_6502_cts_with_manager.py` for running both algorithms
- [x] Run CTS with BufferManager for 6502 (greedy and best placements)
- [ ] Compare BufferManager results vs original CTS implementations

## Technical Details

### BufferManager Features
- **Buffer Inventory**: Tracks available BUF and INV slots from fabric
- **Placement Map Integration**: Reads existing placement.map to exclude used slots
- **Nearest Buffer Lookup**: Uses KD-tree for fast spatial search
- **Claim Management**: Tracks claimed buffers with logical names (cts_buffer_0, etc.)
- **Tree Structure**: Maintains parent-child relationships in claimed buffers
- **Validation**: Validates sufficient buffers before claiming

### Implementation Files
- `buffer_manager.py` - Main BufferManager implementation
- `run_6502_cts_with_manager.py` - Runner script for H-Tree and X-Tree with BufferManager

### Output Structure
- `build/6502/cts_with_manager/greedy_htree/` - H-Tree results for greedy placement
- `build/6502/cts_with_manager/greedy_xtree/` - X-Tree results for greedy placement
- `build/6502/cts_with_manager/best_htree/` - H-Tree results for best SA placement
- `build/6502/cts_with_manager/best_xtree/` - X-Tree results for best SA placement

Each directory contains:
- `6502_cts_tree.json` - Tree structure
- `claimed_buffers.json` - Buffer assignments from BufferManager
- `6502_with_cts.map` - Updated placement map with CTS buffers

## Acceptance Criteria
- [x] BufferManager correctly tracks available vs claimed buffers
- [x] Nearest buffer lookup works correctly (KD-tree or linear search)
- [x] Placement map updates preserve original entries
- [x] Claimed buffers JSON includes tree structure (children)
- [x] Integration works with both H-Tree and X-Tree algorithms
- [x] All 4 combinations (2 trees × 2 placements) run successfully
- [ ] Metrics comparison shows BufferManager results match original CTS

## Related
- Related to H-Tree implementation: `cts_htree.py`
- Related to X-Tree implementation: `cts_xtree.py`
- Comparison: BufferManager vs manual buffer selection

## Labels
- `enhancement`
- `cts`
- `buffer-management`

