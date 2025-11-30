# Implement X-Tree Clock Tree Synthesis Algorithm

## Description
Implement an X-Tree Clock Tree Synthesis (CTS) algorithm as an alternative to the existing H-Tree implementation. The X-Tree uses diagonal partitioning (NW/SE and NE/SW quadrants) instead of horizontal/vertical splits.

## Objectives
- [x] Create `cts_xtree.py` with diagonal partitioning logic
- [x] Implement alternating diagonal splits (NW/SE and NE/SW) at each level
- [x] Use same buffer selection and tree structure as H-Tree
- [x] Return same format: `{buffer_slot_name: {children: [...], sinks: [...], x, y}}`
- [x] Create runner script `run_6502_cts_xtree.py` for 6502 design
- [ ] Run X-Tree on 6502 design (greedy and best placements)
- [ ] Compare X-Tree vs H-Tree metrics (wirelength, skew, buffer usage)

## Technical Details

### X-Tree Partitioning Algorithm
- **Even levels (0, 2, 4...)**: NW/SE split using `x + y < center_x + center_y`
- **Odd levels (1, 3, 5...)**: NE/SW split using `x - y < center_x - center_y`
- Fallback to median split if diagonal partition results in empty partition

### Implementation Files
- `cts_xtree.py` - Main X-Tree CTS implementation
- `run_6502_cts_xtree.py` - Runner script for 6502 design

### Output Directories
- `build/6502/cts/greedy_xtree/` - X-Tree results for greedy placement
- `build/6502/cts/best_xtree/` - X-Tree results for best SA placement

## Acceptance Criteria
- [x] X-Tree implementation matches H-Tree structure and interface
- [x] Diagonal partitioning correctly alternates between NW/SE and NE/SW
- [x] Output files don't override existing H-Tree results
- [ ] X-Tree successfully runs on 6502 design
- [ ] Metrics comparison shows X-Tree performance characteristics

## Related
- Related to H-Tree implementation: `cts_htree.py`
- Comparison study: X-Tree vs H-Tree for clock skew and wirelength

## Labels
- `enhancement`
- `cts`
- `algorithm`

