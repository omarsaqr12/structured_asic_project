# Pull Request: Implement X-Tree Clock Tree Synthesis Algorithm

## Description
This PR implements the X-Tree Clock Tree Synthesis algorithm as an alternative to H-Tree, using diagonal partitioning instead of horizontal/vertical splits.

## Related Issue
Closes #<issue_number>

## Changes Made
- ✅ Created `cts_xtree.py` with X-Tree CTS implementation
  - Diagonal partitioning (NW/SE and NE/SW quadrants)
  - Alternating diagonal splits at each recursion level
  - Same buffer selection and tree structure as H-Tree
- ✅ Created `run_6502_cts_xtree.py` for running X-Tree on 6502 design
  - Separate output directories to avoid overriding H-Tree results
  - Supports both greedy and best SA placements

## Implementation Details

### X-Tree Algorithm
- **Even levels**: NW/SE split using `x + y < center_x + center_y`
- **Odd levels**: NE/SW split using `x - y < center_x - center_y`
- Fallback to median split if partition is empty

### Output Structure
- X-Tree results stored in separate directories:
  - `build/6502/cts/greedy_xtree/`
  - `build/6502/cts/best_xtree/`
- Does not override existing H-Tree results in `greedy/` and `best/` directories

## Testing
- [ ] X-Tree runs successfully on 6502 greedy placement
- [ ] X-Tree runs successfully on 6502 best SA placement
- [ ] Output files are generated correctly
- [ ] No conflicts with existing H-Tree results

## Comparison Metrics (To be added)
- Wirelength comparison: X-Tree vs H-Tree
- Skew comparison: X-Tree vs H-Tree
- Buffer usage comparison

## Checklist
- [x] Code follows project style guidelines
- [x] Implementation matches H-Tree interface
- [x] Output directories don't conflict with H-Tree
- [ ] Tests pass (if applicable)
- [ ] Documentation updated (if applicable)

## Reviewers
Please review:
- Diagonal partitioning logic correctness
- Buffer selection and tree structure
- Output format consistency with H-Tree

