# Issue: Implement H-Tree CTS Algorithm

**Type:** Feature  
**Priority:** High  
**Assignee:** Member A  
**Milestone:** Phase 3 - CTS & ECO Netlist Generation

## Description

Implement a hierarchical H-Tree Clock Tree Synthesis (CTS) algorithm that distributes clock signals to all DFFs (clock sinks) while minimizing clock skew. The H-Tree uses alternating horizontal and vertical partitioning to create a balanced clock distribution network.

## Objectives

1. Parse `placement.map` to identify all placed DFFs (sinks)
2. Parse `fabric_cells.yaml` to find unused buffer/inverter slots
3. Recursively partition DFFs by geometric center using H-pattern (alternate horizontal/vertical splits)
4. For each partition, find the nearest available buffer to the geometric center
5. Build a hierarchical tree structure with buffer assignments
6. Return the tree structure and list of claimed buffers

## Implementation Details

### Input Files
- `placement.map` - Format: `logical_name physical_slot_name`
- `fabric_cells.yaml` - Contains all buffer/inverter slots (BUF, INV types)
- Use `parse_design.py` to identify DFF cell types (look for `sky130_fd_sc_hd__dfbbp_1` or `DFBBP` in fabric)
- Use `parse_fabric.py` to get all buffer/inverter slots

### Algorithm Steps

1. **Identify Clock Sinks (DFFs):**
   - Extract all DFF instances from `placement.map`
   - Get their physical coordinates from `fabric_cells.yaml`
   - Store as list of `{name, x, y, slot_name}`

2. **H-Tree Partitioning:**
   ```
   function build_htree(sinks, available_buffers):
       if len(sinks) <= 1:
           return sinks  # Leaf node
       
       # Partition by geometric center
       center_x = mean([s.x for s in sinks])
       center_y = mean([s.y for s in sinks])
       
       # H-Tree: Split horizontally then vertically
       if depth % 2 == 0:
           # Split vertically (left/right)
           left = [s for s in sinks if s.x < center_x]
           right = [s for s in sinks if s.x >= center_x]
       else:
           # Split horizontally (top/bottom)
           top = [s for s in sinks if s.y < center_y]
           bottom = [s for s in sinks if s.y >= center_y]
       
       # Find nearest buffer to center
       buffer = find_nearest_buffer(center_x, center_y, available_buffers)
       claim_buffer(buffer)
       
       # Recursively build subtrees
       children = []
       for partition in [left, right] or [top, bottom]:
           if len(partition) > 0:
               child_tree = build_htree(partition, available_buffers)
               children.append(child_tree)
       
       return {buffer: children}
   ```

3. **Buffer Selection:**
   - Use KD-tree or distance calculation to find nearest unclaimed buffer
   - Prefer buffers over inverters (buffers are non-inverting)
   - Track claimed buffers to avoid double-assignment

4. **Tree Structure Output:**
   - Return nested dictionary: `{buffer_slot: {children: [...], sinks: [...], x, y}}`
   - Include root buffer (clock source)
   - Track parent-child relationships for wire routing

## Deliverables

- `cts_htree.py` - H-Tree implementation
- Function: `build_htree(placement_map_path, fabric_cells_path) -> Dict`
- Handle edge cases: single DFF, no buffers available, unbalanced partitions

## Test Cases

- Single DFF → Should return DFF directly (no buffer needed)
- 2-4 DFFs → Should use 1 buffer
- 100+ DFFs → Should create balanced tree with ~log2(N) levels

## Dependencies

- Phase 2 complete (`build/[design]/[design].map` exists)
- `parse_design.py` module
- `parse_fabric.py` module

## Acceptance Criteria

- ✅ All DFFs are identified from placement.map
- ✅ Unused buffers are correctly identified
- ✅ H-Tree partitioning creates balanced structure
- ✅ Nearest buffer selection works correctly
- ✅ Tree structure is properly hierarchical
- ✅ Edge cases are handled gracefully

## Related Tasks

- Task 2: X-Tree CTS Algorithm (for comparison)
- Task 3: Buffer Claiming System (will use this tree structure)
- Task 4: Minimal API (will wrap this implementation)

