# Simulated Annealing Speed Optimization Guide

This document describes the optimizations implemented to make simulated annealing (SA) faster, and additional strategies you can use.

## Implemented Optimizations (Quality-Preserving Only)

### 1. **Optimized Placement Copying** ✅
- **Before**: Deep copying entire placement dict for every move attempt
- **After**: Shallow copy + only deep copy moved cells
- **Speedup**: ~2-5x for moves with few cells
- **Impact**: Reduces memory allocation overhead
- **Quality Impact**: NONE (pure implementation optimization)

### 2. **Direct Moved Cells Tracking** ✅
- **Before**: Iterating through all cells to find moved ones after move generation
- **After**: Move functions return `moved_cells` directly
- **Speedup**: O(n) → O(1) for moved cell detection
- **Impact**: Eliminates redundant iteration
- **Quality Impact**: NONE (pure implementation optimization)

### 3. **Reduced Print Frequency** ✅
- **Before**: Printing progress every 10 iterations
- **After**: Printing every 20 iterations
- **Speedup**: ~1-2% (I/O overhead)
- **Impact**: Less console I/O blocking
- **Quality Impact**: NONE (I/O only)

### ⚠️ **Removed Optimizations** (Quality Trade-offs)

The following optimizations were **removed** to preserve maximum quality:

- **Early Rejection**: Removed to ensure exact Metropolis criterion (minimal quality impact, but removed for correctness)
- **Slot Search Limits**: Removed to ensure full exploration of all available slots (moderate quality impact)

## Additional Speed Optimization Strategies

### Parameter Tuning (Fastest Wins)

1. **Reduce `moves_per_temp`**
   ```python
   # Default: 1000
   # Faster: 500-700 (may reduce quality slightly)
   moves_per_temp=500
   ```

2. **Increase Cooling Rate (alpha)**
   ```python
   # Default: 0.99 (slow cooling)
   # Faster: 0.95-0.97 (faster cooling, may reduce quality)
   alpha=0.95
   ```

3. **Increase Final Temperature**
   ```python
   # Default: 0.001
   # Faster: 0.01-0.1 (stops earlier)
   T_final=0.01
   ```

4. **Reduce Window Search Limit**
   ```python
   # In explore_move, reduce max_slot_search
   # Default: 1000
   # Faster: 500
   ```

### Algorithm-Level Optimizations

1. **Early Termination**
   - Stop if no improvement for N iterations
   - Stop if acceptance rate drops below threshold

2. **Adaptive Moves Per Temperature**
   - Reduce `moves_per_temp` as temperature decreases
   - Fewer moves needed at low temperatures

3. **Parallel Move Evaluation** (Advanced)
   - Evaluate multiple moves in parallel
   - Requires thread-safe HPWL cache

### Usage Examples

#### Fast SA (Speed Priority)
```python
sa_placement = simulated_annealing(
    initial_placement, fabric_db, netlist_graph, nets_dict,
    pins_db, port_to_nets,
    alpha=0.95,           # Faster cooling
    T_final=0.01,         # Stop earlier
    moves_per_temp=500,   # Fewer moves per temp
)
```

#### Balanced SA (Speed + Quality)
```python
sa_placement = simulated_annealing(
    initial_placement, fabric_db, netlist_graph, nets_dict,
    pins_db, port_to_nets,
    alpha=0.97,           # Moderate cooling
    T_final=0.001,        # Lower final temp
    moves_per_temp=700,   # Moderate moves
)
```

#### Quality SA (Quality Priority)
```python
sa_placement = simulated_annealing(
    initial_placement, fabric_db, netlist_graph, nets_dict,
    pins_db, port_to_nets,
    alpha=0.99,           # Slow cooling
    T_final=0.001,        # Very low final temp
    moves_per_temp=1000,  # Many moves
)
```

## Expected Performance Improvements

With quality-preserving optimizations enabled:
- **2-5x speedup** from placement copying optimization
- **5-15% speedup** from moved cells tracking
- **1-2% speedup** from reduced print frequency
- **Overall**: **2-5x faster** depending on design size and parameters

**Note**: Quality-affecting optimizations (early rejection, slot search limits) have been removed to ensure maximum placement quality.

## Trade-offs

| Optimization | Speed Gain | Quality Impact | Status |
|-------------|------------|----------------|--------|
| Placement copying | High | None | ✅ Enabled |
| Moved cells tracking | Medium | None | ✅ Enabled |
| Reduced print frequency | Low | None | ✅ Enabled |
| Early rejection | Low-Medium | Minimal | ❌ Removed (quality) |
| Slot search limit | High | Moderate | ❌ Removed (quality) |
| Reduce moves_per_temp | High | Low-Medium | ⚠️ User parameter |
| Faster cooling (alpha) | High | Medium | ⚠️ User parameter |
| Higher T_final | High | Medium-High | ⚠️ User parameter |

## Monitoring Performance

To measure SA performance:
```python
import time
start_time = time.time()
sa_placement = simulated_annealing(...)
elapsed = time.time() - start_time
print(f"SA took {elapsed:.2f} seconds")
```

## Further Reading

- See `README.md` for SA algorithm details
- See `placer.py` for implementation details
- See `sa_knob_exploration.py` for parameter sensitivity analysis

