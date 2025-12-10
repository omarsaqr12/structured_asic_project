# Changelog

## [2.0.0] - 2024-12-10 - Animations, CTS, and Routing Features

This release adds comprehensive visualization/animation capabilities, Clock Tree Synthesis (CTS), automated routing support, and SDC generation for the structured ASIC physical design flow.

---

## New Features

### 🎬 Animation System

A complete visualization suite for animating physical design algorithms in real-time.

#### Simulated Annealing Animation (`animate_sa_placement.py`)
- **Purpose**: Creates MP4 videos showing SA optimization progress
- **Features**:
  - Real-time visualization of cell movements during optimization
  - Temperature, HPWL, and acceptance rate overlays
  - Color-coded cells by type (NAND, OR, INV, DFF, etc.)
  - Configurable frame rate and recording interval
- **Usage**:
  ```bash
  python animate_sa_placement.py --design 6502 --alpha 0.95 --fps 10
  ```
- **Output**: `build/{design}/sa_animation.mp4`

#### CTS Tree Animation (`animate_cts_tree.py`, `create_cts_animations.py`)
- **Purpose**: Visualizes H-Tree and X-Tree clock distribution networks
- **Tree Types**:
  - **H-Tree**: Alternating horizontal/vertical partitioning
  - **X-Tree**: Diagonal NW/SE and NE/SW partitioning
- **Usage**:
  ```bash
  python create_cts_animations.py  # Creates both H-Tree and X-Tree animations
  ```
- **Output**: 
  - `{design}_cts_htree_animation.mp4`
  - `{design}_cts_xtree_animation.mp4`

#### Additional Animation Tools
| Script | Purpose |
|--------|---------|
| `animate_congestion.py` | Visualize routing congestion heatmaps |
| `animate_net_hpwl.py` | Animate net wirelength changes |
| `animate_clock_skew.py` | Show clock skew across the design |
| `animate_xh_trees.py` | Combined X/H tree visualization |
| `animate_greedy_placement.py` | Step-by-step greedy placement animation |

---

### 🌳 Clock Tree Synthesis (CTS)

Complete CTS implementation with multiple tree topologies.

#### CTS API (`cts_api.py`)
- **Clean API** for programmatic CTS generation
- **Buffer Management** integration via `BufferManager` class
- **Supported Topologies**:
  - H-Tree (balanced horizontal/vertical)
  - X-Tree (diagonal partitioning)
- **Key Functions**:
  ```python
  from cts_api import build_tree_with_manager, partition_sinks_htree, partition_sinks_xtree
  ```

#### Buffer Manager (`buffer_manager.py`)
- Manages buffer allocation from fabric slots
- Tracks buffer usage and availability
- Supports `sky130_fd_sc_hd__clkbuf_4` cells

#### CTS Implementations
| File | Description |
|------|-------------|
| `cts_htree.py` | H-Tree implementation |
| `cts_xtree.py` | X-Tree implementation |
| `cts_simulator.py` | Timing simulation for CTS |
| `cts_with_buffer_manager.py` | CTS with integrated buffer management |

#### Example Usage
```bash
python run_6502_cts.py           # Basic CTS
python run_6502_cts_xtree.py     # X-Tree CTS
python run_6502_cts_with_manager.py  # CTS with buffer management
```

---

### 🛣️ Automated Routing

OpenROAD-based routing with automatic error recovery.

#### Auto Router (`auto_route.py`)
- **Iterative routing loop** that automatically:
  1. Generates DEF file
  2. Runs OpenROAD detailed routing
  3. Extracts problematic cells from DRT errors
  4. Excludes problematic cells and retries
- **Usage**:
  ```bash
  python auto_route.py 6502
  ```
- **Features**:
  - Automatic DRT-0073 error parsing
  - Incremental cell exclusion
  - Maximum 30 iteration safety limit

#### Route TCL (`route.tcl`)
- Complete OpenROAD routing script
- Supports global and detailed routing
- Configurable via command-line or script modification

#### DEF Generation (`make_def.py`)
- **Modified** to support:
  - Problematic cell exclusion
  - Net filtering for unroutable connections
  - Proper SKY130 technology mapping

#### Supporting Files
| File | Purpose |
|------|---------|
| `extract_drt_errors.py` | Parse OpenROAD error logs |
| `check_lef_layers.tcl` | Verify LEF layer definitions |
| `merge_lef.py` | Merge technology and cell LEF files |
| `add_tracks_to_lef.py` | Add routing track definitions |

---

### ⏱️ SDC Generation

Automatic constraint file generation for Static Timing Analysis.

#### SDC Generator (`generate_sdc.py`)
- **Auto-generates** Synopsys Design Constraints
- **Per-design frequency targets**:
  | Design | Frequency | Period |
  |--------|-----------|--------|
  | 6502 | 25 MHz | 40 ns |
  | aes_128 | 50 MHz | 20 ns |
  | arith | 50 MHz | 20 ns |
  | z80 | 25 MHz | 40 ns |
- **Generated Constraints**:
  - `create_clock` for clock port
  - `set_input_delay` (excludes clock)
  - `set_output_delay`
- **Usage**:
  ```bash
  python generate_sdc.py --design all  # Generate for all designs
  python generate_sdc.py --design 6502  # Single design
  ```

---

## Modified Files

### Core Flow
| File | Changes |
|------|---------|
| `make_def.py` | Added problematic cell exclusion, improved net handling |
| `parse_fabric.py` | Enhanced fabric parsing for routing |
| `rename.py` | Improved cell renaming logic |
| `rename_helper.py` | Bug fixes for hierarchical names |

### Technology Files
| File | Changes |
|------|---------|
| `tech/sky130_fd_sc_hd.lef` | Fixed pin definitions, added OBS layers |
| `tech/sky130_fd_sc_hd.tlef` | Corrected layer definitions for routing |
| `fabric/pins.yaml` | Updated pin locations and directions |

### Build Outputs
| File | Changes |
|------|---------|
| `build/6502/6502_renamed.v` | Regenerated with fixes |

---

## New Output Files

### Animation Results
- `6502_cts_htree_animation.mp4` - H-Tree CTS animation
- `6502_cts_xtree_animation.mp4` - X-Tree CTS animation  
- `sa_animation.mp4` - SA placement optimization animation
- `build/6502/animation_usingbest/` - Animation frames using best placement

### Routing Results
- `6502_global_routed.def` - Global routing output
- `route.log` - Detailed routing log (generated at runtime)

### SDC Files
- `6502.sdc`, `aes_128.sdc`, `arith.sdc`, `z80.sdc`
- `sdc/` directory with organized SDC files

### Analysis & Results
- `newResults/` - New placement results and logs
- `net_imags/` - Net visualization images (822 PNGs)

---

## Dependencies

New Python package requirements:
```
imageio
imageio-ffmpeg
matplotlib
numpy
```

Install with:
```bash
pip install imageio imageio-ffmpeg matplotlib numpy
```

---

## Documentation Files Added

| File | Purpose |
|------|---------|
| `CTS_API_DOCUMENTATION.md` | CTS API reference |
| `SA_OPTIMIZATION_GUIDE.md` | SA parameter tuning guide |
| `SDC.md` | SDC constraints documentation |
| `ROUTING_FIX_SUMMARY.md` | Routing troubleshooting |
| `TECH_LEF_FIX.md` | Technology LEF fixes |
| Various `*.md` files | Issue tracking and fixes |

---

## Usage Examples

### Complete Flow with Animation
```bash
# 1. Run placement with animation
python animate_sa_placement.py --design 6502 --alpha 0.99 --moves-per-temp 500

# 2. Generate CTS animations
python create_cts_animations.py

# 3. Generate SDC constraints
python generate_sdc.py --design 6502

# 4. Generate DEF for routing
python make_def.py --design 6502

# 5. Run automated routing
python auto_route.py 6502
```

### Quick Visualization
```bash
# View SA optimization
python animate_sa_placement.py --design 6502 --fps 15 --frame-interval 5

# View CTS trees
python animate_cts_tree.py --design 6502 --tree-type h
python animate_cts_tree.py --design 6502 --tree-type x
```

---

## Known Issues

1. **Long routing times**: 100K+ instance designs may take 30-60 minutes for detailed routing
2. **Memory usage**: Animation generation requires ~3GB RAM for large designs
3. **DRT-0073 errors**: Some cells may require manual exclusion if auto_route.py reaches iteration limit

---

## Contributors

- Animations and visualization system
- CTS H-Tree and X-Tree implementations
- Automated routing loop
- SDC generation
- Technology file fixes for SKY130

