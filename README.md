# structured_asic_project

Implementation of a complete physical design flow (Placement , CTS , Routing , and STA ) for a structured ASIC platform, as part of the [CSCE330401 - Digital Design II ] project.

## SA Knob Analysis

### Experiment Description

This analysis explores the impact of Simulated Annealing (SA) parameters on placement quality and runtime:

- **Cooling Rate (alpha)**: Controls temperature decay rate (0.80-0.99)
- **Moves per Temperature**: Number of moves attempted at each temperature (50-500)
- **P_refine**: Probability of using refine moves vs. explore moves (0.5-0.9)

The experiments systematically test combinations of these parameters to identify optimal trade-offs between placement quality (measured by Half-Perimeter Wirelength, HPWL) and runtime.

### Key Findings

To generate analysis results for a specific design, run:

```bash
python sa_knob_exploration.py --design <design_name>
```

This will:

1. Run experiments with all parameter combinations
2. Generate `sa_knob_analysis.png` showing the runtime vs. HPWL trade-off
3. Save results to `sa_knob_results_<design>.csv`
4. Print analysis with best configurations and recommendations

#### Best HPWL Configuration

The configuration that achieves the lowest HPWL (best placement quality).

#### Fastest Runtime Configuration

The configuration that completes placement in the shortest time.

#### Recommended Default Configuration

The configuration on the Pareto frontier that provides the best trade-off between quality and speed. This is recommended as the default for general use.

### Visualization

The analysis generates a scatter plot (`sa_knob_analysis.png`) showing:

- **X-axis**: Runtime (seconds)
- **Y-axis**: Final HPWL (microns)
- **Color-coding**: Points are colored by alpha value (cooling rate)
- **Pareto Frontier**: Red dashed line and star markers indicate Pareto-optimal configurations

**Pareto Frontier**: Points where no other configuration has both lower HPWL and lower runtime. These represent the best trade-offs between quality and speed.

![SA Knob Analysis](sa_knob_analysis.png)

_Note: Run the analysis script to generate the plot and see actual results for your design._

## Design-Specific Highlights

| Design | Best SA configuration (α / moves / T_final) | Output directory                                      |
| ------ | ------------------------------------------- | ----------------------------------------------------- |
| 6502   | 0.99 / 1000 / 0.001                         | build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/ |
| arith  | 0.99 / 500 / 0.001                          | build/arith/Best_sa_alpha0.99_moves500_Tfinal0.001/ |

Both designs rely on the auto-calculated initial temperature T_initial = 10,000 × HPWL_greedy. For 6502, the greedy seed HPWL of 403,784.5 µm produces T_initial ≈ 4.0 × 10^9, which guarantees near-100 % acceptance at the start and lets SA explore aggressively before cooling. We noted from the logs that this initial temperature might be too high and we intend to reduce the 10000 factor to 10 to 100.

### 6502 (heavy exploration mode)

- *What we ran:* Exhaustive sweep across α∈[0.90,0.99], moves∈[300,1000], T_final∈{0.001,0.01,0.1}. Logs for every run live in build/6502/logs/.

- *Best-quality knob set:* α=0.99, moves=1000, T_final=0.001. Artifacts (heatmaps + histograms) are under build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/. Comparative figures:

  - Greedy: build/6502/greedy/greedy_heatmap.png, build/6502/greedy/greedyHisto.jpeg

  - SA vs. Greedy runtime/HPWL frontier: build/6502/runtime_vs_hpwl_6502.png

- *Measured impact (log):* build/6502/logs/6502_sa_a0.99_m1000_T0.001.log shows the greedy baseline at *431,210 µm* and the SA result at *264,256 µm, a drop of **166,954 µm (−38.7 %)*.

- *Distribution insight (histogram):* build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/histogram_final.png still highlights the dramatic shrink in median HPWL (88.8 → 31.5 µm) and mean (155.5 → 95.6 µm), even though a few long global nets stretch to ~1.95 mm.

- *Why SA beats greedy:* Greedy prefers hugging IO pins, producing dense pockets and elongated cross-chip nets once clusters connect to distant logic. SA starts with randomized moves accepted under the high T_initial, then gradually cools (α=0.99) while still trying 1000 moves per temperature step. The random exploration plus uphill acceptance in early stages loosens greedy clusters, redistributes congestion, and lowers overall HPWL even though a handful of long nets get longer.

![6502 Runtime vs HPWL](docs/assets/runtime_vs_hpwl_6502.png)

### arith (moderate exploration mode)

- *What we ran:* Grid over α∈[0.80,0.99], moves∈[50,500], T_final∈{0.1,0.01,0.001}. Logs are stored in build/arith/logs/, and the sweep summary is build/arith/runtime_vs_hpwl_arith.png.

- *Best-quality knob set:* α=0.99, moves=500, T_final=0.001 (see build/arith/Best_sa_alpha0.99_moves500_Tfinal0.001/). The heavier schedule is still the clear QoR leader, while the balanced point (α=0.98, moves=300, T_final=0.01) trades ~12% higher HPWL for ~45% faster runtime.

- *Measured impact (log):* build/arith/logs/arith_sa_a0.99_m500_T0.001.log captures the greedy HPWL at *165,616 µm* and the SA finish at *50,216.99 µm* – a reduction of *115,399 µm (−69.6 %)*.

- *Behavioral notes:* Arith starts from a much cleaner greedy solution, so extreme SA settings primarily polish local structure. The moderate α/moves combos already converge well, and higher initial temperature still helps escape small pockets without the drastic congestion-shifting seen in 6502.
