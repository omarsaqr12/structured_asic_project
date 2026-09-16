# Structured-ASIC placement & physical-design flow

**Python · Tcl · SkyWater Sky130 · OpenROAD · Digital Design II course project**  
[![Build graph and syntax checks](https://github.com/omarsaqr12/structured_asic_project/actions/workflows/build-graph.yml/badge.svg?branch=main)](https://github.com/omarsaqr12/structured_asic_project/actions/workflows/build-graph.yml)

An educational back-end prototype that maps synthesized logic onto **fixed, type-compatible fabric slots**, improves placement with a connectivity-driven greedy heuristic and simulated annealing, and connects that placement to clock-tree/ECO generation, DEF/netlist creation, OpenROAD routing, and static timing analysis scripts. Four mapped design inputs are included: **6502, Z80, AES-128, and arithmetic**.

**Recorded placement result:** the completed [6502 optimization log](build/6502/logs/config_11_m6000_T0.00012_a0.996.log) reports **426,853.12 → 101,346.87 µm HPWL (−76.26%)** for one greedy-to-annealed run. HPWL is an *estimated placement objective*; this is not a measured improvement in routed wire length, delay, area, or power, and the result has not been independently recomputed or reproduced in this review.

## Architecture

```text
Mapped design JSON + fixed fabric / I/O pins
                   │
          Greedy barycenter placement
                   │
       Simulated annealing (HPWL)
                   │
        Clock-tree generation + ECO
                   │
       Netlist / slot-name alignment
                   │
            DEF + OpenROAD route
                   │
          Parasitics + timing scripts
```

| Explore | Key implementation | What to look for |
| --- | --- | --- |
| **Placement** | [`placer.py`](placer.py) · [`parse_design.py`](parse_design.py) · [`parse_fabric.py`](parse_fabric.py) | Connectivity-driven seed/greedy placement, same-type moves, incremental HPWL evaluation |
| **Clocking & ECO** | [`cts_htree.py`](cts_htree.py) · [`cts_xtree.py`](cts_xtree.py) · [`buffer_manager.py`](buffer_manager.py) · [`eco_generator.py`](eco_generator.py) | Alternative tree-generation code, buffer-slot allocation, mapped-netlist/placement updates |
| **Physical flow** | [`make_def.py`](make_def.py) · [`rename.py`](rename.py) · [`route.tcl`](route.tcl) · [`sta.tcl`](sta.tcl) | Instance/slot alignment, DEF generation, routing and timing scripts |
| **Inputs & orchestration** | [`designs/`](designs/) · [`fabric/`](fabric/) · [`Makefile`](Makefile) | Four mapped inputs, fabric geometry, and a staged build graph |

### Placement visualizations

| Greedy placement | After an **earlier** simulated-annealing run |
| :---: | :---: |
| ![Historical 6502 greedy placement heatmap](build/6502/greedy/greedy_heatmap.png) | ![Historical 6502 annealed placement heatmap](build/6502/Best_sa_alpha0.99_moves1000_Tfinal0.001/sa_alpha0.99_moves1000_Tfinal0.001_heatmap.png) |

These historical heatmaps illustrate placement changes from an **earlier, lighter configuration**, **not** the 76.26% run quoted above. More historical artifacts are in [`Results & Graphs/`](Results%20%26%20Graphs/), [`Animations/`](Animations/), and the [course presentation](Final_Presentation.pdf); they have not been independently revalidated here.

## Reproduce the build *steps*

Run commands from the repository root in a suitable Python environment:

```bash
python -m pip install -r requirements.txt

# Inspect the build dependency graph without running optimization or EDA.
make -B -n all DESIGN=arith

# Optional: run only the greedy placement on the arith input.
make greedy DESIGN=arith

# Full experimental flow (OpenROAD + matching Sky130 collateral required):
# make all DESIGN=arith
```

The Makefile describes greedy → annealing → CTS/ECO → instance renaming → routing → STA. The review repaired a missing renamed-netlist dependency and separated the ECO generator's **updated placement JSON** from its **updated mapped-netlist JSON**. See [build and evidence notes](docs/BUILD_AND_EVIDENCE.md) for exact file contracts, prerequisites, and known verification gaps.

## Validation & scope

[CI](.github/workflows/build-graph.yml) checks the Makefile's dry-run dependency graph, ECO output naming contract, and selected Python modules' syntax. **CI does not execute optimization, OpenROAD, routing, STA, or physical verification.** The 6502 figures above are traceable to a completed project log, but no independent HPWL recomputation, four-design end-to-end reproduction, timing closure, DRC, LVS, or signoff was performed in this review. This is **not** an RTL-to-tapeout or tapeout-ready implementation.

For deeper technical detail, see the preserved [original project write-up](docs/README_original_2026-06-30.md) and [annealing optimization notes](SA_OPTIMIZATION_GUIDE.md). The original README contains historical performance and signoff claims that should be treated as project reports rather than newly validated results. Historical source, experiment outputs, visualizations, and technology collateral have been preserved.
