# Structured-ASIC placement and back-end flow

A Python- and Tcl-based **course project** for mapping four synthesized designs onto a fixed Sky130 cell fabric, optimizing placement with a greedy barycenter heuristic and simulated annealing, and exploring clock-tree construction, ECO insertion, OpenROAD routing, and static timing analysis. The main contribution is the placement/flow integration and its inspectable experiments. **This is a research/educational back-end prototype, not a tapeout-ready or independently signed-off ASIC.**

## At a glance

| Inspect | Starting point | Evidence / caveat |
| --- | --- | --- |
| Placement engine | [`placer.py`](placer.py), [`parse_design.py`](parse_design.py), [`parse_fabric.py`](parse_fabric.py), [`validator.py`](validator.py) | Greedy placement, same-type swaps and SA search against HPWL |
| Clocking and ECO | [`eco_generator.py`](eco_generator.py), [`cts_api.py`](cts_api.py), [`cts_htree.py`](cts_htree.py), [`cts_xtree.py`](cts_xtree.py), [`buffer_manager.py`](buffer_manager.py) | H-tree/X-tree alternatives and fabric-slot allocation; clock-tree quality not independently characterized here |
| Physical outputs | [`make_def.py`](make_def.py), [`rename.py`](rename.py), [`route.tcl`](route.tcl), [`sta.tcl`](sta.tcl) | DEF/netlist generation, OpenROAD routing and timing scripts; not a replacement for complete signoff |
| Demonstrations | [`Animations/`](Animations/), [`Results & Graphs/`](Results%20%26%20Graphs/), [`Final_Presentation.pdf`](Final_Presentation.pdf) | Historical visuals and course presentation, **not necessarily from the best SA experiment** |
| Inputs and commands | [`designs/`](designs/), [`fabric/`](fabric/), [`tech/`](tech/), [`Makefile`](Makefile) | Four pre-mapped netlist inputs; requires compatible Python libraries, PDK files and OpenROAD to execute end to end |

## What the reported optimization means

The repository contains an experiment log at [`build/6502/logs/config_11_m6000_T0.00012_a0.996.log`](build/6502/logs/config_11_m6000_T0.00012_a0.996.log) for a 6502 simulated-annealing run. The original project reports **426,853 µm greedy HPWL → 101,347 µm best SA HPWL (~76.3% reduction)** at 6000 moves per temperature, cooling factor 0.996. These figures are **project-reported, single-run placement-objective values**, not a claimed reduction in routed wire length, final delay, power, chip area, or manufacturing cost. This review verified the presence and initial parameters of the named log but did not independently recompute the final metric from the placement, reproduce the optimization, or validate the full physical flow. The prominently displayed heatmaps in the historical [original README](docs/README_original_2026-06-30.md) belong to an earlier, lighter run, **not** the reported best configuration.

## Quickstart: inspect before running expensive steps

```bash
# Python dependencies (prefer a fresh virtual environment)
python -m pip install -r requirements.txt

# Show the full build command graph WITHOUT running any EDA or optimization work.
make -n all DESIGN=arith

# Optional: run greedy placement alone on the smaller arith mapped design.
# Check available CPU, RAM, dependency versions and existing build outputs first.
make greedy DESIGN=arith

# Full experimental back-end flow: requires OpenROAD, the Sky130 technology
# artifacts, time and adequate local resources. NOT executed in this review.
# make all DESIGN=arith
```

Input designs are `arith`, `6502`, `z80`, and `aes_128`. They are **synthesized/mapped JSON inputs**; there is no verified source-RTL-to-tapeout pipeline here. The build graph generates greedy placement, SA placement, CTS/ECO placement and netlist, a renamed netlist, routed DEF, and timing reports. See [`docs/BUILD_AND_EVIDENCE.md`](docs/BUILD_AND_EVIDENCE.md) for the exact artifact mapping, known constraints, and the distinction between a command dry-run and a physically validated result.

## Repository organization and provenance

`designs/` holds mapped design inputs; `fabric/` fixes the candidate slots/pins; `tech/` holds Sky130 technology collateral; root Python/Tcl files implement stages; `build/`, `newResults/`, `Results & Graphs/` and `Animations/` contain historical experiment artifacts/visuals. This review does **not** delete, reclassify, or regenerate those artifacts. The long-form course write-up is preserved verbatim at [`docs/README_original_2026-06-30.md`](docs/README_original_2026-06-30.md); treat its performance and signoff statements as historical claims until reproduced. No personal ownership breakdown is inferred from commit history.

## Verification status

A Makefile dependency and output-path mismatch was corrected in this review. Static Makefile checks are in [`tests/test_makefile_contract.py`](tests/test_makefile_contract.py); the proposed CI checks its command graph and Python syntax without launching an EDA flow. **No OpenROAD route, STA run, timing closure, DRC, LVS, fabricated-chip test, or independent HPWL benchmarking was performed during this review.** For a hardware hiring discussion, distinguish implemented flow scripts and recorded exploratory results from verified engineering signoff.
