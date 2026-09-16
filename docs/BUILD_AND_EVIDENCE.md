# Build graph, artifacts, and evidence boundaries

## Baseline and scope

The review branch starts from `5a52c6266104c7516749dda09ea1898b4ef47c55` (2026-06-30). The historical 40 KB root README is preserved at [`README_original_2026-06-30.md`](README_original_2026-06-30.md) with no changes to its claims. The review concentrates on the Makefile/related generator contracts and recruiter-facing documentation; **it is not an every-file, independent audit of the ~363 MB project and outputs**. Large mapped JSON, timing collateral, images, PDF and video artifacts were inventoried but not independently interpreted in this review.

## Repaired Makefile contract

The old Makefile had `$(ROUTED_DEF): $(RENAMED_VERILOG) $(ECO_MAP)` but only defined a phony `rename:` recipe: from a clean output directory, Make could not find a rule to build the renamed Verilog prerequisite. The repair makes `$(RENAMED_VERILOG)` a real target depending on the ECO Verilog and map, so the `all -> sta -> route -> rename -> CTS/ECO -> SA -> greedy` dependency chain is representable.

The ECO generator writes distinct files: `placement_path.replace('.json', '_eco.json')` is **the updated placement** (`build/<design>/<design>_sa_placement_eco.json`); `--output-json` is the **updated mapped design/netlist** (`build/<design>/<design>_eco_netlist.json`); and the generator additionally emits an `*_eco_eco.json` bookkeeping file from that output path. The original Makefile called the mapped-design output `$(ECO_JSON)` and the README called it the placement, which conflated two different schemas. The repaired Makefile assigns the two primary files different names and checks the produced artifacts exist. Historical artifacts and their names are not renamed.

`make -n all DESIGN=arith` only checks the declared Make dependency graph and expands commands. It does **not** validate optimizer results, correctness of the generated Verilog, map/DEF net consistency, routed connectivity, timing closure or physical manufacturability. Running the real flow needs matching Sky130 collateral, OpenROAD, storage and significant time; inspect `make -n` and choose a small experiment first.

## Evidence table

| Claim | Status in this review | What would establish it |
| --- | --- | --- |
| Four mapped design inputs | File inventory confirmed; contents not independently validated | Schema/port/cell checks for each input and source attribution |
| 6502 best SA HPWL 426,853 → 101,347 µm | Historical log and README claim; final value not independently recomputed | Parse final logged best and recompute full-net HPWL from exact saved placement, compare seeds and runtime |
| Incremental HPWL 10–100× faster | Historical performance claim, unverified | Matched full-cost vs incremental implementation benchmark, hardware and timing logs |
| Generated H-tree/X-tree, ECO, route and STA | Implementation scripts present; complete run unverified | Run each stage from fresh checkout, validate connectivity, report tools/versions, route DRC and timing coverage |
| Timing closure/signoff | Unverified; cannot infer from script presence | Actual constrained STA reports and full relevant physical verification, multiple PVT corners as appropriate |

No new optional algorithm, CTS topology, or feature was introduced: correctness and evidence were prioritized. No historical experiment artifact or third-party collateral was deleted.
