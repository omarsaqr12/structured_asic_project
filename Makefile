# Educational structured-ASIC physical-design flow.
# Run `make -n all DESIGN=arith` to examine the commands without execution.
# A successful dry run is NOT an OpenROAD, timing, DRC or signoff test.
DESIGN ?= arith
PYTHON ?= python3
OPENROAD ?= openroad
SA_ALPHA ?= 0.97
SA_MOVES ?= 700
SA_T_FINAL ?= 0.001
BUILD_DIR ?= build
DESIGN_DIR = $(BUILD_DIR)/$(DESIGN)
DESIGN_JSON = designs/$(DESIGN)_mapped.json
FABRIC_CELLS = fabric/fabric_cells.yaml
PINS_YAML = fabric/pins.yaml
LIBERTY_FILE = tech/sky130_fd_sc_hd__tt_025C_1v80.lib
LEF_FILE = tech/sky130_fd_sc_hd_merged.lef
GREEDY_JSON = $(DESIGN_DIR)/$(DESIGN)_placement.json
GREEDY_MAP = $(DESIGN_DIR)/$(DESIGN).map
SA_JSON = $(DESIGN_DIR)/$(DESIGN)_sa_placement.json
SA_MAP = $(DESIGN_DIR)/$(DESIGN)_sa.map
# ECO generator derives placement output from its --placement argument.
ECO_PLACEMENT_JSON = $(DESIGN_DIR)/$(DESIGN)_sa_placement_eco.json
# --output-json is the modified mapped design, NOT the placement.
ECO_NETLIST_JSON = $(DESIGN_DIR)/$(DESIGN)_eco_netlist.json
ECO_VERILOG = $(DESIGN_DIR)/$(DESIGN)_eco.v
ECO_MAP = $(DESIGN_DIR)/$(DESIGN)_eco.map
RENAMED_VERILOG = $(DESIGN_DIR)/$(DESIGN)_renamed.v
ROUTED_DEF = $(DESIGN_DIR)/$(DESIGN)_routed.def

.PHONY: all help greedy sa cts_eco rename route sta clean
all: sta
help:
	@echo 'Usage: make [greedy|sa|cts_eco|rename|route|sta|all|clean] DESIGN=[arith|6502|z80|aes_128]'
	@echo 'Preview without running: make -n all DESIGN=arith'

greedy: $(GREEDY_JSON) $(GREEDY_MAP)
$(GREEDY_JSON) $(GREEDY_MAP): $(DESIGN_JSON) $(FABRIC_CELLS) $(PINS_YAML) placer.py
	@mkdir -p $(DESIGN_DIR)
	$(PYTHON) placer.py --design $(DESIGN_JSON) --fabric-cells $(FABRIC_CELLS) --pins $(PINS_YAML) --no-sa --output $(DESIGN_DIR)
	@test -f $(GREEDY_JSON) && test -f $(GREEDY_MAP)

sa: $(SA_JSON) $(SA_MAP)
$(SA_JSON) $(SA_MAP): $(GREEDY_JSON) $(GREEDY_MAP) placer.py
	$(PYTHON) placer.py --design $(DESIGN_JSON) --fabric-cells $(FABRIC_CELLS) --pins $(PINS_YAML) --initial-placement $(GREEDY_JSON) --sa-alpha $(SA_ALPHA) --sa-moves $(SA_MOVES) --sa-T-final $(SA_T_FINAL) --output $(DESIGN_DIR)
	@test -f $(SA_JSON) && test -f $(SA_MAP)

cts_eco: $(ECO_VERILOG) $(ECO_MAP) $(ECO_PLACEMENT_JSON) $(ECO_NETLIST_JSON)
$(ECO_VERILOG) $(ECO_MAP) $(ECO_PLACEMENT_JSON) $(ECO_NETLIST_JSON): $(SA_JSON) $(SA_MAP) eco_generator.py
	$(PYTHON) eco_generator.py --placement $(SA_JSON) --design $(DESIGN_JSON) --fabric-cells $(FABRIC_CELLS) --placement-map $(SA_MAP) --enable-cts --cts-tree-type h --output-json $(ECO_NETLIST_JSON) --output-verilog $(ECO_VERILOG)
	@test -f $(ECO_VERILOG) && test -f $(ECO_MAP) && test -f $(ECO_PLACEMENT_JSON) && test -f $(ECO_NETLIST_JSON)

# A file-generating rule, unlike the original phony-only rename target.
rename: $(RENAMED_VERILOG)
$(RENAMED_VERILOG): $(ECO_VERILOG) $(ECO_MAP) rename.py
	$(PYTHON) rename.py --design $(DESIGN) --final-v $(ECO_VERILOG) --map $(ECO_MAP) --output $(RENAMED_VERILOG)
	@test -f $(RENAMED_VERILOG)

route: $(ROUTED_DEF)
$(ROUTED_DEF): $(RENAMED_VERILOG) $(ECO_MAP) make_def.py route.tcl
	$(PYTHON) make_def.py --design $(DESIGN) --map $(ECO_MAP)
	DESIGN_NAME=$(DESIGN) BUILD_DIR=$(BUILD_DIR) LIBERTY_FILE=$(LIBERTY_FILE) LEF_FILE=$(LEF_FILE) $(OPENROAD) -exit route.tcl
	@test -f $(ROUTED_DEF)

sta: $(ROUTED_DEF) sta.tcl
	DESIGN_NAME=$(DESIGN) BUILD_DIR=$(BUILD_DIR) LIBERTY_FILE=$(LIBERTY_FILE) LEF_FILE=$(LEF_FILE) $(OPENROAD) -exit sta.tcl

# Destructive removal affects only the selected design's output directory.
clean:
	rm -rf -- $(DESIGN_DIR)
	rm -f -- route.log
