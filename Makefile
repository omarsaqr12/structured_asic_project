# Makefile for Structured ASIC Project - Full Flow
#
# Steps:
# 1. Greedy Placement
# 2. Simulated Annealing (SA) - Medium parameters
# 3. Clock Tree Synthesis (CTS) & ECO
# 4. Instance Renaming
# 5. Iterative Routing (until complete)
# 6. Static Timing Analysis (STA)

# Design selection (default: arith)
DESIGN ?= arith

# Tools and Parameters
PYTHON = python
OPENROAD = openroad

# SA Medium Parameters (from SA_OPTIMIZATION_GUIDE.md)
SA_ALPHA = 0.97
SA_MOVES = 700
SA_T_FINAL = 0.001

# Directories
BUILD_DIR = build
DESIGN_DIR = $(BUILD_DIR)/$(DESIGN)
DESIGNS_DIR = designs
FABRIC_DIR = fabric
TECH_DIR = tech

# Input Files
DESIGN_JSON = $(DESIGNS_DIR)/$(DESIGN)_mapped.json
FABRIC_CELLS = $(FABRIC_DIR)/fabric_cells.yaml
PINS_YAML = $(FABRIC_DIR)/pins.yaml
LIBERTY_FILE = $(TECH_DIR)/sky130_fd_sc_hd__tt_025C_1v80.lib
LEF_FILE = $(TECH_DIR)/sky130_fd_sc_hd_merged.lef

# Intermediate/Output Files
GREEDY_JSON = $(DESIGN_DIR)/$(DESIGN)_placement.json
GREEDY_MAP = $(DESIGN_DIR)/$(DESIGN).map

SA_JSON = $(DESIGN_DIR)/$(DESIGN)_sa_placement.json
SA_MAP = $(DESIGN_DIR)/$(DESIGN)_sa.map

ECO_VERILOG = $(DESIGN_DIR)/$(DESIGN)_eco.v
ECO_JSON = $(DESIGN_DIR)/$(DESIGN)_eco.json
ECO_MAP = $(DESIGN_DIR)/$(DESIGN)_eco.map

RENAMED_VERILOG = $(DESIGN_DIR)/$(DESIGN)_renamed.v

ROUTED_DEF = $(DESIGN_DIR)/$(DESIGN)_routed.def

.PHONY: all greedy sa cts_eco rename route sta clean help

# Default target runs the full flow
all: sta

help:
	@echo "Structured ASIC Full Flow Makefile"
	@echo "Usage: make [target] DESIGN=[6502|aes_128|arith|z80]"
	@echo ""
	@echo "Targets:"
	@echo "  greedy    - Step 1: Greedy placement"
	@echo "  sa        - Step 2: SA placement (medium params)"
	@echo "  cts_eco   - Step 3: CTS and ECO generation"
	@echo "  rename    - Step 4: Rename instances to match slots"
	@echo "  route     - Step 5: Iterative routing until successful"
	@echo "  sta       - Step 6: Static Timing Analysis"
	@echo "  all       - Run the entire flow (greedy -> sta)"
	@echo "  clean     - Remove build artifacts for the current design"

# --- Step 1: Greedy Placement ---
greedy: $(GREEDY_MAP)

$(GREEDY_MAP) $(GREEDY_JSON): $(DESIGN_JSON) $(FABRIC_CELLS) $(PINS_YAML)
	@echo ">>> Step 1: Running Greedy Placement for $(DESIGN)..."
	@mkdir -p $(DESIGN_DIR)
	$(PYTHON) placer.py --design $(DESIGN_JSON) \
		--fabric-cells $(FABRIC_CELLS) \
		--pins $(PINS_YAML) \
		--no-sa \
		--output $(DESIGN_DIR)
	@echo "✓ Greedy placement complete"

# --- Step 2: SA Placement (Medium Parameters) ---
sa: $(SA_MAP)

$(SA_MAP) $(SA_JSON): $(GREEDY_JSON)
	@echo ">>> Step 2: Running SA Placement (Medium) for $(DESIGN)..."
	$(PYTHON) placer.py --design $(DESIGN_JSON) \
		--fabric-cells $(FABRIC_CELLS) \
		--pins $(PINS_YAML) \
		--initial-placement $(GREEDY_JSON) \
		--sa-alpha $(SA_ALPHA) \
		--sa-moves $(SA_MOVES) \
		--sa-T-final $(SA_T_FINAL) \
		--output $(DESIGN_DIR)
	@echo "✓ SA placement complete"

# --- Step 3: CTS & ECO ---
cts_eco: $(ECO_VERILOG)

$(ECO_VERILOG) $(ECO_JSON) $(ECO_MAP): $(SA_JSON) $(SA_MAP)
	@echo ">>> Step 3: Running CTS and ECO for $(DESIGN)..."
	$(PYTHON) eco_generator.py \
		--placement $(SA_JSON) \
		--design $(DESIGN_JSON) \
		--fabric-cells $(FABRIC_CELLS) \
		--placement-map $(SA_MAP) \
		--enable-cts \
		--cts-tree-type h \
		--output-json $(ECO_JSON) \
		--output-verilog $(ECO_VERILOG)
	@echo "✓ CTS and ECO complete"

# --- Step 4: Rename ---
.PHONY: rename
rename:
	@echo ">>> Step 4: Renaming instances for $(DESIGN)..."
	@if [ ! -f $(ECO_MAP) ]; then \
		echo "ERROR: ECO map file not found: $(ECO_MAP)"; \
		echo "Please run 'make cts_eco DESIGN=$(DESIGN)' first."; \
		exit 1; \
	fi
	@if [ -f $(ECO_VERILOG) ]; then \
		ECO_FILE=$(ECO_VERILOG); \
	elif [ -f $(DESIGN_DIR)/$(DESIGN)_final.v ]; then \
		ECO_FILE=$(DESIGN_DIR)/$(DESIGN)_final.v; \
	else \
		echo "ERROR: ECO Verilog file not found. Expected $(ECO_VERILOG) or $(DESIGN_DIR)/$(DESIGN)_final.v"; \
		echo "Please run 'make cts_eco DESIGN=$(DESIGN)' first."; \
		exit 1; \
	fi; \
	$(PYTHON) rename.py \
		--design $(DESIGN) \
		--final-v $$ECO_FILE \
		--map $(ECO_MAP) \
		--output $(RENAMED_VERILOG)
	@echo "✓ Renaming complete"

# --- Step 5: Routing ---
route: $(ROUTED_DEF)

$(ROUTED_DEF): $(RENAMED_VERILOG) $(ECO_MAP)
	@echo ">>> Step 5: Routing for $(DESIGN)..."
	$(PYTHON) make_def.py --design $(DESIGN) --map $(ECO_MAP)
	DESIGN_NAME=$(DESIGN) \
	BUILD_DIR=$(BUILD_DIR) \
	LIBERTY_FILE=$(LIBERTY_FILE) \
	LEF_FILE=$(LEF_FILE) \
	$(OPENROAD) -exit route.tcl
	@echo "✓ Routing complete"

# --- Step 6: STA ---
sta: $(ROUTED_DEF)
	@echo ">>> Step 6: Running Static Timing Analysis for $(DESIGN)..."
	DESIGN_NAME=$(DESIGN) \
	BUILD_DIR=$(BUILD_DIR) \
	LIBERTY_FILE=$(LIBERTY_FILE) \
	LEF_FILE=$(LEF_FILE) \
	$(OPENROAD) -exit sta.tcl
	@echo "✓ STA complete"

clean:
	rm -rf $(DESIGN_DIR)
	rm -f route.log