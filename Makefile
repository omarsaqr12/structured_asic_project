# Makefile for Structured ASIC Project - Phase 3
# Supports placement, CTS, ECO generation, visualization, and testing

# Default design
DESIGN ?= 6502

# Directories
BUILD_DIR = build
DESIGN_DIR = $(BUILD_DIR)/$(DESIGN)
DESIGNS_DIR = designs
FABRIC_DIR = fabric

# File paths
DESIGN_JSON = $(DESIGNS_DIR)/$(DESIGN)_mapped.json
PLACEMENT_MAP = $(DESIGN_DIR)/$(DESIGN).map
PLACEMENT_JSON = $(DESIGN_DIR)/$(DESIGN)_placement.json
FINAL_VERILOG = test_eco_$(DESIGN).v
FINAL_JSON = test_eco_$(DESIGN).json
RENAMED_VERILOG = $(DESIGN_DIR)/$(DESIGN)_renamed.v

# CTS paths
CTS_TREE_HTREE = $(DESIGN_DIR)/cts_with_manager/best_htree/$(DESIGN)_cts_tree.json
CTS_TREE_XTREE = $(DESIGN_DIR)/cts_with_manager/best_xtree/$(DESIGN)_cts_tree.json
CTS_COMPARISON_JSON = $(DESIGN_DIR)/cts_comparison.json
CTS_COMPARISON_TXT = $(DESIGN_DIR)/cts_comparison.txt
CTS_COMPARISON_PNG = $(DESIGN_DIR)/cts_tree_comparison.png
CTS_VISUALIZATION = $(DESIGN_DIR)/$(DESIGN)_cts_tree.png

# Python command
PYTHON = python

.PHONY: all place eco cts_viz cts_sim cts_report test_eco test_unit test_acceptance clean help

# Default target
all: eco

# Help target
help:
	@echo "Available targets:"
	@echo "  make place DESIGN=6502     - Run placement (Phase 2)"
	@echo "  make eco DESIGN=6502      - Generate ECO netlist (requires placement)"
	@echo "  make cts_viz DESIGN=6502  - Generate CTS tree visualization"
	@echo "  make cts_sim DESIGN=6502  - Run CTS simulation (H-Tree vs X-Tree)"
	@echo "  make cts_report DESIGN=6502 - Generate CTS comparison report"
	@echo "  make test_eco              - Run all tests (unit + acceptance)"
	@echo "  make test_unit             - Run unit tests only"
	@echo "  make test_acceptance       - Run acceptance tests only"
	@echo "  make clean                 - Clean generated files"

# Placement target (Phase 2 - depends on placer.py)
place: $(PLACEMENT_MAP) $(PLACEMENT_JSON)
	@echo "✓ Placement complete: $(PLACEMENT_MAP)"

$(PLACEMENT_MAP) $(PLACEMENT_JSON): $(DESIGN_JSON)
	@echo "Running placement for $(DESIGN)..."
	$(PYTHON) placer.py --design $(DESIGN_JSON) \
		--fabric-cells $(FABRIC_DIR)/fabric_cells.yaml \
		--pins $(FABRIC_DIR)/pins.yaml
	@echo "✓ Placement files generated"

# ECO generation target (Phase 3)
eco: $(RENAMED_VERILOG)
	@echo "✓ ECO generation complete: $(RENAMED_VERILOG)"

$(FINAL_VERILOG): $(PLACEMENT_MAP) $(PLACEMENT_JSON) $(DESIGN_JSON)
	@echo "Generating ECO for $(DESIGN)..."
	$(PYTHON) eco_generator.py \
		--placement $(PLACEMENT_JSON) \
		--design $(DESIGN_JSON) \
		--fabric-cells $(FABRIC_DIR)/fabric_cells.yaml \
		--placement-map $(PLACEMENT_MAP) \
		--enable-cts \
		--cts-tree-type h \
		--output-json $(FINAL_JSON) \
		--output-verilog $(FINAL_VERILOG)
	@echo "✓ ECO Verilog generated: $(FINAL_VERILOG)"

$(RENAMED_VERILOG): $(FINAL_VERILOG) $(PLACEMENT_MAP)
	@echo "Renaming instances in Verilog for $(DESIGN)..."
	$(PYTHON) rename_helper.py \
		--design $(DESIGN) \
		--verilog $(FINAL_VERILOG) \
		--map $(PLACEMENT_MAP) \
		--output $(RENAMED_VERILOG)
	@echo "✓ Renamed Verilog generated: $(RENAMED_VERILOG)"

# CTS visualization target
cts_viz: $(CTS_VISUALIZATION)
	@echo "✓ CTS visualization generated: $(CTS_VISUALIZATION)"

$(CTS_VISUALIZATION): $(CTS_TREE_HTREE)
	@echo "Generating CTS tree visualization for $(DESIGN)..."
	$(PYTHON) visualize.py cts \
		--design $(DESIGN) \
		--cts-tree $(CTS_TREE_HTREE) \
		--output $(CTS_VISUALIZATION)
	@echo "✓ CTS visualization saved"

# CTS simulation target
cts_sim: $(CTS_COMPARISON_JSON)
	@echo "✓ CTS simulation complete: $(CTS_COMPARISON_JSON)"

$(CTS_COMPARISON_JSON): $(CTS_TREE_HTREE) $(CTS_TREE_XTREE)
	@echo "Running CTS simulation for $(DESIGN)..."
	$(PYTHON) cts_simulator.py \
		--htree $(CTS_TREE_HTREE) \
		--xtree $(CTS_TREE_XTREE) \
		--design $(DESIGN) \
		--output $(CTS_COMPARISON_JSON)
	@echo "✓ CTS comparison data saved"

# CTS comparison report target
cts_report: $(CTS_COMPARISON_TXT) $(CTS_COMPARISON_PNG)
	@echo "✓ CTS comparison report generated"

$(CTS_COMPARISON_TXT) $(CTS_COMPARISON_PNG): $(CTS_COMPARISON_JSON)
	@echo "Generating CTS comparison report for $(DESIGN)..."
	$(PYTHON) comparison_report.py \
		--design $(DESIGN) \
		--input $(CTS_COMPARISON_JSON) \
		--output-txt $(CTS_COMPARISON_TXT) \
		--output-png $(CTS_COMPARISON_PNG)
	@echo "✓ Comparison report generated"

# Test targets
test_eco: test_unit test_acceptance
	@echo "✓ All tests passed"

test_unit:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest test_eco_generator.py -v
	@echo "✓ Unit tests passed"

test_acceptance:
	@echo "Running acceptance tests..."
	$(PYTHON) -m pytest test_acceptance.py -v
	@echo "✓ Acceptance tests passed"

# Clean target
clean:
	@echo "Cleaning generated files..."
	rm -f $(FINAL_VERILOG) $(FINAL_JSON) $(RENAMED_VERILOG)
	rm -f $(CTS_COMPARISON_JSON) $(CTS_COMPARISON_TXT) $(CTS_COMPARISON_PNG)
	rm -f $(CTS_VISUALIZATION)
	@echo "✓ Clean complete"

# Clean all (including placement)
clean_all: clean
	@echo "Cleaning all build artifacts..."
	rm -rf $(BUILD_DIR)
	@echo "✓ All build artifacts removed"

