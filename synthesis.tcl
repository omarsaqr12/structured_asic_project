# Yosys synthesis script for Structured ASIC Project
# Maps RTL to Sky130 standard cells
#
# Usage: yosys -c synthesis.tcl
# Or:    DESIGN=arith yosys -c synthesis.tcl

# Get design name from environment or use default
if {[info exists ::env(DESIGN)]} {
    set design_name $::env(DESIGN)
} else {
    set design_name "arith"
}

# Get top module name (usually same as design or with 'top_' prefix)
if {[info exists ::env(TOP_MODULE)]} {
    set top_module $::env(TOP_MODULE)
} else {
    set top_module "top_${design_name}"
}

# Get input file path from environment or use default location
if {[info exists ::env(VERILOG_INPUT)]} {
    set verilog_input $::env(VERILOG_INPUT)
} else {
    # Default location: build/{design_name}/{design_name}_renamed.v
    set verilog_input "build/${design_name}/${design_name}_renamed.v"
    
    # Check if input file exists
    if {![file exists $verilog_input]} {
        puts "ERROR: Input Verilog file not found: $verilog_input"
        puts ""
        puts "Please either:"
        puts "  1. Ensure the file exists at: $verilog_input"
        puts "  2. Set VERILOG_INPUT environment variable to specify a custom path"
        exit 1
    }
}

# Output paths
set verilog_output "designs/${design_name}_mapped.v"
set json_output "designs/${design_name}_mapped.json"
set liberty_file "tech/sky130_fd_sc_hd__tt_025C_1v80.lib"

# Check if liberty file exists
if {![file exists $liberty_file]} {
    puts "ERROR: Liberty file not found: $liberty_file"
    exit 1
}

puts "=========================================="
puts "Yosys Synthesis for ${design_name}"
puts "=========================================="
puts "Input:  $verilog_input"
puts "Output: $verilog_output"
puts "        $json_output"
puts "Top module: $top_module"
puts "=========================================="
puts ""

# After reading your RTL
puts "Step 1: Reading Verilog..."
read_verilog $verilog_input

puts "Step 2: Checking hierarchy..."
hierarchy -check -top $top_module

# Synthesize
puts "Step 3: Synthesizing..."
synth -top $top_module

# CRITICAL: Map ALL cells to Sky130 library
# This replaces internal cells with actual Sky130 cells
puts "Step 4: Mapping flip-flops to Sky130 library..."
dfflibmap -liberty $liberty_file

puts "Step 5: Technology mapping with ABC..."
abc -liberty $liberty_file

# IMPORTANT: Clean and flatten to remove unmapped cells
puts "Step 6: Cleaning unmapped cells..."
opt_clean -purge

puts "Step 7: Flattening design..."
flatten

puts "Step 8: Final cleanup..."
opt_clean -purge

# Check for unmapped cells
puts ""
puts "Step 9: Design statistics..."
stat

puts ""
puts "Step 10: Writing output files..."

# Write output
write_verilog -noattr -noexpr $verilog_output
write_json $json_output

puts ""
puts "=========================================="
puts "Synthesis complete!"
puts "  Verilog: $verilog_output"
puts "  JSON:    $json_output"
puts "=========================================="

