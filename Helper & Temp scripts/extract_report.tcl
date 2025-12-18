#!/usr/bin/env tclsh
#
# extract_report.tcl - Extract Parasitics and Report Congestion
#
# This script takes routed DEF files and generates:
#   1. SPEF files (parasitic extraction)
#   2. Congestion reports
#
# Usage:
#   export DESIGN_NAME=6502
#   export BUILD_DIR=build
#   export LIBERTY_FILE=tech/sky130_fd_sc_hd__tt_025C_1v80.lib
#   export LEF_FILE=tech/sky130_fd_sc_hd_merged.lef
#   export DEF_FILE=build/6502/6502_routed.def
#   export EXTRACTION_MODEL_FILE=tech/sky130_rcx_rules.calibre  # Optional
#   openroad -exit extract_report.tcl
#

# ============================================================================
# Configuration from Environment Variables
# ============================================================================

if {![info exists ::env(DESIGN_NAME)]} {
    puts "ERROR: DESIGN_NAME environment variable not set"
    exit 1
}
set design_name $::env(DESIGN_NAME)

if {![info exists ::env(BUILD_DIR)]} {
    set build_dir "build"
} else {
    set build_dir $::env(BUILD_DIR)
}

if {![info exists ::env(LIBERTY_FILE)]} {
    puts "ERROR: LIBERTY_FILE environment variable not set"
    exit 1
}
set liberty_file $::env(LIBERTY_FILE)

if {![info exists ::env(LEF_FILE)]} {
    set lef_file "tech/sky130_fd_sc_hd_merged.lef"
} else {
    set lef_file $::env(LEF_FILE)
}

if {![info exists ::env(EXTRACTION_MODEL_FILE)]} {
    set extraction_model_file ""
} else {
    set extraction_model_file $::env(EXTRACTION_MODEL_FILE)
}

if {![info exists ::env(DEF_FILE)]} {
    set def_file "${build_dir}/${design_name}/${design_name}_routed.def"
} else {
    set def_file $::env(DEF_FILE)
}

# Determine routing type from filename
set def_basename [file tail ${def_file}]
if {[string match "*global_routed*" ${def_basename}]} {
    set routing_type "global"
} else {
    set routing_type "detailed"
}

set output_dir "${build_dir}/${design_name}"
set congestion_report_file "${output_dir}/${design_name}_congestion.rpt"
set spef_file "${output_dir}/${design_name}.spef"

# ============================================================================
# Main Flow
# ============================================================================

puts "============================================================"
puts "Extract Parasitics and Report Congestion"
puts "============================================================"
puts ""
puts "Configuration:"
puts "  Design:         ${design_name}"
puts "  DEF File:       ${def_file}"
puts "  Routing Type:   ${routing_type}"
puts "  Liberty:        ${liberty_file}"
puts "  LEF:            ${lef_file}"
if {${extraction_model_file} != ""} {
    puts "  RC Model:       ${extraction_model_file}"
}
puts ""
puts "Output Files:"
puts "  Congestion:     ${congestion_report_file}"
puts "  SPEF:           ${spef_file}"
puts ""

# ============================================================================
# Step 1: Read LEF
# ============================================================================

puts "Step 1: Reading LEF file..."
if {![file exists ${lef_file}]} {
    puts "ERROR: LEF file not found: ${lef_file}"
    exit 1
}
if {[catch {read_lef ${lef_file}} result]} {
    puts "ERROR: Failed to read LEF: ${result}"
    exit 1
}
puts "  ✓ LEF loaded"

# ============================================================================
# Step 2: Read Liberty
# ============================================================================

puts "Step 2: Reading Liberty file..."
if {![file exists ${liberty_file}]} {
    puts "ERROR: Liberty file not found: ${liberty_file}"
    exit 1
}
read_liberty ${liberty_file}
puts "  ✓ Liberty loaded"

# ============================================================================
# Step 3: Read DEF
# ============================================================================

puts "Step 3: Reading DEF file..."
if {![file exists ${def_file}]} {
    puts "ERROR: DEF file not found: ${def_file}"
    exit 1
}
if {[catch {read_def ${def_file}} result]} {
    puts "ERROR: Failed to read DEF: ${result}"
    exit 1
}
puts "  ✓ DEF loaded"
puts ""

# ============================================================================
# Step 4: Report Congestion
# ============================================================================

puts "Step 4: Generating congestion report..."

# Create output directory
file mkdir ${output_dir}

# Open congestion report file
set congestion_fd [open ${congestion_report_file} w]

puts $congestion_fd "=================================="
puts $congestion_fd "Congestion Report"
puts $congestion_fd "=================================="
puts $congestion_fd "Design: ${design_name}"
puts $congestion_fd "DEF File: ${def_file}"
puts $congestion_fd "Routing Type: ${routing_type}"
puts $congestion_fd "Generated: [clock format [clock seconds]]"
puts $congestion_fd ""
puts $congestion_fd "=================================="
puts $congestion_fd ""

# For global routing: use report_wire_length and check routing metrics
# For detailed routing: analyze the actual routing
if {${routing_type} == "global"} {
    puts $congestion_fd "Global Routing Statistics:"
    puts $congestion_fd "--------------------------"
    puts $congestion_fd ""
    
    # Try to get routing metrics from global router
    if {[catch {
        # Get block
        set block [ord::get_db_block]
        if {$block != "NULL"} {
            # Get GCells if available
            set gcell_grid [$block getGCellGrid]
            if {$gcell_grid != "NULL"} {
                puts $congestion_fd "GCell Grid: [format "%d x %d" [$gcell_grid getXGridSize] [$gcell_grid getYGridSize]]"
            }
            
            # Report nets
            set nets [$block getNets]
            set total_nets [llength $nets]
            puts $congestion_fd "Total Nets: ${total_nets}"
            puts $congestion_fd ""
        }
    } result]} {
        puts $congestion_fd "Note: Could not extract detailed congestion metrics"
        puts $congestion_fd "Error: ${result}"
    }
    
} else {
    # Detailed routing
    puts $congestion_fd "Detailed Routing Statistics:"
    puts $congestion_fd "----------------------------"
    puts $congestion_fd ""
    
    # Get detailed routing statistics
    if {[catch {
        set block [ord::get_db_block]
        if {$block != "NULL"} {
            set nets [$block getNets]
            set net_count 0
            set routed_net_count 0
            
            # Count nets
            foreach net $nets {
                incr net_count
                set wire [$net getWire]
                if {$wire != "NULL"} {
                    incr routed_net_count
                }
            }
            
            puts $congestion_fd "Total Nets: ${net_count}"
            puts $congestion_fd "Routed Nets: ${routed_net_count}"
            puts $congestion_fd ""
        }
    } result]} {
        puts $congestion_fd "Note: Could not extract all routing metrics"
        puts $congestion_fd "Error: ${result}"
    }
}

# Get wire length statistics by iterating through nets
puts $congestion_fd ""
puts $congestion_fd "Wire Length Statistics:"
puts $congestion_fd "-----------------------"
puts $congestion_fd ""

if {[catch {
    set block [ord::get_db_block]
    if {$block != "NULL"} {
        set nets [$block getNets]
        
        # Use report_net_length which is more reliable
        # We'll get summary statistics
        set routed_count 0
        set net_count 0
        
        foreach net $nets {
            incr net_count
            set wire [$net getWire]
            if {$wire != "NULL"} {
                incr routed_count
            }
        }
        
        puts $congestion_fd "Total Nets: ${net_count}"
        puts $congestion_fd "Nets with Routing: ${routed_count}"
        
        if {$routed_count > 0} {
            set routed_pct [expr {($routed_count * 100.0) / $net_count}]
            puts $congestion_fd "Routing Completion: [format "%.1f" ${routed_pct}]%"
        }
        
        puts $congestion_fd ""
        puts $congestion_fd "Note: For detailed wire length per net, use:"
        puts $congestion_fd "  report_net -connections <net_name>"
        puts $congestion_fd ""
        
        puts "  Basic routing statistics collected"
    }
} result]} {
    puts $congestion_fd "Note: Could not calculate routing statistics"
    puts $congestion_fd "Error: ${result}"
    puts "  Warning: Could not calculate routing statistics"
}

close $congestion_fd

# Append note about viewing in GUI
set congestion_fd [open ${congestion_report_file} a]
puts $congestion_fd ""
puts $congestion_fd "=================================="
puts $congestion_fd "Additional Information"
puts $congestion_fd "=================================="
puts $congestion_fd ""
puts $congestion_fd "For detailed congestion visualization, use OpenROAD GUI:"
puts $congestion_fd "  openroad -gui"
puts $congestion_fd "  File -> Read DEF -> ${def_file}"
puts $congestion_fd "  Tools -> Heat Maps -> Routing Congestion"
puts $congestion_fd ""
puts $congestion_fd "=================================="
puts $congestion_fd "End of Congestion Report"
puts $congestion_fd "=================================="
close $congestion_fd

puts ""
puts "  ✓ Congestion report saved: ${congestion_report_file}"
puts ""

# ============================================================================
# Step 5: Extract Parasitics
# ============================================================================

puts "Step 5: Extracting parasitics..."

# Check for extraction model file
if {${extraction_model_file} != ""} {
    if {![file exists ${extraction_model_file}]} {
        puts "  ✗ RC model file not found: ${extraction_model_file}"
        puts "  Attempting LEF-based extraction instead..."
        set use_lef_res 1
    } else {
        puts "  Using RC model: ${extraction_model_file}"
        set use_lef_res 0
    }
} else {
    puts "  ⚠ No RC model specified, using LEF-based extraction"
    set use_lef_res 1
}

# Perform extraction
set extraction_success 0
if {${use_lef_res}} {
    # LEF-based extraction (approximate)
    # Note: OpenRCX may still try to open a model file even with -lef_res
    # We'll create a minimal dummy rules file if needed
    if {[catch {extract_parasitics -lef_res} result]} {
        if {[string match "*Can't open extraction model file*" $result]} {
            puts "  ⚠ OpenRCX requires a rules file even for LEF extraction"
            puts "  Creating minimal extraction rules file..."
            
            # Create a minimal rules file
            set dummy_rules "${output_dir}/minimal_extraction.rules"
            set rules_fd [open ${dummy_rules} w]
            puts $rules_fd "# Minimal extraction rules for LEF-based extraction"
            puts $rules_fd "# OpenRCX requires this file even when using -lef_res"
            close $rules_fd
            
            # Try again with the dummy rules file
            if {[catch {extract_parasitics -ext_model_file ${dummy_rules} -lef_res} result2]} {
                puts "  ✗ LEF-based extraction failed: ${result2}"
            } else {
                puts "  ✓ LEF-based extraction complete (approximate values)"
                set extraction_success 1
            }
        } else {
            puts "  ✗ LEF-based extraction failed: ${result}"
        }
    } else {
        puts "  ✓ LEF-based extraction complete (approximate values)"
        set extraction_success 1
    }
} else {
    # Full RC extraction
    if {[catch {extract_parasitics -ext_model_file ${extraction_model_file}} result]} {
        puts "  ✗ RC extraction failed: ${result}"
        puts "  Falling back to LEF-based extraction..."
        
        # Try LEF-based as fallback
        set dummy_rules "${output_dir}/minimal_extraction.rules"
        if {![file exists ${dummy_rules}]} {
            set rules_fd [open ${dummy_rules} w]
            puts $rules_fd "# Minimal extraction rules for LEF-based extraction"
            close $rules_fd
        }
        
        if {[catch {extract_parasitics -ext_model_file ${dummy_rules} -lef_res} result2]} {
            puts "  ✗ LEF-based extraction also failed: ${result2}"
        } else {
            puts "  ✓ LEF-based extraction complete (approximate values)"
            set extraction_success 1
        }
    } else {
        puts "  ✓ RC extraction complete"
        set extraction_success 1
    }
}

# Write SPEF file
if {${extraction_success}} {
    puts ""
    puts "Step 6: Writing SPEF file..."
    if {[catch {write_spef ${spef_file}} result]} {
        puts "  ✗ Failed to write SPEF: ${result}"
    } else {
        puts "  ✓ SPEF file saved: ${spef_file}"
        if {${use_lef_res}} {
            puts "  ⚠ Note: SPEF contains approximate values (LEF-based only)"
        }
    }
} else {
    puts ""
    puts "  ✗ Skipping SPEF generation due to extraction failure"
    puts ""
    puts "  To enable full parasitic extraction, provide an RC model file:"
    puts "    export EXTRACTION_MODEL_FILE=tech/sky130_rcx_rules.calibre"
}

# ============================================================================
# Summary
# ============================================================================

puts ""
puts "============================================================"
puts "Extraction and Reporting Complete"
puts "============================================================"
puts ""
puts "Generated Files:"
if {[file exists ${congestion_report_file}]} {
    puts "  ✓ ${congestion_report_file}"
} else {
    puts "  ✗ Congestion report (failed)"
}
if {[file exists ${spef_file}]} {
    puts "  ✓ ${spef_file}"
    if {${use_lef_res}} {
        puts "    (approximate - LEF-based values only)"
    }
} else {
    puts "  ✗ SPEF file (extraction failed)"
}
puts ""
puts "Next Steps:"
puts "  1. Review congestion report: ${congestion_report_file}"
puts "  2. Use SPEF in STA: sta.tcl with read_spef ${spef_file}"
puts "  3. Generate congestion heatmap with visualize.py"
puts ""
puts "============================================================"

exit 0