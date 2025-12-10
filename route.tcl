#!/usr/bin/env tclsh
#
# route.tcl - OpenROAD Routing Flow
# Member C - Task 3: Routing Script
#
# This script performs routing using OpenROAD.
# Uses environment variables for configuration.
#
# Usage:
#   export DESIGN_NAME=6502
#   export BUILD_DIR=build
#   export LIBERTY_FILE=tech/sky130_fd_sc_hd__tt_025C_1v80.lib
#   export LEF_FILE=tech/sky130_fd_sc_hd_merged.lef  # Use merged LEF!
#   openroad -exit route.tcl
#
# Note: The merged LEF file is required to avoid library conflicts.
#       Create it by running: python merge_lef.py
#

# ============================================================================
# Configuration from Environment Variables
# ============================================================================

# Design name (required)
if {![info exists ::env(DESIGN_NAME)]} {
    puts "ERROR: DESIGN_NAME environment variable not set"
    puts "Usage: export DESIGN_NAME=6502"
    exit 1
}
set design_name $::env(DESIGN_NAME)

# Build directory (default: "build")
if {![info exists ::env(BUILD_DIR)]} {
    set build_dir "build"
} else {
    set build_dir $::env(BUILD_DIR)
}

# Liberty file (required)
if {![info exists ::env(LIBERTY_FILE)]} {
    puts "ERROR: LIBERTY_FILE environment variable not set"
    puts "Usage: export LIBERTY_FILE=tech/sky130_fd_sc_hd__tt_025C_1v80.lib"
    exit 1
}
set liberty_file $::env(LIBERTY_FILE)

# LEF file (merged file containing BOTH technology and macro definitions)
# The merged file has proper ordering: SITE/LAYER/VIA definitions first, then MACROs
# This avoids the library conflict and "unknown site" warnings
if {![info exists ::env(LEF_FILE)]} {
    set lef_file "tech/sky130_fd_sc_hd_merged.lef"
} else {
    set lef_file $::env(LEF_FILE)
}

# ============================================================================
# File Paths
# ============================================================================

set verilog_file "${build_dir}/${design_name}/${design_name}_renamed.v"
set def_file "${build_dir}/${design_name}/${design_name}.def"
set routed_def_file "${build_dir}/${design_name}/${design_name}_routed.def"

# ============================================================================
# Main Routing Flow
# ============================================================================

puts "============================================================"
puts "Routing Flow for ${design_name}"
puts "============================================================"
puts ""
puts "Configuration:"
puts "  Design Name:    ${design_name}"
puts "  Build Directory: ${build_dir}"
puts "  Liberty File:   ${liberty_file}"
puts "  LEF File:       ${lef_file}"
puts ""

# ============================================================================
# Step 1: Reading Merged LEF file
# ============================================================================
# Using a merged LEF file that contains BOTH technology definitions (SITE, LAYER, VIA)
# AND macro/cell definitions in the correct order.
# This avoids the "unknown site unithd" and library conflict issues.

puts "Step 1: Reading Merged LEF file..."
puts ""
puts "  The merged LEF file contains:"
puts "    - SITE definitions (unithd, unithddbl)"
puts "    - LAYER definitions (li1, met1-met5, vias)"
puts "    - VIA/VIARULE definitions"
puts "    - MACRO/cell definitions (437 standard cells)"
puts ""

puts "  File: ${lef_file}"
if {![file exists ${lef_file}]} {
    puts "ERROR: Merged LEF file not found: ${lef_file}"
    puts ""
    puts "  To create the merged LEF file, run:"
    puts "    python merge_lef.py"
    puts ""
    puts "  This will combine:"
    puts "    - tech/sky130_fd_sc_hd.tlef (technology definitions)"
    puts "    - tech/sky130_fd_sc_hd.lef (macro definitions)"
    puts "  into: tech/sky130_fd_sc_hd_merged.lef"
    exit 1
}

# Read merged LEF file
if {[catch {read_lef ${lef_file}} result]} {
    puts "ERROR: Failed to read merged LEF: ${result}"
    puts "       File: ${lef_file}"
    exit 1
}

# Verify cells were loaded
set db_after_lef [ord::get_db]
set cell_count 0
if {$db_after_lef != "NULL"} {
    set libs_after_lef [$db_after_lef getLibs]
    foreach lib_lef $libs_after_lef {
        set masters_lef [$lib_lef getMasters]
        incr cell_count [llength $masters_lef]
    }
}

if {$cell_count > 0} {
    puts "    ✓ Cells loaded: $cell_count"
} else {
    puts "    ✗ ERROR: No cells loaded from LEF!"
    puts "      Check that ${lef_file} contains MACRO definitions"
    exit 1
}

# Verify layers were loaded
set tech_db [ord::get_db_tech]
if {$tech_db != "NULL"} {
    set all_layers [$tech_db getLayers]
    set layer_count [llength $all_layers]
    
    set routing_count 0
    set routing_names {}
    foreach layer_item $all_layers {
        if {[$layer_item getType] == "ROUTING"} {
            incr routing_count
            lappend routing_names [$layer_item getName]
        }
    }
    
    if {$layer_count > 0} {
        puts "    ✓ Layers loaded: $layer_count total"
        if {$routing_count > 0} {
            puts "    ✓ Routing layers: [join $routing_names {, }]"
        } else {
            puts "    ✗ WARNING: No routing layers found!"
        }
    } else {
        puts "    ✗ CRITICAL ERROR: No layers found!"
        puts "      The merged LEF file may be missing technology definitions."
        exit 1
    }
} else {
    puts "    ✗ ERROR: Could not access technology database!"
    exit 1
}
puts ""

# ============================================================================
# Step 2: Reading Liberty file...
# ============================================================================

puts "Step 2: Reading Liberty file..."
puts "  File: ${liberty_file}"
if {![file exists ${liberty_file}]} {
    puts "ERROR: Liberty file not found: ${liberty_file}"
    exit 1
}
read_liberty ${liberty_file}
puts "  ✓ Liberty file loaded"
puts ""

# ============================================================================
# Step 3: Reading DEF file...
# ============================================================================

puts "Step 3: Reading DEF file..."
puts "  File: ${def_file}"
if {![file exists ${def_file}]} {
    puts "ERROR: DEF file not found: ${def_file}"
    exit 1
}

# Verify cells are available before reading DEF
puts "  Verifying cells are loaded before reading DEF..."
set db_check [ord::get_db]
if {$db_check != "NULL"} {
    set libs_check [$db_check getLibs]
    set cell_count_check 0
    puts "    Checking libraries:"
    foreach lib_check $libs_check {
        set lib_name_check [$lib_check getName]
        set masters_check [$lib_check getMasters]
        set count_check [llength $masters_check]
        incr cell_count_check $count_check
        puts "      Library '$lib_name_check': $count_check cells"
        
        # If this library has cells, show a few examples
        if {$count_check > 0 && $count_check < 10} {
            puts "        Cells:"
            foreach master_check $masters_check {
                puts "          - [$master_check getName]"
            }
        } elseif {$count_check > 0} {
            puts "        First 5 cells:"
            set shown 0
            foreach master_check $masters_check {
                if {$shown < 5} {
                    puts "          - [$master_check getName]"
                    incr shown
                } else {
                    break
                }
            }
        }
    }
    puts "    Total cells in all libraries: $cell_count_check"
    
    if {$cell_count_check == 0} {
        puts ""
        puts "    ✗ ERROR: No cells loaded! Cannot read DEF file."
        puts ""
        puts "    Possible causes:"
        puts "      1. Merged LEF file doesn't contain MACRO definitions"
        puts "      2. LEF file is corrupted or incomplete"
        puts ""
        puts "    Try:"
        puts "      - Regenerate the merged LEF: python merge_lef.py"
        puts "      - Check that ${lef_file} contains MACRO definitions"
        exit 1
    }
}

# Read DEF with error handling
if {[catch {read_def ${def_file}} result]} {
    puts "ERROR: Failed to read DEF file: ${result}"
    puts ""
    puts "Common causes:"
    puts "  1. DEF file references cells not in the LEF library"
    puts "  2. Cell names in DEF don't match LEF cell names"
    puts ""
    puts "Check that:"
    puts "  - Merged LEF file contains all cells referenced in DEF"
    puts "  - Cell names match exactly (case-sensitive)"
    puts "  - Regenerate merged LEF if needed: python merge_lef.py"
    exit 1
}
puts "  ✓ DEF file loaded"

# Verify layers are still accessible after reading DEF
puts ""
puts "  Verifying layers are accessible after reading DEF..."
set tech_after_def [ord::get_db_tech]
if {$tech_after_def != "NULL"} {
    set layers_after_def [$tech_after_def getLayers]
    set layer_count_after_def [llength $layers_after_def]
    if {$layer_count_after_def > 0} {
        puts "    ✓ Found $layer_count_after_def layers after DEF read"
        
        # Show routing layers
        set routing_after_def {}
        foreach layer_def $layers_after_def {
            if {[$layer_def getType] == "ROUTING"} {
                lappend routing_after_def [$layer_def getName]
            }
        }
        if {[llength $routing_after_def] > 0} {
            puts "    ✓ Routing layers: [join $routing_after_def {, }]"
        }
    } else {
        puts "    ✗ WARNING: No layers found after reading DEF!"
        puts "    This may indicate a technology database access issue."
    }
} else {
    puts "    ✗ ERROR: Could not access technology database after DEF read!"
}

# ============================================================================
# Step 4: Define Routing Tracks using make_tracks
# ============================================================================
# Use OpenROAD's make_tracks command which reads track info from the LEF file
# and creates properly aligned track grids for all routing layers.

puts "Step 4: Setting up routing tracks..."

# Access technology database
set tech [ord::get_db_tech]
if {$tech == "NULL"} {
    puts "  ✗ ERROR: Could not access technology database!"
    exit 1
}

# Get layers and show routing layers
set all_layers [$tech getLayers]
set layer_count [llength $all_layers]
puts "  Found $layer_count layers in technology database"
puts "  Routing layers available:"
foreach layer_iter $all_layers {
    if {[$layer_iter getType] == "ROUTING"} {
        puts "    - [$layer_iter getName]"
    }
}

# Use make_tracks to create track grids from LEF layer definitions
# This ensures tracks align with the standard cell library
puts ""
puts "  Creating tracks using make_tracks command..."

# make_tracks creates routing tracks for all layers based on LEF pitch/offset
# Syntax: make_tracks layer -x_offset offset -x_pitch pitch -y_offset offset -y_pitch pitch
# Values in microns

# li1: local interconnect - PITCH 0.46/0.34, OFFSET 0.23/0.17
make_tracks li1 -x_offset 0.23 -x_pitch 0.46 -y_offset 0.17 -y_pitch 0.34
puts "    + li1 tracks created"

# met1: horizontal preferred - PITCH 0.34, OFFSET 0.17
make_tracks met1 -x_offset 0.17 -x_pitch 0.34 -y_offset 0.17 -y_pitch 0.34
puts "    + met1 tracks created"

# met2: vertical preferred - PITCH 0.46, OFFSET 0.23
make_tracks met2 -x_offset 0.23 -x_pitch 0.46 -y_offset 0.23 -y_pitch 0.46
puts "    + met2 tracks created"

# met3: horizontal preferred - PITCH 0.68, OFFSET 0.34
make_tracks met3 -x_offset 0.34 -x_pitch 0.68 -y_offset 0.34 -y_pitch 0.68
puts "    + met3 tracks created"

# met4: vertical preferred - PITCH 0.92, OFFSET 0.46
make_tracks met4 -x_offset 0.46 -x_pitch 0.92 -y_offset 0.46 -y_pitch 0.92
puts "    + met4 tracks created"

# met5: horizontal preferred - PITCH 3.4, OFFSET 1.7
make_tracks met5 -x_offset 1.7 -x_pitch 3.4 -y_offset 1.7 -y_pitch 3.4
puts "    + met5 tracks created"

puts "  ✓ Tracks initialized"
puts ""

# ============================================================================
# Step 5: Configure routing layers
# ============================================================================
# li1 (local interconnect) is very constrained and primarily for cell-internal
# routing. For inter-cell routing, prefer met1 and above.

puts "Step 5: Configuring routing layers..."

# Set the routing layers for signal nets (met1 through met5)
# This excludes li1 from inter-cell routing
set_routing_layers -signal met1-met5

# Optionally adjust layer usage to prefer lower metal layers
# This can help with congestion on higher layers
set_global_routing_layer_adjustment met1 0.5
set_global_routing_layer_adjustment met2 0.5
set_global_routing_layer_adjustment met3 0.3
set_global_routing_layer_adjustment met4 0.2
set_global_routing_layer_adjustment met5 0.1

puts "  Signal routing layers: met1-met5 (excluding li1)"
puts "  Layer adjustments applied"
puts ""

# ============================================================================
# Step 6: Running routing...
# ============================================================================

puts "Step 6: Running routing..."
puts "  This will route all nets in the design"

# Global Routing with verbose output
# -allow_congestion: proceed even with minor overflow (detailed route can fix it)
# -congestion_iterations: extra iterations to reduce congestion
puts "  Running global routing..."
if {[catch {global_route -verbose -allow_congestion -congestion_iterations 50} result]} {
    puts "  ✗ Global routing failed: ${result}"
    puts ""
    puts "  ERROR: Routing cannot proceed."
    puts "  Check the error message above for details."
    exit 1
}
puts "  ✓ Global routing complete"

# Save global routing result before attempting detailed routing
set global_routed_def "${build_dir}/${design_name}/${design_name}_global_routed.def"
puts "  Saving global routing result to $global_routed_def"
write_def $global_routed_def

# Detailed Routing with repair iterations
puts "  Running detailed routing..."
puts "  (This may take a while for large designs)"
puts "  NOTE: Some structured ASIC cells may have pin access issues."

# detailed_route options:
# -output_maze_log: output maze routing log for debugging
# -droute_end_iter: number of iterations for detailed route cleanup
# -or_seed: random seed for tie-breaking
set detailed_routing_failed 0
if {[catch {detailed_route -droute_end_iter 10} result]} {
    puts "  ✗ Detailed routing encountered errors: ${result}"
    puts ""
    puts "  WARNING: Some cells have pins that cannot be accessed by the router."
    puts "  This is common in structured ASIC designs where cell positions are fixed."
    puts ""
    puts "  The global routing result has been saved to:"
    puts "    $global_routed_def"
    puts ""
    set detailed_routing_failed 1
} else {
    puts "  ✓ Detailed routing complete"
}

if {$detailed_routing_failed} {
    puts "  Continuing with global-routed design (detailed routing skipped)"
}
puts ""

# ============================================================================
# Step 7: Write routed DEF
# ============================================================================

puts "Step 7: Writing routed DEF file..."
puts "  File: ${routed_def_file}"

# Create output directory if it doesn't exist
file mkdir [file dirname ${routed_def_file}]

write_def ${routed_def_file}

puts "  ✓ Routed DEF file written"
puts ""

# ============================================================================
# Final Summary
# ============================================================================

puts "============================================================"
if {$detailed_routing_failed} {
    puts "Routing Complete (Global Routing Only)"
    puts "============================================================"
    puts ""
    puts "WARNING: Detailed routing failed due to pin access issues."
    puts "         The output contains global routing guides only."
    puts ""
    puts "Output Files:"
    puts "  Global-routed DEF: ${global_routed_def}"
    puts "  (Final DEF also written but may have incomplete routing)"
} else {
    puts "Routing Complete (Full Detail Routing)"
    puts "============================================================"
    puts ""
    puts "Output Files:"
}
puts "  Routed DEF: ${routed_def_file}"
puts ""
puts "============================================================"

# Exit successfully
exit 0