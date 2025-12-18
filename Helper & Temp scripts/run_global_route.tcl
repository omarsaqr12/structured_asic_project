#!/usr/bin/env tclsh
#
# run_global_route.tcl - OpenROAD Global Routing Script
# Runs global routing and reports congestion analysis
#
# Usage:
#   export DESIGN_NAME=6502
#   export BUILD_DIR=build
#   export LIBERTY_FILE=tech/sky130_fd_sc_hd__tt_025C_1v80.lib
#   export LEF_FILE=tech/sky130_fd_sc_hd_merged.lef
#   openroad run_global_route.tcl
#
# Note: The merged LEF file is required to avoid library conflicts.
#       This script performs global routing and generates congestion reports.
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

set def_file "${build_dir}/${design_name}/${design_name}_global.def"
set global_routed_def "${build_dir}/${design_name}/${design_name}_global_routed.def"
set congestion_report_file "${build_dir}/${design_name}/${design_name}_congestion.rpt"

# ============================================================================
# Main Global Routing Flow
# ============================================================================

puts "============================================================"
puts "Global Routing Flow for ${design_name}"
puts "============================================================"
puts ""
puts "Configuration:"
puts "  Design Name:    ${design_name}"
puts "  Build Directory: ${build_dir}"
puts "  Liberty File:   ${liberty_file}"
puts "  LEF File:       ${lef_file}"
puts "  Input DEF:      ${def_file}"
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
    foreach lib_check $libs_check {
        set masters_check [$lib_check getMasters]
        set count_check [llength $masters_check]
        incr cell_count_check $count_check
    }
    
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
puts ""

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

# met5: horizontal preferred - PITCH 3.40, OFFSET 1.70
make_tracks met5 -x_offset 1.70 -x_pitch 3.40 -y_offset 1.70 -y_pitch 3.40
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
# Step 6: Place Pins
# ============================================================================
# Pins must be placed on the die boundary aligned to routing tracks before routing.
# According to Pin_Placement.md, pins should be placed using place_pins command.
# For sky130: met2 is horizontal preferred, met3 is vertical preferred.

puts "Step 6: Placing pins on die boundary..."
puts "  Pins will be placed on routing track grid for proper connectivity"

# Check if pins already exist in the design
set block [ord::get_db_block]
set pins_exist 0
if {$block != "NULL"} {
    set pins [$block getBTerms]
    set pin_count [llength $pins]
    if {$pin_count > 0} {
        puts "  Found $pin_count pins in design"
        set pins_exist 1
        
        # Check if pins are already placed by checking for BPins (boundary pins)
        # BPins indicate that a pin has been placed on the die boundary
        set placed_count 0
        foreach pin $pins {
            set bpins [$pin getBPins]
            if {[llength $bpins] > 0} {
                incr placed_count
            }
        }
        
        if {$placed_count == $pin_count} {
            puts "  All pins are already placed ($placed_count/$pin_count)"
            puts "  Re-placing pins to ensure proper track alignment..."
        } elseif {$placed_count > 0} {
            puts "  $placed_count of $pin_count pins are placed"
            puts "  Placing remaining pins and re-aligning all pins to tracks..."
        } else {
            puts "  No pins are placed yet"
            puts "  Placing all pins on track grid..."
        }
    } else {
        puts "  No pins found in design"
        puts "  Skipping pin placement"
        set pins_exist 0
    }
}

if {$pins_exist} {
    # Clear any existing pin constraints to allow place_pins to work properly
    # According to Pin_Placement.md, clear_io_pin_constraints clears all constraints
    if {[catch {clear_io_pin_constraints} result]} {
        puts "  Note: Could not clear pin constraints (may not be critical): ${result}"
    } else {
        puts "  Cleared existing pin constraints"
    }
    
    # Set pin offset for global routing (optional, helps with routing)
    # Pin offset is the distance from die boundary
    # Default is usually 0, but setting a small offset can help
    if {[catch {set_pin_offset 0} result]} {
        puts "  Note: Could not set pin offset (may not be critical)"
    }
    
    # Place pins using OpenROAD pin placer
    # Pins are written as PLACED (not FIXED) in DEF file, so place_pins can adjust them
    # For sky130 technology:
    # - Horizontal pins (top/bottom edges): met2 (horizontal preferred layer)
    # - Vertical pins (left/right edges): met3 (vertical preferred layer)
    # - min_distance: minimum spacing between pins (in tracks when -min_distance_in_tracks is used)
    #   When using -min_distance_in_tracks, this must be an INTEGER (number of tracks)
    #   Using 1 track spacing for minimum pin-to-pin distance
    # - corner_avoidance: avoid placing pins near corners (5 microns)
    # - min_distance_in_tracks: use track-based spacing instead of microns
    
    if {[catch {
        place_pins \
            -hor_layers met2 \
            -ver_layers met3 \
            -min_distance 1 \
            -min_distance_in_tracks \
            -corner_avoidance 5.0
    } result]} {
        puts "  ✗ Pin placement failed: ${result}"
        puts ""
        puts "  WARNING: Pin placement encountered errors."
        puts "  This may cause routing issues if pins are not on track grid."
        puts ""
        puts "  Possible causes:"
        puts "    1. Insufficient space on die boundary"
        puts "    2. Pin coordinates are not on routing tracks"
        puts "    3. Pin geometries conflict with routing layer constraints"
        puts ""
        puts "  Continuing with existing pin positions..."
        puts "  If routing fails, check that pins are aligned to tracks."
    } else {
        puts "  ✓ Pins placed successfully"
        puts "    Horizontal pins on: met2"
        puts "    Vertical pins on: met3"
        puts "    Minimum spacing: 1 track"
        puts "    Corner avoidance: 5.0 microns"
    }
} else {
    puts "  ⚠ No pins to place - design may not have I/O pins"
}
puts ""

# ============================================================================
# Step 7: Global Routing
# ============================================================================

puts "Step 7: Running global routing..."
puts "  This will generate routing guides and congestion analysis"

# Global Routing with verbose output
# -allow_congestion: proceed even with minor overflow (detailed route can fix it)
# -congestion_iterations: extra iterations to reduce congestion
# -congestion_report_file: file to save congestion report

puts "  Running global routing..."
puts "  Congestion report will be saved to: ${congestion_report_file}"

if {[catch {
    global_route \
        -verbose \
        -allow_congestion \
        -congestion_iterations 50 \
        -congestion_report_file ${congestion_report_file}
} result]} {
    puts "  ✗ Global routing failed: ${result}"
    exit 1
}
puts "  ✓ Global routing complete"
puts "  ✓ Congestion report saved: ${congestion_report_file}"
puts ""

# Save global routing result
puts "  Saving global routing result..."
file mkdir [file dirname ${global_routed_def}]
write_def ${global_routed_def}
puts "  ✓ Global routed DEF saved: ${global_routed_def}"
puts ""

# ============================================================================
# Step 8: Congestion Report Analysis
# ============================================================================

puts "Step 8: Analyzing congestion report..."
puts ""

if {![file exists ${congestion_report_file}]} {
    puts "  ⚠ WARNING: Congestion report file not found: ${congestion_report_file}"
    puts "  Global routing may not have generated the report."
} else {
    puts "  Reading congestion report: ${congestion_report_file}"
    puts ""
    
    # Read and display congestion report
    set report_fp [open ${congestion_report_file} r]
    set report_content [read $report_fp]
    close $report_fp
    
    # Display the congestion report
    puts "  ============================================================"
    puts "  Congestion Report Summary"
    puts "  ============================================================"
    puts ""
    puts $report_content
    puts "  ============================================================"
    puts ""
    
    # Parse and summarize key metrics from the report
    # Look for common congestion report patterns
    set lines [split $report_content "\n"]
    set total_overflow 0
    set max_h_overflow 0
    set max_v_overflow 0
    set congestion_found 0
    
    foreach line $lines {
        # Look for overflow information
        if {[string match "*Overflow*" $line] || [string match "*overflow*" $line]} {
            set congestion_found 1
            puts "  ⚠ Congestion detected in report"
        }
        # Look for usage percentages
        if {[string match "*Usage*" $line] || [string match "*usage*" $line]} {
            puts "  $line"
        }
    }
    
    if {!$congestion_found} {
        puts "  ✓ No significant congestion detected"
    }
    puts ""
}

# ============================================================================
# Step 9: Design Statistics
# ============================================================================

puts "Step 9: Design Statistics..."
set block [ord::get_db_block]
if {$block != "NULL"} {
    set insts [$block getInsts]
    set nets [$block getNets]
    set pins [$block getBTerms]
    
    puts "  Components: [llength $insts]"
    puts "  Nets: [llength $nets]"
    puts "  I/O Pins: [llength $pins]"
    
    # Check if design has global routing guides
    set nets_with_guides 0
    foreach net $nets {
        set guides [$net getGuides]
        if {[llength $guides] > 0} {
            incr nets_with_guides
        }
    }
    puts "  Nets with Global Routing Guides: $nets_with_guides / [llength $nets]"
}
puts ""

# ============================================================================
# Step 10: Launch GUI for Visualization
# ============================================================================


puts ""
puts "============================================================"
puts "Global Routing Complete"
puts "============================================================"
puts ""
puts "Output Files:"
puts "  Global-routed DEF: ${global_routed_def}"
puts "  Congestion Report: ${congestion_report_file}"
puts ""
puts "The design has been globally routed with routing guides."
puts "Use GUI commands to visualize routing guides and congestion."
puts ""

