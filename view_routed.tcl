#!/usr/bin/env tclsh
#
# view_routed.tcl - OpenROAD Visualization Script for Detailed Routing
# Loads a pre-routed DEF file and opens the GUI for visualization
#
# Usage:
#   export DESIGN_NAME=6502
#   export BUILD_DIR=build
#   export LIBERTY_FILE=tech/sky130_fd_sc_hd__tt_025C_1v80.lib
#   export LEF_FILE=tech/sky130_fd_sc_hd_merged.lef
#   openroad -gui view_routed.tcl
#
# Note: The merged LEF file is required to avoid library conflicts.
#       This script loads an already-routed DEF file for visualization only.

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
if {![info exists ::env(LEF_FILE)]} {
    set lef_file "tech/sky130_fd_sc_hd_merged.lef"
} else {
    set lef_file $::env(LEF_FILE)
}

# ============================================================================
# File Paths
# ============================================================================

# Read the already-routed DEF file
set routed_def_file "${build_dir}/${design_name}/${design_name}_routed.def"

# ============================================================================
# Main Visualization Flow
# ============================================================================

puts "============================================================"
puts "Loading Routed Design for Visualization: ${design_name}"
puts "============================================================"
puts ""
puts "Configuration:"
puts "  Design Name:    ${design_name}"
puts "  Build Directory: ${build_dir}"
puts "  Liberty File:   ${liberty_file}"
puts "  LEF File:       ${lef_file}"
puts "  Routed DEF:     ${routed_def_file}"
puts ""

# ============================================================================
# Step 1: Reading Merged LEF file
# ============================================================================

puts "Step 1: Reading Merged LEF file..."
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
    puts "  ✓ Cells loaded: $cell_count"
} else {
    puts "  ✗ ERROR: No cells loaded from LEF!"
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
        puts "  ✓ Layers loaded: $layer_count total"
        if {$routing_count > 0} {
            puts "  ✓ Routing layers: [join $routing_names {, }]"
        } else {
            puts "  ✗ WARNING: No routing layers found!"
        }
    } else {
        puts "  ✗ CRITICAL ERROR: No layers found!"
        puts "      The merged LEF file may be missing technology definitions."
        exit 1
    }
} else {
    puts "  ✗ ERROR: Could not access technology database!"
    exit 1
}
puts ""

# ============================================================================
# Step 2: Reading Liberty file
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
# Step 3: Reading Routed DEF file
# ============================================================================

puts "Step 3: Reading Routed DEF file..."
puts "  File: ${routed_def_file}"
if {![file exists ${routed_def_file}]} {
    puts "ERROR: Routed DEF file not found: ${routed_def_file}"
    puts ""
    puts "  Expected file: ${routed_def_file}"
    puts "  Make sure the design has been routed first."
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
if {[catch {read_def ${routed_def_file}} result]} {
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
puts ""

# ============================================================================
# Step 4: Design Statistics
# ============================================================================

puts "Step 4: Design Statistics..."
set block [ord::get_db_block]
if {$block != "NULL"} {
    set insts [$block getInsts]
    set nets [$block getNets]
    set pins [$block getBTerms]
    
    puts "  Components: [llength $insts]"
    puts "  Nets: [llength $nets]"
    puts "  I/O Pins: [llength $pins]"
    
    # Check if design has routing
    set routed_nets 0
    foreach net $nets {
        set wires [$net getSWires]
        if {[llength $wires] > 0} {
            incr routed_nets
        }
    }
    puts "  Routed Nets: $routed_nets / [llength $nets]"
    
    # Get die area
    set die_area [$block getDieArea]
    if {$die_area != "NULL"} {
        set die_bbox [$die_area getBox]
        set die_x1 [$die_bbox xMin]
        set die_y1 [$die_bbox yMin]
        set die_x2 [$die_bbox xMax]
        set die_y2 [$die_bbox yMax]
        set die_width [expr {($die_x2 - $die_x1) / 1000.0}]
        set die_height [expr {($die_y2 - $die_y1) / 1000.0}]
        puts "  Die Area: ${die_width} x ${die_height} microns"
    }
}
puts ""

# ============================================================================
# Step 5: Launch GUI for Visualization
# ============================================================================

puts "Step 5: Launching GUI for visualization..."
puts ""
puts "  Design loaded successfully!"
puts "  Opening GUI window..."
puts ""

# Launch GUI
if {[catch {gui::show} result]} {
    puts "  ✗ GUI launch failed: ${result}"
    puts ""
    puts "  Possible causes:"
    puts "    - Display not available (X11 forwarding not set up)"
    puts "    - GUI not available in this OpenROAD build"
    puts ""
    puts "  You can still use OpenROAD commands interactively."
    puts "  Try running with: openroad -gui view_routed.tcl"
    exit 1
} else {
    puts "  ✓ GUI launched successfully"
}

# Zoom to fit the design
puts "  Zooming to fit design..."
if {[catch {
    set block [ord::get_db_block]
    if {$block != "NULL"} {
        gui::zoom [gui::get_db_block]
        puts "  ✓ Zoomed to fit design"
    }
} result]} {
    puts "  ⚠ Could not auto-zoom: ${result}"
    puts "  You can manually zoom using: gui::zoom [gui::get_db_block]"
}

puts ""
puts "============================================================"
puts "Design Loaded for Visualization"
puts "============================================================"
puts ""
puts "The routed design is now loaded in OpenROAD GUI."
puts ""
puts "GUI Tips:"
puts "  - Use mouse wheel to zoom in/out"
puts "  - Click and drag to pan"
puts "  - View → Layers to show/hide routing layers"
puts "  - View → Congestion to see routing congestion heatmaps"
puts "  - Use layer visibility controls to view different metal layers"
puts ""
puts "Common commands in Tcl console:"
puts "  gui::highlight_net <net_name>  - Highlight a specific net"
puts "  gui::highlight_inst <inst_name> - Highlight a specific instance"
puts "  gui::zoom [gui::get_db_block]  - Zoom to fit the design"
puts ""

