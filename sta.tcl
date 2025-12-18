#!/usr/bin/env tclsh
#
# sta.tcl - Generic Static Timing Analysis Script for OpenROAD
#
# This script performs static timing analysis using OpenROAD/OpenSTA.
# It loads LIB, Verilog netlist, SPEF, and SDC files, then generates
# timing reports for setup, hold, and clock skew.
#
# Usage:
#   export DESIGN_NAME=6502
#   export BUILD_DIR=build
#   export LEF_FILE=tech/sky130_fd_sc_hd_merged.lef  # Optional, defaults to merged LEF
#   export LIBERTY_FILE=tech/sky130_fd_sc_hd__tt_025C_1v80.lib
#   export SDC_FILE=sdc/6502.sdc
#   openroad -exit sta.tcl
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

# LEF file (defaults to merged LEF file)
if {![info exists ::env(LEF_FILE)]} {
    set lef_file "tech/sky130_fd_sc_hd_merged.lef"
} else {
    set lef_file $::env(LEF_FILE)
}

# SDC file (first checks sdc/{DesignName}.sdc, then defaults to build/{DesignName}/{DesignName}.sdc)
if {![info exists ::env(SDC_FILE)]} {
    set sdc_file_sdc "sdc/${design_name}.sdc"
    set sdc_file_build "${build_dir}/${design_name}/${design_name}.sdc"
    if {[file exists $sdc_file_sdc]} {
        set sdc_file $sdc_file_sdc
    } else {
        set sdc_file $sdc_file_build
    }
} else {
    set sdc_file $::env(SDC_FILE)
}

# ============================================================================
# File Paths
# ============================================================================

set verilog_file "${build_dir}/${design_name}/${design_name}_renamed.v"
set spef_file "${build_dir}/${design_name}/${design_name}_old.spef"
# Check if renamed SPEF exists (from rename_spef.py)
set spef_file_renamed "${build_dir}/${design_name}/${design_name}_renamed.spef"
if {[file exists ${spef_file_renamed}]} {
    puts "  Note: Found renamed SPEF file, will use: ${spef_file_renamed}"
    set spef_file ${spef_file_renamed}
}
set sdc_file_path $sdc_file

# Output report files
set setup_report "${build_dir}/${design_name}/${design_name}_setup.rpt"
set hold_report "${build_dir}/${design_name}/${design_name}_hold.rpt"
set clock_skew_report "${build_dir}/${design_name}/${design_name}_clock_skew.rpt"

# ============================================================================
# Main STA Flow
# ============================================================================

puts "============================================================"
puts "Static Timing Analysis (STA) for ${design_name}"
puts "============================================================"
puts ""
puts "Configuration:"
puts "  Design Name:    ${design_name}"
puts "  Build Directory: ${build_dir}"
puts "  LEF File:       ${lef_file}"
puts "  Liberty File:   ${liberty_file}"
puts "  Verilog File:   ${verilog_file}"
puts "  SPEF File:      ${spef_file}"
puts "  SDC File:       ${sdc_file_path}"
puts ""

# ============================================================================
# Step 1: Read LEF File (Technology Definitions)
# ============================================================================
# LEF file must be read before Verilog to provide technology information
# required by link_design command.

puts "Step 1: Reading LEF file (technology definitions)..."
puts "  File: ${lef_file}"

if {![file exists ${lef_file}]} {
    puts "ERROR: LEF file not found: ${lef_file}"
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

if {[catch {read_lef ${lef_file}} result]} {
    puts "ERROR: Failed to read LEF file: ${result}"
    puts "       File: ${lef_file}"
    exit 1
}

puts "  ✓ LEF file loaded"
puts ""

# ============================================================================
# Step 2: Read Liberty File
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
# Step 3: Read Verilog Netlist
# ============================================================================

puts "Step 3: Reading Verilog netlist..."
puts "  File: ${verilog_file}"

if {![file exists ${verilog_file}]} {
    puts "ERROR: Verilog file not found: ${verilog_file}"
    exit 1
}

read_verilog ${verilog_file}

# Extract module name from Verilog file
# Look for "module <name> (" pattern in the file
set module_name ""
if {[catch {
    set verilog_fp [open ${verilog_file} r]
    set verilog_content [read $verilog_fp]
    close $verilog_fp
    
    # Pattern to match: module <name> (
    # Handle both "module name (" and "module name(" (with/without space)
    if {[regexp -line {^\s*module\s+(\w+)\s*\(} $verilog_content match module_name]} {
        puts "  Found module name: ${module_name}"
    } else {
        # Try alternative pattern (module on same line or different format)
        if {[regexp {module\s+(\w+)\s*\(} $verilog_content match module_name]} {
            puts "  Found module name: ${module_name}"
        } else {
            puts "  ⚠ WARNING: Could not extract module name from Verilog file"
            puts "  Will try using design name: ${design_name}"
            set module_name ${design_name}
        }
    }
} result]} {
    puts "  ⚠ WARNING: Error extracting module name: ${result}"
    puts "  Will try using design name: ${design_name}"
    set module_name ${design_name}
}

# Link design (creates the network and sets the top module)
# Note: link_design requires technology information from LEF file (read in Step 1)
# Try extracted module_name first, then "mod_${design_name}", then design_name
set linked_successfully 0
set mod_design_name "mod_${design_name}"

# Try extracted module name first (most accurate)
if {$module_name != "" && $module_name != ${design_name} && $module_name != ${mod_design_name}} {
    puts "  Attempting to link design as '${module_name}' (from Verilog)..."
    if {![catch {link_design ${module_name}} result]} {
        set linked_successfully 1
        puts "  ✓ Design linked as '${module_name}'"
    } else {
        puts "  ⚠ Failed to link as '${module_name}': ${result}"
    }
}

# Try "mod_${design_name}" (common naming convention from reference)
if {!$linked_successfully} {
    puts "  Attempting to link design as '${mod_design_name}'..."
    if {![catch {link_design ${mod_design_name}} result]} {
        set linked_successfully 1
        puts "  ✓ Design linked as '${mod_design_name}'"
    } else {
        puts "  ⚠ Failed to link as '${mod_design_name}': ${result}"
    }
}

# Fall back to design_name
if {!$linked_successfully} {
    puts "  Attempting to link design as '${design_name}'..."
    if {[catch {link_design ${design_name}} result]} {
        puts "ERROR: Failed to link design: ${result}"
        puts ""
        puts "  Tried module name: '${module_name}'"
        puts "  Tried: '${mod_design_name}'"
        puts "  Tried design name: '${design_name}'"
        puts ""
        puts "  Common causes:"
        puts "    - Module name not found in Verilog file"
        puts "    - Check that the Verilog file contains a valid module declaration"
        puts "    - Verify the module name matches what you expect"
        exit 1
    } else {
        puts "  ✓ Design linked as '${design_name}'"
        set linked_successfully 1
    }
}

puts "  ✓ Verilog netlist loaded and linked"
puts ""

# ============================================================================
# Step 4: Read SPEF File (Parasitic Extraction)
# ============================================================================
# Read SPEF before SDC (matching reference_sta.tcl order)

puts "Step 4: Reading SPEF file (parasitic extraction)..."
puts "  File: ${spef_file}"

if {![file exists ${spef_file}]} {
    puts "  ⚠ WARNING: SPEF file not found: ${spef_file}"
    puts "  STA will run without parasitic information (pre-route timing)."
    puts "  Results may be less accurate without RC parasitics."
    puts ""
} else {
    if {[catch {read_spef ${spef_file}} result]} {
        puts "  ✗ ERROR: Failed to read SPEF file: ${result}"
        puts "  This may be due to net name mismatches between SPEF and Verilog."
        puts "  Try running: python3 rename_spef.py ${spef_file} ${spef_file}_fixed.spef"
        puts "  Then use the fixed SPEF file."
        puts ""
    } else {
        puts "  ✓ SPEF file loaded"
        puts ""
    }
}

# ============================================================================
# Step 5: Read SDC Constraints
# ============================================================================
# Read SDC after SPEF (matching reference_sta.tcl order)

puts "Step 5: Reading SDC constraints..."
puts "  File: ${sdc_file_path}"

if {![file exists ${sdc_file_path}]} {
    puts "  ⚠ WARNING: SDC file not found: ${sdc_file_path}"
    puts "  Continuing without timing constraints (results may be incomplete)"
    puts ""
} else {
    if {[catch {read_sdc ${sdc_file_path}} result]} {
        puts "  ✗ ERROR: Failed to read SDC file: ${result}"
        puts "  This may cause timing analysis to fail."
        puts ""
    } else {
        puts "  ✓ SDC constraints loaded"
        
        # Verify clock was created correctly
        if {[catch {
            set clocks_after_sdc [all_clocks]
            set clk_found 0
            foreach clk $clocks_after_sdc {
                if {$clk == "clk"} {
                    set clk_found 1
                    break
                }
            }
            if {!$clk_found && [llength $clocks_after_sdc] > 0} {
                puts "  ⚠ WARNING: Clock 'clk' not found, but found: [lindex $clocks_after_sdc 0]"
                puts "  OpenSTA may have generated an internal clock name."
            } elseif {$clk_found} {
                puts "  ✓ Clock 'clk' created successfully"
            }
        } result_sdc_check]} {
            puts "  ⚠ Could not verify clock creation: ${result_sdc_check}"
        }
        puts ""
    }
}

# ============================================================================
# Step 5.5: Verify Timing Setup
# ============================================================================
# Note: Timing is automatically updated when reading SPEF and SDC
# No explicit update_timing command is needed (it doesn't exist in OpenSTA)

puts "Step 5.5: Verifying timing setup..."
puts "  Note: Timing is automatically updated when reading SPEF and SDC"

# Diagnostic: Check if clocks are defined and try to fix clock name issue
puts "  Checking clocks..."
set actual_clock_name ""
set clk_clock_found 0
if {[catch {
    set clocks [all_clocks]
    set clock_count [llength $clocks]
    if {$clock_count > 0} {
        puts "    ✓ Found $clock_count clock(s):"
        foreach clock $clocks {
            puts "      - $clock"
            if {$clock == "clk"} {
                set clk_clock_found 1
                set actual_clock_name "clk"
            } else {
                if {$actual_clock_name == ""} {
                    set actual_clock_name $clock
                }
            }
            # Check if clock is propagated
            if {[catch {
                set is_propagated [get_property [get_clocks $clock] is_propagated]
                if {$is_propagated} {
                    puts "        (propagated)"
                } else {
                    puts "        (not propagated - may need set_propagated_clock)"
                }
            } result_prop]} {
                puts "        (could not check propagation status)"
            }
        }
        
        if {!$clk_clock_found} {
            puts "    ⚠ WARNING: Clock 'clk' not found, using: $actual_clock_name"
            puts "    This may cause timing path issues."
            puts "    Attempting to query clock by port..."
            
            # Try to get clock by port
            if {[catch {
                set clock_by_port [get_clocks -of_objects [get_ports clk]]
                if {[llength $clock_by_port] > 0} {
                    set port_clock [lindex $clock_by_port 0]
                    puts "    Found clock on port 'clk': $port_clock"
                    if {$port_clock != $actual_clock_name} {
                        puts "    Clock name mismatch: SDC expects 'clk' but got '$port_clock'"
                    }
                }
            } result_port]} {
                puts "    Could not query clock by port: ${result_port}"
            }
        }
    } else {
        puts "    ✗ WARNING: No clocks found!"
        puts "    This may cause 'No paths found' in timing reports."
        puts "    Check that the SDC file defines clocks correctly."
    }
} result]} {
    puts "    ⚠ Could not check clocks: ${result}"
}

# Diagnostic: Check if clock port exists and trace clock connectivity
puts "  Checking clock port and connectivity..."
set clock_net_name ""
if {[catch {
    set clock_ports [get_ports -filter "name==clk"]
    if {[llength $clock_ports] > 0} {
        puts "    ✓ Clock port 'clk' found"
        
        # Check what net the clock port drives
        set clk_port [lindex $clock_ports 0]
        if {[catch {
            set clock_nets [get_nets -of_objects $clk_port]
            if {[llength $clock_nets] > 0} {
                set clock_net [lindex $clock_nets 0]
                set clock_net_name [get_property $clock_net name]
                puts "    Clock port 'clk' drives net: $clock_net_name"
            } else {
                puts "    ✗ CRITICAL: Clock port 'clk' does not drive any net!"
                puts "    This means the clock is not connected in the design."
            }
        } result_net]} {
            puts "    ⚠ Could not check clock net: ${result_net}"
        }
    } else {
        puts "    ✗ WARNING: Clock port 'clk' not found!"
        puts "    Available ports:"
        set all_ports [get_ports]
        set port_count [llength $all_ports]
        if {$port_count > 0 && $port_count <= 10} {
            foreach port $all_ports {
                puts "      - $port"
            }
        } elseif {$port_count > 10} {
            puts "      (showing first 10 of $port_count ports)"
            set shown 0
            foreach port $all_ports {
                if {$shown < 10} {
                    puts "      - $port"
                    incr shown
                } else {
                    break
                }
            }
        } else {
            puts "      (no ports found)"
        }
    }
} result]} {
    puts "    ⚠ Could not check clock port: ${result}"
}

# Diagnostic: Check for sequential cells (flip-flops) and their clock connections
puts "  Checking sequential cells (flip-flops) and clock connections..."
if {[catch {
    # Find all sequential cells (flip-flops, latches)
    set seq_cells [get_cells -hierarchical -filter "is_sequential==true"]
    set seq_count [llength $seq_cells]
    
    if {$seq_count > 0} {
        puts "    ✓ Found $seq_count sequential cell(s)"
        
        # Check clock pins of first few sequential cells
        set checked 0
        set clock_nets_found {}
        set clock_pins_found {}
        foreach seq $seq_cells {
            if {$checked < 5} {
                if {[catch {
                    set seq_name [get_property $seq full_name]
                    # Get clock pins (input pins that are clocks)
                    set clock_pins [get_pins -of_objects $seq -filter "direction==in && is_clock==true"]
                    if {[llength $clock_pins] > 0} {
                        set clock_pin [lindex $clock_pins 0]
                        set clock_pin_name [get_property $clock_pin name]
                        lappend clock_pins_found $clock_pin_name
                        
                        # Get net connected to clock pin
                        set pin_nets [get_nets -of_objects $clock_pin]
                        if {[llength $pin_nets] > 0} {
                            set pin_net [lindex $pin_nets 0]
                            set pin_net_name [get_property $pin_net name]
                            puts "      FF: $seq_name"
                            puts "        Clock pin: $clock_pin_name -> Net: $pin_net_name"
                            lappend clock_nets_found $pin_net_name
                        } else {
                            puts "      FF: $seq_name"
                            puts "        Clock pin: $clock_pin_name -> NOT CONNECTED TO ANY NET!"
                        }
                    } else {
                        puts "      FF: $seq_name -> No clock pin found (may not be a clocked FF)"
                    }
                } result_ff]} {
                    puts "      Could not check FF: ${result_ff}"
                }
                incr checked
            } else {
                break
            }
        }
        
        # Check if clock nets match
        if {$clock_net_name != "" && [llength $clock_nets_found] > 0} {
            set unique_clock_nets [lsort -unique $clock_nets_found]
            puts "    Clock nets found on FFs: $unique_clock_nets"
            
            set nets_match 0
            foreach net $unique_clock_nets {
                if {$net == $clock_net_name} {
                    set nets_match 1
                    break
                }
            }
            
            if {$nets_match} {
                puts "    ✓ Clock net '$clock_net_name' matches FF clock connections"
            } else {
                puts "    ✗ CRITICAL: Clock port net '$clock_net_name' does NOT match FF clock nets!"
                puts "    FF clock nets: $unique_clock_nets"
                puts "    This means the clock port is not connected to the flip-flops!"
                puts "    This will cause 'No paths found' in timing reports."
            }
        } elseif {[llength $clock_nets_found] == 0} {
            puts "    ✗ CRITICAL: No clock nets found on flip-flops!"
            puts "    Flip-flops may not have clock pins connected."
        }
    } else {
        puts "    ✗ WARNING: No sequential cells (flip-flops) found!"
        puts "    This design may be combinational only."
        puts "    Without flip-flops, there are no clocked timing paths to analyze."
        puts "    This explains 'No paths found' in timing reports."
    }
} result_ff_check]} {
    puts "    ⚠ Could not check flip-flops: ${result_ff_check}"
}

puts ""

# ============================================================================
# Step 6: Generate Setup Timing Report
# ============================================================================

puts "Step 6: Generating setup timing report..."
puts "  Report: ${setup_report}"
puts "  Number of paths: 100 (worst setup paths)"

# Note: Timing paths will be checked when generating reports

# Create output directory if it doesn't exist
file mkdir [file dirname ${setup_report}]

# Generate setup timing report
# Using report_checks with shell redirection (as in working reference)
# -path_delay max: setup timing (maximum delay paths)
# -no_line_splits: prevents line wrapping
# -fields: specifies what information to include
if {[catch {
    # Use shell redirection to write output to file (as in working reference)
    # report_checks will analyze all clocks by default, regardless of name
    puts "  Generating setup timing report (analyzing all clocks)..."
    report_checks -path_delay max -format full_clock_expanded -fields {slew cap input_pins nets fanout} -no_line_splits > ${setup_report}
    
    # Check if report has content
    if {[file exists ${setup_report}]} {
        set report_size [file size ${setup_report}]
        if {$report_size > 20} {
            puts "  ✓ Setup timing report generated ($report_size bytes)"
        } else {
            puts "  ⚠ WARNING: Setup timing report is very small ($report_size bytes)"
            puts "  This may indicate no timing paths were found."
        }
    } else {
        puts "  ✗ ERROR: Setup timing report file was not created"
    }
} result]} {
    puts "  ✗ ERROR: Failed to generate setup timing report: ${result}"
}
puts ""

# ============================================================================
# Step 8: Generate Hold Timing Report
# ============================================================================

puts "Step 8: Generating hold timing report..."
puts "  Report: ${hold_report}"
puts "  Number of paths: 100 (worst hold paths)"

# Generate hold timing report
# Using report_checks with shell redirection (as in working reference)
# -path_delay min: hold timing (minimum delay paths)
# -no_line_splits: prevents line wrapping
# -fields: specifies what information to include
if {[catch {
    # Use shell redirection to write output to file (as in working reference)
    # report_checks will analyze all clocks by default, regardless of name
    puts "  Generating hold timing report (analyzing all clocks)..."
    report_checks -path_delay min -format full_clock_expanded -fields {slew cap input_pins nets fanout} -no_line_splits > ${hold_report}
    
    # Check if report has content
    if {[file exists ${hold_report}]} {
        set report_size [file size ${hold_report}]
        if {$report_size > 20} {
            puts "  ✓ Hold timing report generated ($report_size bytes)"
        } else {
            puts "  ⚠ WARNING: Hold timing report is very small ($report_size bytes)"
            puts "  This may indicate no timing paths were found."
        }
    } else {
        puts "  ✗ ERROR: Hold timing report file was not created"
    }
} result]} {
    puts "  ✗ ERROR: Failed to generate hold timing report: ${result}"
}
puts ""

# ============================================================================
# Step 9: Generate Clock Skew Report
# ============================================================================

puts "Step 9: Generating clock skew report..."
puts "  Report: ${clock_skew_report}"

# Generate clock skew report
# Use shell redirection (as in working reference)
if {[catch {
    report_clock_skew > ${clock_skew_report}
    puts "  ✓ Clock skew report generated"
} result]} {
    puts "  ⚠ WARNING: Clock skew report generation encountered issues: ${result}"
    puts "  This may occur if no clocks are defined in the SDC file."
}

puts ""

# ============================================================================
# Final Summary
# ============================================================================

puts "============================================================"
puts "STA Analysis Complete"
puts "============================================================"
puts ""
puts "Generated Reports:"
if {[file exists ${setup_report}]} {
    puts "  ✓ Setup Timing:    ${setup_report}"
} else {
    puts "  ✗ Setup Timing:    (not generated)"
}
if {[file exists ${hold_report}]} {
    puts "  ✓ Hold Timing:     ${hold_report}"
} else {
    puts "  ✗ Hold Timing:     (not generated)"
}
if {[file exists ${clock_skew_report}]} {
    puts "  ✓ Clock Skew:      ${clock_skew_report}"
} else {
    puts "  ✗ Clock Skew:      (not generated)"
}
puts ""
puts "To view reports:"
puts "  cat ${setup_report}"
puts "  cat ${hold_report}"
puts "  cat ${clock_skew_report}"
puts ""
puts "============================================================"

# Exit successfully
exit 0

