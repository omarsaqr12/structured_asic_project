#!/usr/bin/env tclsh
#
# extract_spef.tcl - OpenROAD SPEF Extraction Script
#
# This script loads a detailed routed DEF file and extracts SPEF (parasitic) data.
# Uses environment variables for configuration.
#
# Usage:
#   export DESIGN_NAME=6502
#   export BUILD_DIR=build
#   export LEF_FILE=tech/sky130_fd_sc_hd_merged.lef
#   export ROUTED_DEF=build/6502/6502_routed.def  # Optional, defaults to build/{design}/{design}_routed.def
#   export EXTRACTION_RULES=rcx_patterns.rules    # Optional, defaults to rcx_patterns.rules
#   openroad -exit extract_spef.tcl
#
# Note: The merged LEF file is required for technology information.
#       The routed DEF file must contain detailed routing information.

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

# LEF file (merged file containing BOTH technology and macro definitions)
if {![info exists ::env(LEF_FILE)]} {
    set lef_file "tech/sky130_fd_sc_hd_merged.lef"
} else {
    set lef_file $::env(LEF_FILE)
}

# Routed DEF file (defaults to build/{design}/{design}_routed.def)
if {![info exists ::env(ROUTED_DEF)]} {
    set routed_def_file "${build_dir}/${design_name}/${design_name}_routed.def"
} else {
    set routed_def_file $::env(ROUTED_DEF)
}

# Extraction rules file (defaults to rcx_patterns.rules)
if {![info exists ::env(EXTRACTION_RULES)]} {
    set extraction_rules_file "rcx_patterns.rules"
} else {
    set extraction_rules_file $::env(EXTRACTION_RULES)
}

# Output SPEF file (defaults to build/{design}/{design}.spef)
if {![info exists ::env(SPEF_FILE)]} {
    set spef_file "${build_dir}/${design_name}/${design_name}.spef"
} else {
    set spef_file $::env(SPEF_FILE)
}

# ============================================================================
# Main Extraction Flow
# ============================================================================

puts "============================================================"
puts "SPEF Extraction for ${design_name}"
puts "============================================================"
puts ""
puts "Configuration:"
puts "  Design Name:        ${design_name}"
puts "  Build Directory:    ${build_dir}"
puts "  LEF File:           ${lef_file}"
puts "  Routed DEF File:    ${routed_def_file}"
puts "  Extraction Rules:   ${extraction_rules_file}"
puts "  Output SPEF File:   ${spef_file}"
puts ""

# ============================================================================
# Step 1: Reading LEF file
# ============================================================================

puts "Step 1: Reading LEF file..."
puts "  File: ${lef_file}"
if {![file exists ${lef_file}]} {
    puts "ERROR: LEF file not found: ${lef_file}"
    puts ""
    puts "  The LEF file is required for technology information."
    puts "  Please ensure ${lef_file} exists."
    exit 1
}

# Read LEF file
if {[catch {read_lef ${lef_file}} result]} {
    puts "ERROR: Failed to read LEF file: ${result}"
    puts "       File: ${lef_file}"
    exit 1
}
puts "  ✓ LEF file loaded"
puts ""

# ============================================================================
# Step 2: Reading Routed DEF file
# ============================================================================

puts "Step 2: Reading routed DEF file..."
puts "  File: ${routed_def_file}"
if {![file exists ${routed_def_file}]} {
    puts "ERROR: Routed DEF file not found: ${routed_def_file}"
    puts ""
    puts "  The routed DEF file must contain detailed routing information."
    puts "  Please ensure ${routed_def_file} exists."
    puts ""
    puts "  Expected file location: ${build_dir}/${design_name}/${design_name}_routed.def"
    puts "  Or specify custom path with: export ROUTED_DEF=/path/to/routed.def"
    exit 1
}

# Read DEF file
if {[catch {read_def ${routed_def_file}} result]} {
    puts "ERROR: Failed to read DEF file: ${result}"
    puts "       File: ${routed_def_file}"
    exit 1
}
puts "  ✓ Routed DEF file loaded"
puts ""

# ============================================================================
# Step 3: Extracting Parasitics
# ============================================================================

puts "Step 3: Extracting parasitics..."
puts "  Using detailed routing from DEF file for extraction"

# Check if extraction rules file exists
puts "  Extraction Rules File: ${extraction_rules_file}"
if {![file exists ${extraction_rules_file}]} {
    puts "  ⚠ WARNING: Extraction rules file not found: ${extraction_rules_file}"
    puts "  Will attempt fallback extraction methods..."
    puts ""
}

set extraction_success 0

# Extract parasitics using the extraction rules file
# The -ext_model_file option specifies the extraction rules file
if {[file exists ${extraction_rules_file}]} {
    puts "  Using extraction rules file for accurate RC extraction..."
    if {[catch {
        extract_parasitics -ext_model_file ${extraction_rules_file}
    } result]} {
        puts "  ✗ Extraction with rules file failed: ${result}"
        puts ""
        puts "  Attempting fallback: LEF-based extraction..."
        # Fallback: try LEF-based extraction
        if {[catch {
            extract_parasitics -lef_res
        } result2]} {
            puts "  ✗ LEF-based extraction also failed: ${result2}"
            puts ""
            puts "  Attempting final fallback: default extraction..."
            # Final fallback: try without any options
            if {[catch {
                extract_parasitics
            } result3]} {
                puts "  ✗ Parasitic extraction failed: ${result3}"
                puts ""
                puts "  ERROR: Could not extract parasitics."
                puts "  This may occur if:"
                puts "    - The extraction rules file format is incorrect"
                puts "    - The design doesn't have routing information"
                puts "    - Required technology information is missing"
                set extraction_success 0
            } else {
                puts "  ✓ Parasitic extraction complete (default method)"
                puts "  ⚠ WARNING: Using default extraction without rules file (less accurate)"
                set extraction_success 1
            }
        } else {
            puts "  ✓ Parasitic extraction complete (LEF-based fallback)"
            puts "  ⚠ WARNING: Using LEF-based extraction instead of rules file (less accurate)"
            set extraction_success 1
        }
    } else {
        puts "  ✓ Parasitic extraction complete (using extraction rules file)"
        set extraction_success 1
    }
} else {
    # No rules file, try LEF-based extraction first
    puts "  Attempting LEF-based extraction..."
    if {[catch {
        extract_parasitics -lef_res
    } result]} {
        puts "  ✗ LEF-based extraction failed: ${result}"
        puts ""
        puts "  Attempting fallback: default extraction..."
        if {[catch {
            extract_parasitics
        } result2]} {
            puts "  ✗ Parasitic extraction failed: ${result2}"
            puts ""
            puts "  ERROR: Could not extract parasitics."
            puts "  This may occur if:"
            puts "    - The design doesn't have routing information"
            puts "    - Required technology information is missing"
            set extraction_success 0
        } else {
            puts "  ✓ Parasitic extraction complete (default method)"
            puts "  ⚠ WARNING: Using default extraction (less accurate)"
            set extraction_success 1
        }
    } else {
        puts "  ✓ Parasitic extraction complete (LEF-based)"
        set extraction_success 1
    }
}

# ============================================================================
# Step 4: Writing SPEF file
# ============================================================================

if {$extraction_success} {
    puts ""
    puts "Step 4: Writing SPEF file..."
    
    # Create output directory if it doesn't exist
    file mkdir [file dirname ${spef_file}]
    
    puts "  File: ${spef_file}"
    if {[catch {write_spef ${spef_file}} result]} {
        puts "  ✗ Failed to write SPEF file: ${result}"
        exit 1
    } else {
        puts "  ✓ SPEF file saved: ${spef_file}"
        if {[file exists ${extraction_rules_file}]} {
            puts "  Note: SPEF contains RC values from extraction rules file"
        } else {
            puts "  Note: SPEF contains RC values from LEF-based or default extraction"
        }
    }
} else {
    puts ""
    puts "ERROR: Parasitic extraction failed. Cannot generate SPEF file."
    exit 1
}

puts ""

# ============================================================================
# Final Summary
# ============================================================================

puts "============================================================"
puts "SPEF Extraction Complete"
puts "============================================================"
puts ""
puts "Output File:"
puts "  SPEF: ${spef_file}"
puts ""
if {[file exists ${extraction_rules_file}]} {
    puts "Extraction Method:"
    puts "  ✓ Used extraction rules file: ${extraction_rules_file}"
} else {
    puts "Extraction Method:"
    puts "  ⚠ Used fallback extraction (LEF-based or default)"
    puts "  Note: For more accurate extraction, provide rcx_patterns.rules"
}
puts ""
puts "============================================================"

