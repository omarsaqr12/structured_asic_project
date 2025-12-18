#!/usr/bin/env python3
"""Check the generated DEF file."""
import os
import json
import sys

try:
    def_path = 'C:/Users/AUC/Downloads/dd2/structured_asic_project/build/6502/6502_fixed.def'
    output_path = 'C:/Users/AUC/Downloads/dd2/structured_asic_project/def_check_result.json'

    if not os.path.exists(def_path):
        result = {"error": f"DEF file not found: {def_path}"}
    else:
        with open(def_path, 'r') as f:
            lines = f.readlines()

        # Check for END statements
        has_end_pins = any("END PINS" in line for line in lines)
        has_end_components = any("END COMPONENTS" in line for line in lines)
        has_end_design = any("END DESIGN" in line for line in lines)

        result = {
            "total_lines": len(lines),
            "last_10_lines": [line.rstrip() for line in lines[-10:]],
            "has_end_pins": has_end_pins,
            "has_end_components": has_end_components,
            "has_end_design": has_end_design,
            "file_size_bytes": os.path.getsize(def_path)
        }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Result written to {output_path}")
    print(json.dumps(result, indent=2))

except Exception as e:
    error_result = {"error": str(e), "type": type(e).__name__}
    print(f"Error: {e}")
    with open(output_path, 'w') as f:
        json.dump(error_result, f, indent=2)
