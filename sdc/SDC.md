# SDC (Synopsys Design Constraints) Files

This document contains the SDC constraint files and generation script for all designs in the structured ASIC project.

## Timing Constraints Summary

| Design      | File Size | Recommended Frequency | Period (ns) | Rationale                                                                                                      |
| ----------- | --------- | --------------------- | ----------- | -------------------------------------------------------------------------------------------------------------- |
| **aes_128** | 50 MB     | 50 MHz                | 20.0 ns     | Big netlist → start conservative; 50 MHz is a balanced starting point for optimization                         |
| **z80**     | 5.6 MB    | 25 MHz                | 40.0 ns     | Control-heavy, similar to 6502 — safe default                                                                  |
| **6502**    | 1.7 MB    | 25 MHz                | 40.0 ns     | Long decode/control paths; 50 MHz is unlikely to pass                                                          |
| **arith**   | 303 KB    | 50 MHz (safe)         | 20.0 ns     | Small datapath — can be much faster. Start at 50 MHz if you want consistent safety (can go to 100 MHz / 10 ns) |

### Notes

- **aes_128**: Conservative option is 25 MHz (40.0 ns), but 50 MHz (20.0 ns) is recommended as a balanced starting point
- **6502**: Conservative range is 20-25 MHz (40-50 ns); 50 MHz is unlikely to pass due to long decode/control paths
- **arith**: Can potentially run at 100 MHz (10.0 ns), but 50 MHz (20.0 ns) is recommended for safety

---

## SDC File Format

Each SDC file contains exactly three mandatory constraints:

1. **Clock Definition**: `create_clock -name clk -period <period_ns> [get_ports clk]`
2. **Input Delay**: `set_input_delay 2.0 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]`
3. **Output Delay**: `set_output_delay 2.0 -clock clk [all_outputs]`

All delay values are set to 2.0 ns as specified.

---

## Generated SDC Files

### 6502.sdc

```
# SDC Constraints for 6502
# Clock period: 40.0 ns

# 1. Clock definition
create_clock -name clk -period 40.0 [get_ports clk]

# 2. Input delay (exclude the clock port)
set_input_delay 2.0 -clock clk \
    [remove_from_collection [all_inputs] [get_ports clk]]

# 3. Output delay
set_output_delay 2.0 -clock clk [all_outputs]
```

### aes_128.sdc

```
# SDC Constraints for aes_128
# Clock period: 20.0 ns

# 1. Clock definition
create_clock -name clk -period 20.0 [get_ports clk]

# 2. Input delay (exclude the clock port)
set_input_delay 2.0 -clock clk \
    [remove_from_collection [all_inputs] [get_ports clk]]

# 3. Output delay
set_output_delay 2.0 -clock clk [all_outputs]
```

### arith.sdc

```
# SDC Constraints for arith
# Clock period: 20.0 ns

# 1. Clock definition
create_clock -name clk -period 20.0 [get_ports clk]

# 2. Input delay (exclude the clock port)
set_input_delay 2.0 -clock clk \
    [remove_from_collection [all_inputs] [get_ports clk]]

# 3. Output delay
set_output_delay 2.0 -clock clk [all_outputs]
```

### z80.sdc

```
# SDC Constraints for z80
# Clock period: 40.0 ns

# 1. Clock definition
create_clock -name clk -period 40.0 [get_ports clk]

# 2. Input delay (exclude the clock port)
set_input_delay 2.0 -clock clk \
    [remove_from_collection [all_inputs] [get_ports clk]]

# 3. Output delay
set_output_delay 2.0 -clock clk [all_outputs]
```

---

## SDC Generation Script

The `generate_sdc.py` script automatically generates SDC files for all designs. See the script file for the complete source code.

---

## Usage

### Generate all SDC files:

```bash
python generate_sdc.py
```

### Generate SDC for a specific design:

```bash
python generate_sdc.py --design 6502
python generate_sdc.py --design aes_128
python generate_sdc.py --design arith
python generate_sdc.py --design z80
```

### Custom output directory:

```bash
python generate_sdc.py --output-dir build/sdc/
```

---

## Validation

Each SDC file is validated to ensure:

- ✅ One and only one clock definition using `create_clock`
- ✅ Input delay specified for all non-clock inputs using `remove_from_collection`
- ✅ Output delay specified for all outputs using `all_outputs`
- ✅ All delay values set to 2.0 ns
- ✅ Files parse successfully in OpenROAD / OpenSTA with no warnings

---

## Notes

- All designs use `clk` as the clock port name
- Input delays exclude the clock port using `remove_from_collection`
- Output delays apply to all outputs using `all_outputs`
- These constraints are starting points for first STA runs and may need adjustment based on timing analysis results
