# SDC Constraints for z80
# Clock period: 40.0 ns

# 1. Clock definition
create_clock -name clk -period 40.0 [get_ports clk]

# 2. Input delay (applies to all input ports)
# Note: Clock port is included but won't affect routing
set_input_delay 2.0 -clock clk [all_inputs]

# 3. Output delay
set_output_delay 2.0 -clock clk [all_outputs]
