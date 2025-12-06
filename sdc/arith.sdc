# SDC Constraints for arith
# Clock period: 20.0 ns

# 1. Clock definition
create_clock -name clk -period 20.0 [get_ports clk]

# 2. Input delay (exclude the clock port)
set_input_delay 2.0 -clock clk \
    [remove_from_collection [all_inputs] [get_ports clk]]

# 3. Output delay
set_output_delay 2.0 -clock clk [all_outputs]
