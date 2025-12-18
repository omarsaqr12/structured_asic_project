# SDC File for arith
# Clock Period: 60.0 ns (16.67 MHz)

# Note: current_design is not needed for OpenSTA (it's an OpenROAD command)
# set_units is optional but good practice
set_units -time ns -resistance kOhm -capacitance pF -voltage V -current mA

# 1. Create Clock
create_clock -name clk -period 60.0 [get_ports clk]
set_clock_uncertainty 0.25 [get_clocks clk]
# Propagated clock for post-route STA (uses actual clock tree delays)
set_propagated_clock [get_clocks clk]

# 2. Input Delays (15% of period = 9.0 ns)
set_input_delay 9.0 -clock clk [get_ports { in_0 in_1 in_10 in_11 in_12 in_13 in_14 in_15 in_16 in_17 in_18 in_19 in_2 in_20 in_21 in_22 in_23 in_24 in_25 in_26 in_27 in_28 in_29 in_3 in_30 in_31 in_32 in_33 in_34 in_35 in_36 in_37 in_38 in_39 in_4 in_5 in_6 in_7 in_8 in_9 rst_n }]

# 3. Output Delays (15% of period = 9.0 ns)
set_output_delay 9.0 -clock clk [get_ports { oeb_0 oeb_1 oeb_10 oeb_11 oeb_12 oeb_13 oeb_14 oeb_15 oeb_16 oeb_17 oeb_18 oeb_19 oeb_2 oeb_20 oeb_21 oeb_22 oeb_23 oeb_24 oeb_25 oeb_26 oeb_27 oeb_28 oeb_29 oeb_3 oeb_30 oeb_31 oeb_32 oeb_33 oeb_34 oeb_35 oeb_36 oeb_37 oeb_38 oeb_39 oeb_4 oeb_5 oeb_6 oeb_7 oeb_8 oeb_9 out_0 out_1 out_10 out_11 out_12 out_13 out_14 out_15 out_16 out_17 out_18 out_19 out_2 out_20 out_21 out_22 out_23 out_24 out_25 out_26 out_27 out_28 out_29 out_3 out_30 out_31 out_32 out_33 out_34 out_35 out_36 out_37 out_38 out_39 out_4 out_5 out_6 out_7 out_8 out_9 }]

# Set load for outputs (5fF)
set_load 0.005 [get_ports { oeb_0 oeb_1 oeb_10 oeb_11 oeb_12 oeb_13 oeb_14 oeb_15 oeb_16 oeb_17 oeb_18 oeb_19 oeb_2 oeb_20 oeb_21 oeb_22 oeb_23 oeb_24 oeb_25 oeb_26 oeb_27 oeb_28 oeb_29 oeb_3 oeb_30 oeb_31 oeb_32 oeb_33 oeb_34 oeb_35 oeb_36 oeb_37 oeb_38 oeb_39 oeb_4 oeb_5 oeb_6 oeb_7 oeb_8 oeb_9 out_0 out_1 out_10 out_11 out_12 out_13 out_14 out_15 out_16 out_17 out_18 out_19 out_2 out_20 out_21 out_22 out_23 out_24 out_25 out_26 out_27 out_28 out_29 out_3 out_30 out_31 out_32 out_33 out_34 out_35 out_36 out_37 out_38 out_39 out_4 out_5 out_6 out_7 out_8 out_9 }]

# 4. Design Rule Constraints
set_max_fanout 10.0 [get_ports { in_0 in_1 in_10 in_11 in_12 in_13 in_14 in_15 in_16 in_17 in_18 in_19 in_2 in_20 in_21 in_22 in_23 in_24 in_25 in_26 in_27 in_28 in_29 in_3 in_30 in_31 in_32 in_33 in_34 in_35 in_36 in_37 in_38 in_39 in_4 in_5 in_6 in_7 in_8 in_9 }]
set_max_transition 1.2 [get_ports { in_0 in_1 in_10 in_11 in_12 in_13 in_14 in_15 in_16 in_17 in_18 in_19 in_2 in_20 in_21 in_22 in_23 in_24 in_25 in_26 in_27 in_28 in_29 in_3 in_30 in_31 in_32 in_33 in_34 in_35 in_36 in_37 in_38 in_39 in_4 in_5 in_6 in_7 in_8 in_9 }]

# 5. (Optional) Set false path for asynchronous reset
set_false_path -from [get_ports rst_n]