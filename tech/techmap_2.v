// arithmetic_fabric.v
// Techmap for arithmetic operations using HA + OR2 fabric architecture
//
// Fabric arithmetic cluster per tile:
//   - 2× Half Adder (HA): Each has SUM (XOR) and COUT (AND)
//   - 1× OR2: Used to combine carries for Full Adder
//
// Full Adder Construction:
//   FA(A,B,Cin) = 2×HA + 1×OR2
//   HA1: sum1=A⊕B, c1=A·B
//   HA2: sum=sum1⊕Cin, c2=sum1·Cin  
//   Cout = c1 | c2  (uses the OR2!)
//
// Bonus: HA outputs can be reused for logic:
//   - XOR: Use HA.SUM output
//   - AND: Use HA.COUT output

// =============================================================================
// Half Adder Mapping (1-bit addition without carry-in)
// =============================================================================

(* techmap_celltype = "$_HA_" *)
module _80_ha_map (A, B, SUM, COUT);
    input A, B;
    output SUM, COUT;
    
    // Direct mapping to sky130 half adder cell
    sky130_fd_sc_hd__ha_2 _TECHMAP_REPLACE_ (
        .A(A),
        .B(B), 
        .SUM(SUM),
        .COUT(COUT),
        .VPWR(1'b1),
        .VGND(1'b0),
        .VPB(1'b1),
        .VNB(1'b0)
    );
endmodule

// =============================================================================
// Full Adder Mapping (1-bit addition with carry-in)
// Uses: 2× Half Adders + 1× OR2
// =============================================================================

(* techmap_celltype = "$_FA_" *)
module _80_fa_map (A, B, CI, SUM, CO);
    input A, B, CI;
    output SUM, CO;
    
    wire sum1, carry1, carry2;
    
    // First half adder: A + B
    sky130_fd_sc_hd__ha_2 ha_first (
        .A(A),
        .B(B),
        .SUM(sum1),     // A ⊕ B
        .COUT(carry1),  // A · B
        .VPWR(1'b1),
        .VGND(1'b0),
        .VPB(1'b1),
        .VNB(1'b0)
    );
    
    // Second half adder: sum1 + CI
    sky130_fd_sc_hd__ha_2 ha_second (
        .A(sum1),
        .B(CI),
        .SUM(SUM),      // Final sum = (A ⊕ B) ⊕ CI
        .COUT(carry2),  // (A ⊕ B) · CI
        .VPWR(1'b1),
        .VGND(1'b0),
        .VPB(1'b1),
        .VNB(1'b0)
    );
    
    // Combine carries with OR2 (this is why OR2 exists!)
    sky130_fd_sc_hd__or2_2 or_carry (
        .A(carry1),
        .B(carry2),
        .X(CO),         // Final carry = (A·B) | ((A⊕B)·CI)
        .VPWR(1'b1),
        .VGND(1'b0),
        .VPB(1'b1),
        .VNB(1'b0)
    );
endmodule

// =============================================================================
// Multi-bit Adder Mapping
// Yosys $add operation -> Chain of Full Adders
// =============================================================================

(* techmap_celltype = "$add" *)
module _80_add (A, B, Y);
    parameter A_SIGNED = 0;
    parameter B_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter B_WIDTH = 1;
    parameter Y_WIDTH = 1;
    
    input [A_WIDTH-1:0] A;
    input [B_WIDTH-1:0] B;
    output [Y_WIDTH-1:0] Y;
    
    // For single-bit, use half adder (no carry in)
    generate
        if (Y_WIDTH == 1) begin
            wire unused_cout;
            $_HA_ ha_single (
                .A(A[0]),
                .B(B[0]),
                .SUM(Y[0]),
                .COUT(unused_cout)
            );
        end
        // For multi-bit, use ripple carry with full adders
        else begin
            wire [Y_WIDTH:0] carry;
            assign carry[0] = 1'b0;  // No carry in
            
            genvar i;
            for (i = 0; i < Y_WIDTH; i = i + 1) begin:adder_chain
                wire a_bit, b_bit;
                assign a_bit = (i < A_WIDTH) ? A[i] : (A_SIGNED ? A[A_WIDTH-1] : 1'b0);
                assign b_bit = (i < B_WIDTH) ? B[i] : (B_SIGNED ? B[B_WIDTH-1] : 1'b0);
                
                // First bit can use HA if no carry in
                if (i == 0) begin
                    $_HA_ ha_lsb (
                        .A(a_bit),
                        .B(b_bit),
                        .SUM(Y[i]),
                        .COUT(carry[i+1])
                    );
                end
                // Other bits use FA
                else begin
                    $_FA_ fa_bit (
                        .A(a_bit),
                        .B(b_bit),
                        .CI(carry[i]),
                        .SUM(Y[i]),
                        .CO(carry[i+1])
                    );
                end
            end
        end
    endgenerate
endmodule

// =============================================================================
// Subtraction using Full Adder + Inversion
// A - B = A + (~B) + 1
// =============================================================================

(* techmap_celltype = "$sub" *)
module _80_sub (A, B, Y);
    parameter A_SIGNED = 0;
    parameter B_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter B_WIDTH = 1;
    parameter Y_WIDTH = 1;
    
    input [A_WIDTH-1:0] A;
    input [B_WIDTH-1:0] B;
    output [Y_WIDTH-1:0] Y;
    
    // Invert B
    wire [Y_WIDTH-1:0] b_inverted;
    genvar i;
    generate
        for (i = 0; i < Y_WIDTH; i = i + 1) begin:invert_b
            wire b_bit;
            assign b_bit = (i < B_WIDTH) ? B[i] : (B_SIGNED ? B[B_WIDTH-1] : 1'b0);
            $_NOT_ inv_b (.A(b_bit), .Y(b_inverted[i]));
        end
    endgenerate
    
    // Add A + (~B) + 1 using full adders
    wire [Y_WIDTH:0] carry;
    assign carry[0] = 1'b1;  // The +1 for two's complement
    
    generate
        for (i = 0; i < Y_WIDTH; i = i + 1) begin:sub_chain
            wire a_bit;
            assign a_bit = (i < A_WIDTH) ? A[i] : (A_SIGNED ? A[A_WIDTH-1] : 1'b0);
            
            $_FA_ fa_sub (
                .A(a_bit),
                .B(b_inverted[i]),
                .CI(carry[i]),
                .SUM(Y[i]),
                .CO(carry[i+1])
            );
        end
    endgenerate
endmodule

// =============================================================================
// Increment Operation (optimized)
// A + 1 = Use half adders with carry chain
// =============================================================================

(* techmap_celltype = "$add" *)
module _80_increment (A, B, Y);
    parameter A_SIGNED = 0;
    parameter B_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter B_WIDTH = 1;
    parameter Y_WIDTH = 1;
    
    input [A_WIDTH-1:0] A;
    input [B_WIDTH-1:0] B;
    output [Y_WIDTH-1:0] Y;
    
    // Detect if this is an increment (B == 1)
    // This optimization only applies for literal 1
    generate
        if (B_WIDTH == 1 && B_SIGNED == 0) begin
            // This might be A + 1, use carry chain
            wire [Y_WIDTH:0] carry;
            assign carry[0] = B[0];
            
            genvar i;
            for (i = 0; i < Y_WIDTH; i = i + 1) begin:inc_chain
                wire a_bit;
                assign a_bit = (i < A_WIDTH) ? A[i] : 1'b0;
                
                if (i == 0) begin
                    $_HA_ ha_inc (
                        .A(a_bit),
                        .B(carry[i]),
                        .SUM(Y[i]),
                        .COUT(carry[i+1])
                    );
                end else begin
                    $_HA_ ha_prop (
                        .A(a_bit),
                        .B(carry[i]),
                        .SUM(Y[i]),
                        .COUT(carry[i+1])
                    );
                end
            end
        end
    endgenerate
endmodule