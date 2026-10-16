// ============================================================================
// Module Name: structural_multiplier
// Description: 8x8 Unsigned Structural Multiplier using Carry-Save Array (CSA)
//              without using the '*' multiplication operator.
// ============================================================================

// 1-bit Half Adder
module half_adder (
    input  a,
    input  b,
    output sum,
    output cout
);
    assign sum  = a ^ b;
    assign cout = a & b;
endmodule

// 1-bit Full Adder
module full_adder (
    input  a,
    input  b,
    input  cin,
    output sum,
    output cout
);
    assign sum  = a ^ b ^ cin;
    assign cout = (a & b) | (b & cin) | (a & cin);
endmodule

// 8x8 Structural Multiplier
module structural_multiplier (
    input  [7:0] A,
    input  [7:0] B,
    output [15:0] P
);

    // ------------------------------------------------------------------------
    // Step 1: Partial Product Generation (64 1-bit AND operations)
    // ------------------------------------------------------------------------
    wire [7:0] pp [7:0];

    genvar i, j;
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_pp_rows
            for (j = 0; j < 8; j = j + 1) begin : gen_pp_cols
                assign pp[i][j] = A[j] & B[i];
            end
        end
    endgenerate

    // Bit 0 of product is the least significant partial product
    assign P[0] = pp[0][0];

    // ------------------------------------------------------------------------
    // Step 2: Carry-Save Addition Tree
    // ------------------------------------------------------------------------
    // Sum and Carry nets for stages 1 through 7
    wire [7:0] S [1:7];
    wire [7:0] C [1:7];

    // --- Stage 1: Add {1'b0, pp[0][7:1]} and pp[1][7:0] with cin = 0 ---
    generate
        for (j = 0; j < 8; j = j + 1) begin : gen_stage_1
            wire in_s = (j < 7) ? pp[0][j+1] : 1'b0;
            full_adder fa_stg1 (
                .a(in_s),
                .b(pp[1][j]),
                .cin(1'b0),
                .sum(S[1][j]),
                .cout(C[1][j])
            );
        end
    endgenerate
    assign P[1] = S[1][0];

    // --- Stages 2 to 7: Add {1'b0, S[i-1][7:1]}, pp[i][7:0], and C[i-1][7:0] ---
    generate
        for (i = 2; i <= 7; i = i + 1) begin : gen_stages
            for (j = 0; j < 8; j = j + 1) begin : gen_cols
                wire in_s = (j < 7) ? S[i-1][j+1] : 1'b0;
                full_adder fa_inst (
                    .a(in_s),
                    .b(pp[i][j]),
                    .cin(C[i-1][j]),
                    .sum(S[i][j]),
                    .cout(C[i][j])
                );
            end
        end
    endgenerate

    // Product bits 2 to 7 come directly from stage sum bit 0
    generate
        for (i = 2; i <= 7; i = i + 1) begin : gen_p_out
            assign P[i] = S[i][0];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Step 3: Final Carry-Propagate Adder (CPA) for upper bits P[15:8]
    // ------------------------------------------------------------------------
    wire [8:0] cpa_carry;
    assign cpa_carry[0] = 1'b0;

    generate
        for (j = 0; j < 8; j = j + 1) begin : gen_cpa
            wire in_s = (j < 7) ? S[7][j+1] : 1'b0;
            full_adder fa_cpa (
                .a(in_s),
                .b(C[7][j]),
                .cin(cpa_carry[j]),
                .sum(P[8+j]),
                .cout(cpa_carry[j+1])
            );
        end
    endgenerate

endmodule
