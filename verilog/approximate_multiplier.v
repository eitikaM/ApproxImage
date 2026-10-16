// ============================================================================
// Module Name: approximate_multiplier
// Description: Configurable 8x8 Hybrid Lower-Part-OR Approximate Multiplier
//
// Modes:
//   2'b00 (Mode 0) : Exact (100% accurate structural CSA multiplication)
//   2'b01 (Mode 1) : Low Approximation    (Columns 0..2 approximated via OR)
//   2'b10 (Mode 2) : Medium Approximation (Columns 0..4 approximated via OR)
//   2'b11 (Mode 3) : High Approximation   (Columns 0..6 approximated via OR)
//
// In all approximate modes:
//   - Lower columns (< K) use multi-input OR reduction.
//   - Higher columns (>= K) use exact Carry-Save and Carry-Propagate addition.
//   - No arithmetic '*' operator is used anywhere.
// ============================================================================

`timescale 1ns/1ps

module approximate_multiplier (
    input  [7:0] A,
    input  [7:0] B,
    input  [1:0] mode,
    output [15:0] P
);

    // ------------------------------------------------------------------------
    // Step 1: Structural Partial Product Generation (64 1-bit AND gates)
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

    // ------------------------------------------------------------------------
    // Step 2: Lower-Part OR Reduction Logic (Columns 0 to 6)
    //         Weight of term pp[i][j] is c = i + j
    // ------------------------------------------------------------------------
    wire [6:0] col_or;

    // Column 0: 1 term (pp[0][0])
    assign col_or[0] = pp[0][0];

    // Column 1: 2 terms
    assign col_or[1] = pp[0][1] | pp[1][0];

    // Column 2: 3 terms
    assign col_or[2] = pp[0][2] | pp[1][1] | pp[2][0];

    // Column 3: 4 terms
    assign col_or[3] = pp[0][3] | pp[1][2] | pp[2][1] | pp[3][0];

    // Column 4: 5 terms
    assign col_or[4] = pp[0][4] | pp[1][3] | pp[2][2] | pp[3][1] | pp[4][0];

    // Column 5: 6 terms
    assign col_or[5] = pp[0][5] | pp[1][4] | pp[2][3] | pp[3][2] | pp[4][1] | pp[5][0];

    // Column 6: 7 terms
    assign col_or[6] = pp[0][6] | pp[1][5] | pp[2][4] | pp[3][3] | pp[4][2] | pp[5][1] | pp[6][0];

    // ------------------------------------------------------------------------
    // Step 3: Exact CSA Adder Tree Input Selection
    //         If column (i+j) is in the approximate zone for the active mode,
    //         its partial product is decoupled from the exact adder tree (tied to 0).
    // ------------------------------------------------------------------------
    wire [7:0] pp_csa [7:0];

    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_csa_in_rows
            for (j = 0; j < 8; j = j + 1) begin : gen_csa_in_cols
                wire is_approx_col;
                assign is_approx_col = (mode == 2'b01 && (i + j) < 3) ||
                                       (mode == 2'b10 && (i + j) < 5) ||
                                       (mode == 2'b11 && (i + j) < 7);

                assign pp_csa[i][j] = is_approx_col ? 1'b0 : pp[i][j];
            end
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Step 4: Structural Carry-Save Addition Tree
    // ------------------------------------------------------------------------
    wire [15:0] P_csa;
    wire [7:0] S [1:7];
    wire [7:0] C [1:7];

    assign P_csa[0] = pp_csa[0][0];

    // Stage 1
    generate
        for (j = 0; j < 8; j = j + 1) begin : gen_stage_1
            wire in_s = (j < 7) ? pp_csa[0][j+1] : 1'b0;
            full_adder fa_stg1 (
                .a(in_s),
                .b(pp_csa[1][j]),
                .cin(1'b0),
                .sum(S[1][j]),
                .cout(C[1][j])
            );
        end
    endgenerate
    assign P_csa[1] = S[1][0];

    // Stages 2 to 7
    generate
        for (i = 2; i <= 7; i = i + 1) begin : gen_stages
            for (j = 0; j < 8; j = j + 1) begin : gen_cols
                wire in_s = (j < 7) ? S[i-1][j+1] : 1'b0;
                full_adder fa_inst (
                    .a(in_s),
                    .b(pp_csa[i][j]),
                    .cin(C[i-1][j]),
                    .sum(S[i][j]),
                    .cout(C[i][j])
                );
            end
        end
    endgenerate

    // Outputs P_csa[2] through P_csa[7]
    generate
        for (i = 2; i <= 7; i = i + 1) begin : gen_p_csa_out
            assign P_csa[i] = S[i][0];
        end
    endgenerate

    // Final Carry-Propagate Adder (CPA) for upper bits P_csa[15:8]
    wire [8:0] cpa_carry;
    assign cpa_carry[0] = 1'b0;

    generate
        for (j = 0; j < 8; j = j + 1) begin : gen_cpa
            wire in_s = (j < 7) ? S[7][j+1] : 1'b0;
            full_adder fa_cpa (
                .a(in_s),
                .b(C[7][j]),
                .cin(cpa_carry[j]),
                .sum(P_csa[8+j]),
                .cout(cpa_carry[j+1])
            );
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Step 5: Output Multiplexing (Controlled by Mode)
    //         Approximate columns (< K) select col_or.
    //         Exact columns (>= K) select P_csa.
    // ------------------------------------------------------------------------
    genvar b;
    generate
        for (b = 0; b < 16; b = b + 1) begin : gen_p_out
            if (b < 7) begin : gen_lower_bits
                wire is_approx_bit;
                assign is_approx_bit = (mode == 2'b01 && b < 3) ||
                                       (mode == 2'b10 && b < 5) ||
                                       (mode == 2'b11 && b < 7);
                assign P[b] = is_approx_bit ? col_or[b] : P_csa[b];
            end else begin : gen_upper_bits
                // Bits 7 to 15 are always purely exact across all modes
                assign P[b] = P_csa[b];
            end
        end
    endgenerate

endmodule
