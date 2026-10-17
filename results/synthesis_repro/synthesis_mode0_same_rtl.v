`timescale 1ns/1ps

module synthesis_mode0_same_rtl (
    input  [7:0] A,
    input  [7:0] B,
    output [15:0] P
);

    approximate_multiplier uut (
        .A(A),
        .B(B),
        .mode(2'b00),
        .P(P)
    );

endmodule
