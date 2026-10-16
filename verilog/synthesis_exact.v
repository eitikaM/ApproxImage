`timescale 1ns/1ps

module synthesis_exact (
    input  [7:0] A,
    input  [7:0] B,
    output [15:0] P
);

    structural_multiplier uut (
        .A(A),
        .B(B),
        .P(P)
    );

endmodule
