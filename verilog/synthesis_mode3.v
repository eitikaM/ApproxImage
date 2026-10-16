`timescale 1ns/1ps

module synthesis_mode3 (
    input  [7:0] A,
    input  [7:0] B,
    output [15:0] P
);

    approximate_multiplier uut (
        .A(A),
        .B(B),
        .mode(2'b11),
        .P(P)
    );

endmodule
