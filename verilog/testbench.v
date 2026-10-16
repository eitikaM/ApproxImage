`timescale 1ns/1ps

module testbench;

    reg  [7:0] A;
    reg  [7:0] B;
    wire [15:0] P;

    exact_multiplier uut (
        .A(A),
        .B(B),
        .P(P)
    );

    initial begin

        $monitor("Time=%0t | A=%d | B=%d | P=%d", 
                 $time, A, B, P);

        A = 8'd10;  B = 8'd20;
        #10;

        A = 8'd15;  B = 8'd7;
        #10;

        A = 8'd25;  B = 8'd12;
        #10;

        A = 8'd100; B = 8'd5;
        #10;

        A = 8'd255; B = 8'd255;
        #10;

        $finish;
    end

endmodule