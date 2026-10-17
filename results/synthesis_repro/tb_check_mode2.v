`timescale 1ns/1ps

module tb_check_mode2;
    reg  [7:0] A;
    reg  [7:0] B;
    wire [15:0] p_rtl;
    wire [15:0] p_net;

    reg [15:0] lut_mem [0:65535];
    integer a_int, b_int, idx;
    integer mismatches_rtl = 0;
    integer mismatches_lut = 0;

    // Instantiate RTL wrapper
    synthesis_mode2 rtl_inst (
        .A(A),
        .B(B),
        .P(p_rtl)
    );

    // Instantiate synthesized netlist
    net_mode2 net_inst (
        .A(A),
        .B(B),
        .P(p_net)
    );

    initial begin
        $readmemh("results/synthesis_repro/lut_mode2.hex", lut_mem);
        for (a_int = 0; a_int < 256; a_int = a_int + 1) begin
            for (b_int = 0; b_int < 256; b_int = b_int + 1) begin
                A = a_int[7:0];
                B = b_int[7:0];
                #1; // propagate combinational logic
                idx = (a_int << 8) | b_int;
                if (p_net !== p_rtl) begin
                    mismatches_rtl = mismatches_rtl + 1;
                end
                if (p_net !== lut_mem[idx]) begin
                    mismatches_lut = mismatches_lut + 1;
                end
            end
        end
        $display("MODE 2 RESULTS: mismatches_vs_rtl=%0d, mismatches_vs_lut=%0d", mismatches_rtl, mismatches_lut);
        $finish;
    end
endmodule
