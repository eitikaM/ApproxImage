`timescale 1ns/1ps

module tb_check_configurable;
    reg  [7:0] A;
    reg  [7:0] B;
    reg  [1:0] mode;
    wire [15:0] P_net;

    reg [15:0] lut_mem [0:65535];
    integer a_int, b_int, idx;
    integer mismatches_mode [0:3];

    // Instantiate synthesized configurable netlist
    net_configurable net_inst (
        .A(A),
        .B(B),
        .mode(mode),
        .P(P_net)
    );

    initial begin
        mismatches_mode[0] = 0;
        mismatches_mode[1] = 0;
        mismatches_mode[2] = 0;
        mismatches_mode[3] = 0;

        // Mode 0
        $readmemh("results/synthesis_repro/lut_mode0.hex", lut_mem);
        mode = 2'b00;
        for (a_int = 0; a_int < 256; a_int = a_int + 1) begin
            for (b_int = 0; b_int < 256; b_int = b_int + 1) begin
                A = a_int[7:0];
                B = b_int[7:0];
                #1;
                idx = (a_int << 8) | b_int;
                if (P_net !== lut_mem[idx]) begin
                    mismatches_mode[0] = mismatches_mode[0] + 1;
                end
            end
        end
        $display("CONFIGURABLE NETLIST MODE 0: mismatches_vs_lut=%0d", mismatches_mode[0]);

        // Mode 1
        $readmemh("results/synthesis_repro/lut_mode1.hex", lut_mem);
        mode = 2'b01;
        for (a_int = 0; a_int < 256; a_int = a_int + 1) begin
            for (b_int = 0; b_int < 256; b_int = b_int + 1) begin
                A = a_int[7:0];
                B = b_int[7:0];
                #1;
                idx = (a_int << 8) | b_int;
                if (P_net !== lut_mem[idx]) begin
                    mismatches_mode[1] = mismatches_mode[1] + 1;
                end
            end
        end
        $display("CONFIGURABLE NETLIST MODE 1: mismatches_vs_lut=%0d", mismatches_mode[1]);

        // Mode 2
        $readmemh("results/synthesis_repro/lut_mode2.hex", lut_mem);
        mode = 2'b10;
        for (a_int = 0; a_int < 256; a_int = a_int + 1) begin
            for (b_int = 0; b_int < 256; b_int = b_int + 1) begin
                A = a_int[7:0];
                B = b_int[7:0];
                #1;
                idx = (a_int << 8) | b_int;
                if (P_net !== lut_mem[idx]) begin
                    mismatches_mode[2] = mismatches_mode[2] + 1;
                end
            end
        end
        $display("CONFIGURABLE NETLIST MODE 2: mismatches_vs_lut=%0d", mismatches_mode[2]);

        // Mode 3
        $readmemh("results/synthesis_repro/lut_mode3.hex", lut_mem);
        mode = 2'b11;
        for (a_int = 0; a_int < 256; a_int = a_int + 1) begin
            for (b_int = 0; b_int < 256; b_int = b_int + 1) begin
                A = a_int[7:0];
                B = b_int[7:0];
                #1;
                idx = (a_int << 8) | b_int;
                if (P_net !== lut_mem[idx]) begin
                    mismatches_mode[3] = mismatches_mode[3] + 1;
                end
            end
        end
        $display("CONFIGURABLE NETLIST MODE 3: mismatches_vs_lut=%0d", mismatches_mode[3]);

        $finish;
    end
endmodule
