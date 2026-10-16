`timescale 1ns/1ps

module export_rtl_truth_table;

    reg  [7:0] A;
    reg  [7:0] B;
    reg  [1:0] mode;
    wire [15:0] P;

    integer file_handle;
    integer m, i, j;

    approximate_multiplier uut (
        .A(A),
        .B(B),
        .mode(mode),
        .P(P)
    );

    initial begin
        file_handle = $fopen("../results/rtl_truth_table.txt", "w");
        if (file_handle == 0) begin
            file_handle = $fopen("results/rtl_truth_table.txt", "w");
        end

        for (m = 0; m < 4; m = m + 1) begin
            mode = m[1:0];
            for (i = 0; i < 256; i = i + 1) begin
                for (j = 0; j < 256; j = j + 1) begin
                    A = i[7:0];
                    B = j[7:0];
                    #1;
                    $fdisplay(file_handle, "%0d", P);
                end
            end
        end

        $fclose(file_handle);
        $display("RTL truth table exported successfully (262,144 entries).");
        $finish;
    end

endmodule
