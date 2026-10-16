
`timescale 1ns/1ps

module approximate_testbench;

    reg  [7:0] A;
    reg  [7:0] B;
    reg  [1:0] mode;
    wire [15:0] P_exact;
    wire [15:0] P_approx;

    exact_multiplier golden_uut (
        .A(A),
        .B(B),
        .P(P_exact)
    );

    approximate_multiplier approx_uut (
        .A(A),
        .B(B),
        .mode(mode),
        .P(P_approx)
    );

    integer m_idx;
    integer i, j;
    integer exact_matches;
    integer error_cases;
    integer max_error;
    integer diff;
    reg [63:0] total_error;
    real error_rate;
    real mae;

    task test_samples;
        input [1:0] test_mode;
        input [255:0] mode_name;
        begin
            mode = test_mode;

            $display("------------------------------------------------");
            $display(" Sample Inputs for Mode %0d (%0s):",
                     test_mode, mode_name);
            $display(" A | B | Exact Product | Approx Product | Abs Error");
            $display("------------------------------------------------");

            A = 8'd10; B = 8'd20; #10;
            diff = (P_exact >= P_approx) ?
                   (P_exact - P_approx) : (P_approx - P_exact);
            $display("%3d | %3d | %6d | %6d | %5d",
                     A, B, P_exact, P_approx, diff);

            A = 8'd15; B = 8'd7; #10;
            diff = (P_exact >= P_approx) ?
                   (P_exact - P_approx) : (P_approx - P_exact);
            $display("%3d | %3d | %6d | %6d | %5d",
                     A, B, P_exact, P_approx, diff);

            A = 8'd25; B = 8'd12; #10;
            diff = (P_exact >= P_approx) ?
                   (P_exact - P_approx) : (P_approx - P_exact);
            $display("%3d | %3d | %6d | %6d | %5d",
                     A, B, P_exact, P_approx, diff);

            A = 8'd100; B = 8'd5; #10;
            diff = (P_exact >= P_approx) ?
                   (P_exact - P_approx) : (P_approx - P_exact);
            $display("%3d | %3d | %6d | %6d | %5d",
                     A, B, P_exact, P_approx, diff);

            A = 8'd255; B = 8'd255; #10;
            diff = (P_exact >= P_approx) ?
                   (P_exact - P_approx) : (P_approx - P_exact);
            $display("%3d | %3d | %6d | %6d | %5d",
                     A, B, P_exact, P_approx, diff);

            A = 8'd0; B = 8'd128; #10;
            diff = (P_exact >= P_approx) ?
                   (P_exact - P_approx) : (P_approx - P_exact);
            $display("%3d | %3d | %6d | %6d | %5d",
                     A, B, P_exact, P_approx, diff);
        end
    endtask

    initial begin
        // Waveform generation
        $dumpfile("approximate_waveform.vcd");
        $dumpvars(0, approximate_testbench);

        $display("==============================================");
        $display(" CONFIGURABLE 8x8 APPROXIMATE MULTIPLIER");
        $display("==============================================");

        test_samples(2'b00, "Mode 0: Exact");
        test_samples(2'b01, "Mode 1: Low Approx");
        test_samples(2'b10, "Mode 2: Medium Approx");
        test_samples(2'b11, "Mode 3: High Approx");

        $display("\nEXHAUSTIVE EVALUATION");

        for (m_idx = 0; m_idx < 4; m_idx = m_idx + 1) begin
            mode = m_idx[1:0];
            exact_matches = 0;
            error_cases = 0;
            max_error = 0;
            total_error = 64'd0;

            for (i = 0; i < 256; i = i + 1) begin
                for (j = 0; j < 256; j = j + 1) begin
                    A = i[7:0];
                    B = j[7:0];
                    #1;

                    diff = (P_exact >= P_approx) ?
                           (P_exact - P_approx) : (P_approx - P_exact);

                    if (diff == 0)
                        exact_matches = exact_matches + 1;
                    else begin
                        error_cases = error_cases + 1;
                        total_error = total_error + diff;
                        if (diff > max_error)
                            max_error = diff;
                    end
                end
            end

            error_rate = (error_cases * 100.0) / 65536.0;
            mae = total_error / 65536.0;

            $display("----------------------------------------------");
            $display("MODE %0d", m_idx);
            $display("Total test cases: 65536");
            $display("Exact matches: %0d", exact_matches);
            $display("Error cases: %0d", error_cases);
            $display("Error rate: %0.2f%%", error_rate);
            $display("Maximum absolute error: %0d", max_error);
            $display("Mean absolute error: %0.4f", mae);
        end

        $display("==============================================");
        $finish;
    end

endmodule
