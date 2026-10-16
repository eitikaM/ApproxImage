`timescale 1ns/1ps

module structural_testbench;

    reg  [7:0] A;
    reg  [7:0] B;
    wire [15:0] P_exact;
    wire [15:0] P_structural;

    integer error_count;
    integer i, j;

    // Instantiate Golden Model (exact_multiplier using A * B)
    exact_multiplier golden_uut (
        .A(A),
        .B(B),
        .P(P_exact)
    );

    // Instantiate Structural Multiplier (without '*')
    structural_multiplier structural_uut (
        .A(A),
        .B(B),
        .P(P_structural)
    );

    initial begin
        error_count = 0;

        $display("=============================================================");
        $display("  PART 1: Sample Test Vectors Check                          ");
        $display("=============================================================");

        // Test vector 1
        A = 8'd10; B = 8'd20; #10;
        $display("A = %3d | B = %3d | Exact = %5d | Structural = %5d | Match = %s",
                 A, B, P_exact, P_structural, (P_exact === P_structural) ? "PASS" : "FAIL");
        if (P_exact !== P_structural) error_count = error_count + 1;

        // Test vector 2
        A = 8'd15; B = 8'd7; #10;
        $display("A = %3d | B = %3d | Exact = %5d | Structural = %5d | Match = %s",
                 A, B, P_exact, P_structural, (P_exact === P_structural) ? "PASS" : "FAIL");
        if (P_exact !== P_structural) error_count = error_count + 1;

        // Test vector 3
        A = 8'd25; B = 8'd12; #10;
        $display("A = %3d | B = %3d | Exact = %5d | Structural = %5d | Match = %s",
                 A, B, P_exact, P_structural, (P_exact === P_structural) ? "PASS" : "FAIL");
        if (P_exact !== P_structural) error_count = error_count + 1;

        // Test vector 4
        A = 8'd100; B = 8'd5; #10;
        $display("A = %3d | B = %3d | Exact = %5d | Structural = %5d | Match = %s",
                 A, B, P_exact, P_structural, (P_exact === P_structural) ? "PASS" : "FAIL");
        if (P_exact !== P_structural) error_count = error_count + 1;

        // Test vector 5
        A = 8'd255; B = 8'd255; #10;
        $display("A = %3d | B = %3d | Exact = %5d | Structural = %5d | Match = %s",
                 A, B, P_exact, P_structural, (P_exact === P_structural) ? "PASS" : "FAIL");
        if (P_exact !== P_structural) error_count = error_count + 1;

        // Test vector 6 (Edge case 0)
        A = 8'd0; B = 8'd128; #10;
        $display("A = %3d | B = %3d | Exact = %5d | Structural = %5d | Match = %s",
                 A, B, P_exact, P_structural, (P_exact === P_structural) ? "PASS" : "FAIL");
        if (P_exact !== P_structural) error_count = error_count + 1;

        $display("=============================================================");
        $display("  PART 2: Exhaustive Verification (All 65,536 combinations) ");
        $display("=============================================================");

        for (i = 0; i < 256; i = i + 1) begin
            for (j = 0; j < 256; j = j + 1) begin
                A = i[7:0];
                B = j[7:0];
                #1; // propagate combo logic
                if (P_exact !== P_structural) begin
                    error_count = error_count + 1;
                    if (error_count <= 5) begin
                        $display("MISMATCH at A=%d, B=%d: Exact=%d, Structural=%d",
                                 A, B, P_exact, P_structural);
                    end
                end
            end
        end

        $display("-------------------------------------------------------------");
        if (error_count == 0) begin
            $display("SUCCESS: All 65,536 test cases matched perfectly (0 errors)!");
            $display("Structural multiplier is 100%% verified against exact golden model.");
        end else begin
            $display("FAILURE: %0d total mismatches found!", error_count);
        end
        $display("=============================================================");

        $finish;
    end

endmodule
