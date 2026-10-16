@echo off
REM ============================================================================
REM ApproxImage - Mode Comparison RTL Simulation Runner
REM ============================================================================
set "PATH=D:\ucrt64\bin;D:\usr\bin;%PATH%"

echo [1/3] Compiling RTL with Icarus Verilog...
iverilog -o mode_comparison_sim structural_multiplier.v exact_multiplier.v approximate_multiplier.v mode_comparison_testbench.v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo [2/3] Running simulation with vvp...
vvp mode_comparison_sim

echo [3/3] Opening waveform in GTKWave...
start gtkwave mode_comparison.vcd mode_comparison.gtkw

echo Done.
