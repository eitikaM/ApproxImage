import subprocess
import os
import re
import csv

YOSYS_PATH = r"D:\ucrt64\bin\yosys.exe"
VERILOG_DIR = r"C:\Users\ASUS\OneDrive\Desktop\final_year\approximate_multipliers\verilog"
RESULTS_DIR = r"C:\Users\ASUS\OneDrive\Desktop\final_year\approximate_multipliers\results"

os.makedirs(RESULTS_DIR, exist_ok=True)

targets = [
    {
        "name": "Exact (Structural CSA)",
        "id": "exact",
        "top": "structural_multiplier",
        "files": ["structural_multiplier.v"]
    },
    {
        "name": "Fixed Mode 1 (Low Approx)",
        "id": "mode1",
        "top": "synthesis_mode1",
        "files": ["structural_multiplier.v", "approximate_multiplier.v", "synthesis_mode1.v"]
    },
    {
        "name": "Fixed Mode 2 (Med Approx)",
        "id": "mode2",
        "top": "synthesis_mode2",
        "files": ["structural_multiplier.v", "approximate_multiplier.v", "synthesis_mode2.v"]
    },
    {
        "name": "Fixed Mode 3 (High Approx)",
        "id": "mode3",
        "top": "synthesis_mode3",
        "files": ["structural_multiplier.v", "approximate_multiplier.v", "synthesis_mode3.v"]
    },
    {
        "name": "Configurable (Runtime Mode)",
        "id": "configurable",
        "top": "approximate_multiplier",
        "files": ["structural_multiplier.v", "approximate_multiplier.v"]
    }
]

env = os.environ.copy()
env["PATH"] = r"D:\ucrt64\bin;D:\usr\bin;" + env.get("PATH", "")

results = []

for t in targets:
    file_args = " ".join([os.path.join(VERILOG_DIR, f).replace("\\", "/") for f in t["files"]])
    top_mod = t["top"]
    
    script = f"""
read_verilog {file_args}
synth -top {top_mod}
flatten
opt
abc -g gates
stat
ltp
"""
    script_file = os.path.join(RESULTS_DIR, f"synth_{t['id']}.ys")
    with open(script_file, "w") as f:
        f.write(script)
        
    log_file = os.path.join(RESULTS_DIR, f"synth_{t['id']}.log")
    
    print(f"Synthesizing {t['name']}...")
    cmd = [YOSYS_PATH, "-s", script_file]
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    with open(log_file, "w") as f:
        f.write(proc.stdout)
        
    # Extract LTP
    ltp_len = "N/A"
    m_ltp = re.findall(r"Longest topological path in .*?\(length=(\d+)\)", proc.stdout)
    if m_ltp:
        ltp_len = int(m_ltp[-1])
        
    # Extract the LAST stat block
    stat_blocks = proc.stdout.split(f"=== {top_mod} ===")
    if len(stat_blocks) > 1:
        stat_text = stat_blocks[-1]
    else:
        stat_text = proc.stdout

    total_cells = 0
    m_tot = re.search(r"(\d+)\s+cells", stat_text)
    if m_tot:
        total_cells = int(m_tot.group(1))
        
    wires = 0
    m_wire = re.search(r"(\d+)\s+wires", stat_text)
    if m_wire:
        wires = int(m_wire.group(1))
        
    wire_bits = 0
    m_wbits = re.search(r"(\d+)\s+wire bits", stat_text)
    if m_wbits:
        wire_bits = int(m_wbits.group(1))
        
    gate_counts = {
        "AND": 0, "NAND": 0, "OR": 0, "NOR": 0, "XOR": 0, "XNOR": 0,
        "NOT": 0, "ANDNOT": 0, "ORNOT": 0, "MUX": 0, "scopeinfo": 0
    }
    
    for g in gate_counts.keys():
        if g == "scopeinfo":
            m_g = re.search(r"(\d+)\s+\$scopeinfo", stat_text)
        else:
            m_g = re.search(r"(\d+)\s+\$_" + g + r"_", stat_text)
        if m_g:
            gate_counts[g] = int(m_g.group(1))
            
    # Pure logic cells excluding scopeinfo
    logic_cells = total_cells - gate_counts.get("scopeinfo", 0)
    
    res_entry = {
        "Design": t["name"],
        "Top": top_mod,
        "Total Cells": total_cells,
        "Logic Cells": logic_cells,
        "AND": gate_counts["AND"],
        "NAND": gate_counts["NAND"],
        "OR": gate_counts["OR"],
        "NOR": gate_counts["NOR"],
        "XOR": gate_counts["XOR"],
        "XNOR": gate_counts["XNOR"],
        "NOT": gate_counts["NOT"],
        "ANDNOT": gate_counts["ANDNOT"],
        "ORNOT": gate_counts["ORNOT"],
        "MUX": gate_counts["MUX"],
        "Wires": wires,
        "Wire Bits": wire_bits,
        "LTP (Gate Levels)": ltp_len
    }
    results.append(res_entry)

# Write CSV
csv_file = os.path.join(RESULTS_DIR, "synthesis_results.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("Saved CSV:", csv_file)

# Write TXT report
txt_file = os.path.join(RESULTS_DIR, "synthesis_report.txt")
exact_logic = results[0]["Logic Cells"]
exact_ltp = results[0]["LTP (Gate Levels)"]

with open(txt_file, "w") as f:
    f.write("=" * 86 + "\n")
    f.write("                 YOSYS HARDWARE SYNTHESIS EVALUATION REPORT\n")
    f.write("=" * 86 + "\n\n")
    f.write("Note: Synthesized cell count and topological path length represent\n")
    f.write("synthesis-level logic resource usage and combinational delay proxies.\n")
    f.write("Power estimation is not available in the current synthesis flow.\n\n")
    
    f.write(f"{'Design':<28} | {'Logic Cells':<11} | {'Reduction (%)':<13} | {'LTP (Levels)':<12} | {'Wires':<6}\n")
    f.write("-" * 86 + "\n")
    
    for r in results:
        cell_red = "Baseline (0.00%)"
        if r["Design"] != results[0]["Design"]:
            diff = exact_logic - r["Logic Cells"]
            red_pct = (diff / exact_logic) * 100.0
            cell_red = f"{red_pct:+.2f}%"
        f.write(f"{r['Design']:<28} | {r['Logic Cells']:<11} | {cell_red:<13} | {str(r['LTP (Gate Levels)']):<12} | {r['Wires']:<6}\n")
    
    f.write("-" * 86 + "\n\n")
    f.write("DETAILED GATE-LEVEL BREAKDOWN:\n")
    f.write("-" * 86 + "\n")
    header = f"{'Design':<28} | {'AND':<5} | {'NAND':<5} | {'OR':<5} | {'XOR':<5} | {'XNOR':<5} | {'NOT':<5} | {'ANDNOT':<6} | {'MUX':<4}\n"
    f.write(header)
    f.write("-" * 86 + "\n")
    for r in results:
        row = f"{r['Design']:<28} | {r['AND']:<5} | {r['NAND']:<5} | {r['OR']:<5} | {r['XOR']:<5} | {r['XNOR']:<5} | {r['NOT']:<5} | {r['ANDNOT']:<6} | {r['MUX']:<4}\n"
        f.write(row)
    f.write("-" * 86 + "\n")

print("Saved Report:", txt_file)
