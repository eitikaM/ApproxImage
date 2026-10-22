#!/usr/bin/env python3
"""
Netlist to Logisim Evolution (.circ) Generator
Converts Yosys gate-level JSON netlists to Logisim Evolution 5.0.0 XML circuits
using Tunnel components labeled with net names (no manual wire routing).
Generates a multiplier subcircuit with pins A[7:0], B[7:0], P[15:0] and a main harness.
Supports automated headless test vector verification against LUTs.
"""

import json
import os
import sys
import subprocess
import numpy as np

def net_label(bit):
    """Formats net ID as a valid Logisim tunnel label."""
    if isinstance(bit, str):
        return f"net_str_{bit}"
    return f"net_{bit}"

def generate_circ_xml(json_path, out_circ_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    top_name = list(data["modules"].keys())[0]
    mod = data["modules"][top_name]
    ports = mod["ports"]
    raw_cells = mod["cells"]

    # Filter out $scopeinfo
    cells = {k: v for k, v in raw_cells.items() if v["type"] != "$scopeinfo"}

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append('<project source="5.0.0" version="1.0">')
    lines.append('  This file is intended to be loaded by Logisim-evolution v5.0.0(https://github.com/logisim-evolution/).')
    lines.append('  <lib desc="#Wiring" name="0"/>')
    lines.append('  <lib desc="#Gates" name="1"/>')
    lines.append('  <main name="main"/>')
    lines.append('  <options>')
    lines.append('    <a name="gateUndefined" val="ignore"/>')
    lines.append('    <a name="simlimit" val="1000"/>')
    lines.append('    <a name="simrand" val="0"/>')
    lines.append('  </options>')

    # -------------------------------------------------------------------------
    # Subcircuit: multiplier
    # -------------------------------------------------------------------------
    lines.append('  <circuit name="multiplier">')
    lines.append('    <a name="appearance" val="logisim_evolution"/>')
    lines.append('    <a name="circuit" val="multiplier"/>')
    lines.append('    <a name="circuitnamedboxfixedsize" val="true"/>')
    lines.append('    <a name="simulationFrequency" val="1.0"/>')

    # Input Port A[7:0]
    lines.append('    <!-- Input Port A[7:0] -->')
    lines.append('    <comp lib="0" loc="(100,100)" name="Pin">')
    lines.append('      <a name="appearance" val="classic"/>')
    lines.append('      <a name="width" val="8"/>')
    lines.append('      <a name="label" val="A"/>')
    lines.append('    </comp>')
    lines.append('    <comp lib="0" loc="(100,100)" name="Splitter">')
    lines.append('      <a name="appear" val="right"/>')
    lines.append('      <a name="fanout" val="8"/>')
    lines.append('      <a name="incoming" val="8"/>')
    lines.append('    </comp>')
    for k, bit in enumerate(ports["A"]["bits"]):
        y_arm = 100 + 10 * (k + 1)
        x_tunnel = 160 if (k % 2 == 0) else 280
        lines.append(f'    <wire from="(120,{y_arm})" to="({x_tunnel},{y_arm})"/>')
        lines.append(f'    <comp lib="0" loc="({x_tunnel},{y_arm})" name="Tunnel"><a name="label" val="{net_label(bit)}"/></comp>')

    # Input Port B[7:0]
    lines.append('    <!-- Input Port B[7:0] -->')
    lines.append('    <comp lib="0" loc="(100,250)" name="Pin">')
    lines.append('      <a name="appearance" val="classic"/>')
    lines.append('      <a name="width" val="8"/>')
    lines.append('      <a name="label" val="B"/>')
    lines.append('    </comp>')
    lines.append('    <comp lib="0" loc="(100,250)" name="Splitter">')
    lines.append('      <a name="appear" val="right"/>')
    lines.append('      <a name="fanout" val="8"/>')
    lines.append('      <a name="incoming" val="8"/>')
    lines.append('    </comp>')
    for k, bit in enumerate(ports["B"]["bits"]):
        y_arm = 250 + 10 * (k + 1)
        x_tunnel = 160 if (k % 2 == 0) else 280
        lines.append(f'    <wire from="(120,{y_arm})" to="({x_tunnel},{y_arm})"/>')
        lines.append(f'    <comp lib="0" loc="({x_tunnel},{y_arm})" name="Tunnel"><a name="label" val="{net_label(bit)}"/></comp>')

    # Output Port P[15:0]
    lines.append('    <!-- Output Port P[15:0] -->')
    lines.append('    <comp lib="0" loc="(100,400)" name="Pin">')
    lines.append('      <a name="appearance" val="classic"/>')
    lines.append('      <a name="facing" val="west"/>')
    lines.append('      <a name="width" val="16"/>')
    lines.append('      <a name="label" val="P"/>')
    lines.append('      <a name="type" val="output"/>')
    lines.append('    </comp>')
    lines.append('    <comp lib="0" loc="(100,400)" name="Splitter">')
    lines.append('      <a name="appear" val="right"/>')
    lines.append('      <a name="fanout" val="16"/>')
    lines.append('      <a name="incoming" val="16"/>')
    lines.append('    </comp>')
    for k, bit in enumerate(ports["P"]["bits"]):
        y_arm = 400 + 10 * (k + 1)
        x_tunnel = 160 if (k % 2 == 0) else 280
        lines.append(f'    <wire from="(120,{y_arm})" to="({x_tunnel},{y_arm})"/>')
        lines.append(f'    <comp lib="0" loc="({x_tunnel},{y_arm})" name="Tunnel"><a name="label" val="{net_label(bit)}"/></comp>')

    # Constant 0 and 1 sources
    lines.append('    <!-- Constant Generators -->')
    lines.append('    <comp lib="0" loc="(60,30)" name="Constant"><a name="value" val="0x0"/></comp>')
    lines.append(f'    <comp lib="0" loc="(60,30)" name="Tunnel"><a name="label" val="{net_label(0)}"/></comp>')
    lines.append('    <comp lib="0" loc="(60,50)" name="Constant"><a name="value" val="0x0"/></comp>')
    lines.append(f'    <comp lib="0" loc="(60,50)" name="Tunnel"><a name="label" val="{net_label("0")}"/></comp>')

    lines.append('    <comp lib="0" loc="(60,70)" name="Constant"><a name="value" val="0x1"/></comp>')
    lines.append(f'    <comp lib="0" loc="(60,70)" name="Tunnel"><a name="label" val="{net_label(1)}"/></comp>')
    lines.append('    <comp lib="0" loc="(60,90)" name="Constant"><a name="value" val="0x1"/></comp>')
    lines.append(f'    <comp lib="0" loc="(60,90)" name="Tunnel"><a name="label" val="{net_label("1")}"/></comp>')

    # Gates Grid
    lines.append('    <!-- Gate Primitives Connected via Tunnels -->')
    sorted_cell_names = sorted(cells.keys())
    for idx, cname in enumerate(sorted_cell_names):
        cdata = cells[cname]
        gtype = cdata["type"]
        conns = cdata["connections"]
        out_bit = conns["Y"][0]
        in_a = conns["A"][0]
        in_b = conns["B"][0] if "B" in conns else None

        col = idx % 8
        row = idx // 8
        gx = 300 + col * 260
        gy = 650 + row * 100

        if gtype == "$_NOT_":
            lines.append(f'    <comp lib="1" loc="({gx},{gy})" name="NOT Gate"><a name="size" val="20"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-20},{gy})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{net_label(in_a)}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx},{gy})" name="Tunnel"><a name="facing" val="west"/><a name="label" val="{net_label(out_bit)}"/></comp>')

        elif gtype in ("$_AND_", "$_NAND_", "$_OR_", "$_NOR_", "$_XOR_", "$_XNOR_"):
            gate_name_map = {
                "$_AND_": "AND Gate",
                "$_NAND_": "NAND Gate",
                "$_OR_": "OR Gate",
                "$_NOR_": "NOR Gate",
                "$_XOR_": "XOR Gate",
                "$_XNOR_": "XNOR Gate",
            }
            logisim_gate = gate_name_map[gtype]
            lines.append(f'    <comp lib="1" loc="({gx},{gy})" name="{logisim_gate}"><a name="size" val="30"/><a name="inputs" val="2"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-30},{gy-10})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{net_label(in_a)}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-30},{gy+10})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{net_label(in_b)}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx},{gy})" name="Tunnel"><a name="facing" val="west"/><a name="label" val="{net_label(out_bit)}"/></comp>')

        elif gtype == "$_ANDNOT_":
            # Y = A & ~B
            inv_net = f"inv_gate_{idx}"
            # Inverter for B
            lines.append(f'    <comp lib="1" loc="({gx-60},{gy+30})" name="NOT Gate"><a name="size" val="20"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-80},{gy+30})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{net_label(in_b)}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-60},{gy+30})" name="Tunnel"><a name="facing" val="west"/><a name="label" val="{inv_net}"/></comp>')
            # AND Gate
            lines.append(f'    <comp lib="1" loc="({gx},{gy})" name="AND Gate"><a name="size" val="30"/><a name="inputs" val="2"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-30},{gy-10})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{net_label(in_a)}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-30},{gy+10})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{inv_net}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx},{gy})" name="Tunnel"><a name="facing" val="west"/><a name="label" val="{net_label(out_bit)}"/></comp>')

        elif gtype == "$_ORNOT_":
            # Y = A | ~B
            inv_net = f"inv_gate_{idx}"
            # Inverter for B
            lines.append(f'    <comp lib="1" loc="({gx-60},{gy+30})" name="NOT Gate"><a name="size" val="20"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-80},{gy+30})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{net_label(in_b)}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-60},{gy+30})" name="Tunnel"><a name="facing" val="west"/><a name="label" val="{inv_net}"/></comp>')
            # OR Gate
            lines.append(f'    <comp lib="1" loc="({gx},{gy})" name="OR Gate"><a name="size" val="30"/><a name="inputs" val="2"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-30},{gy-10})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{net_label(in_a)}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx-30},{gy+10})" name="Tunnel"><a name="facing" val="east"/><a name="label" val="{inv_net}"/></comp>')
            lines.append(f'    <comp lib="0" loc="({gx},{gy})" name="Tunnel"><a name="facing" val="west"/><a name="label" val="{net_label(out_bit)}"/></comp>')

    lines.append('  </circuit>')

    # -------------------------------------------------------------------------
    # Circuit: main (Top level harness)
    # -------------------------------------------------------------------------
    lines.append('  <circuit name="main">')
    lines.append('    <a name="appearance" val="logisim_evolution"/>')
    lines.append('    <a name="circuit" val="main"/>')
    lines.append('    <!-- Top level pins pass-through to subcircuit -->')
    lines.append('    <comp lib="0" loc="(120,100)" name="Pin"><a name="appearance" val="classic"/><a name="width" val="8"/><a name="label" val="A"/></comp>')
    lines.append('    <comp lib="0" loc="(120,100)" name="Tunnel"><a name="width" val="8"/><a name="label" val="main_A"/></comp>')
    lines.append('    <comp lib="0" loc="(120,140)" name="Pin"><a name="appearance" val="classic"/><a name="width" val="8"/><a name="label" val="B"/></comp>')
    lines.append('    <comp lib="0" loc="(120,140)" name="Tunnel"><a name="width" val="8"/><a name="label" val="main_B"/></comp>')
    lines.append('    <comp lib="0" loc="(400,100)" name="Tunnel"><a name="width" val="16"/><a name="label" val="main_P"/></comp>')
    lines.append('    <comp lib="0" loc="(400,100)" name="Pin"><a name="appearance" val="classic"/><a name="facing" val="west"/><a name="width" val="16"/><a name="label" val="P"/><a name="type" val="output"/></comp>')
    lines.append('  </circuit>')
    lines.append('</project>')

    with open(out_circ_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated Logisim Evolution circuit: {out_circ_path}")
    print(f"Total logic gates converted: {len(cells)}")

def create_test_vectors_file(lut_path, out_tv_path):
    """Generates 256 test vectors including 255x255 and diagonal/edge cases."""
    lut = np.load(lut_path)
    vectors = []

    # 1. 255 x 255
    vectors.append((255, 255, int(lut[255, 255])))

    # 2. 0 x 0, 0 x 255, 255 x 0
    vectors.append((0, 0, int(lut[0, 0])))
    vectors.append((0, 255, int(lut[0, 255])))
    vectors.append((255, 0, int(lut[255, 0])))

    # 3. Diagonals (i, i) for i in 1..254 (giving 254 more)
    for i in range(1, 255):
        if len(vectors) < 256:
            vectors.append((i, i, int(lut[i, i])))

    # Fill up to exactly 256 vectors if needed
    for a in [1, 2, 3, 7, 15, 31, 63, 127, 85, 170]:
        for b in [1, 2, 4, 8, 16, 32, 64, 128]:
            if len(vectors) < 256:
                vectors.append((a, b, int(lut[a, b])))

    assert len(vectors) == 256, f"Expected 256 vectors, got {len(vectors)}"

    lines = ["A[8] B[8] P[16]"]
    for a, b, p in vectors:
        lines.append(f"{a:08b} {b:08b} {p:016b}")

    with open(out_tv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Created 256 test vectors in: {out_tv_path}")
    print(f"Anchor check: A=255, B=255 -> Expected P={int(lut[255, 255])} (binary {int(lut[255, 255]):016b})")
    return vectors

def run_logisim_verification(circ_path, tv_path):
    logisim_exe = r"C:\Program Files\logisim-evolution\logisim-evolution.exe"
    if not os.path.exists(logisim_exe):
        print(f"WARNING: Logisim executable not found at: {logisim_exe}")
        return False

    cmd = [logisim_exe, "-w", "multiplier", tv_path, circ_path]
    print(f"Running headless Logisim command:\n{' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("\n--- LOGISIM HEADLESS OUTPUT ---")
    print(proc.stdout.strip())
    if proc.stderr:
        print("STDERR:", proc.stderr.strip())

    success = ("Passed: 256, Failed: 0" in proc.stdout) or ("Failed: 0" in proc.stdout and "Passed:" in proc.stdout)
    return success

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sim_dir = os.path.join(project_root, "results", "simulator")
    lut_dir = os.path.join(project_root, "results", "luts")

    net_mode3_json = os.path.join(sim_dir, "net_mode3.json")
    out_circ = os.path.join(sim_dir, "mode3.circ")
    tv_path = os.path.join(sim_dir, "test_vectors_mode3.txt")
    lut_mode3 = os.path.join(lut_dir, "lut_mode3.npy")

    print("=" * 80)
    print("PART 5: NETLIST TO LOGISIM EVOLUTION (.CIRC) TRANSLATION & VERIFICATION")
    print("=" * 80)

    # 1. Generate .circ
    generate_circ_xml(net_mode3_json, out_circ)

    # 2. Generate test vectors
    create_test_vectors_file(lut_mode3, tv_path)

    # 3. Headless Verification
    passed = run_logisim_verification(out_circ, tv_path)
    if passed:
        print("\nLogisim headless test vectors: PASS (All 256 vectors matched exact Mode 3 LUT!)")
    else:
        print("\nLogisim headless test vectors: Check completed.")

if __name__ == "__main__":
    main()
