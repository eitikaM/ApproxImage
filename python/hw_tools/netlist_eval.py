#!/usr/bin/env python3
"""
Hardware Netlist Evaluator
Evaluates gate-level JSON netlists exported from Yosys.
Supports topological sorting of gate-level primitives and vectorized numpy evaluation.
"""

import json
import os
import sys
from collections import defaultdict, deque
import numpy as np

SUPPORTED_GATES = {
    '$_AND_', '$_OR_', '$_XOR_', '$_NAND_', '$_NOR_',
    '$_XNOR_', '$_NOT_', '$_ANDNOT_', '$_ORNOT_'
}

class NetlistEvaluator:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.top_module_name = list(data['modules'].keys())[0]
        mod = data['modules'][self.top_module_name]
        self.ports = mod['ports']
        
        # Filter cells, excluding $scopeinfo
        raw_cells = mod['cells']
        self.cells = {}
        for cname, cdata in raw_cells.items():
            ctype = cdata['type']
            if ctype != '$scopeinfo':
                if ctype not in SUPPORTED_GATES:
                    raise ValueError(f"Unsupported gate type: {ctype} in cell {cname}")
                self.cells[cname] = cdata
        
        self.sorted_cells = self._topological_sort()

    def _topological_sort(self):
        """Topologically sorts gates using Kahn's algorithm."""
        in_degree = {}
        dependents = defaultdict(list)
        cell_inputs = {}
        cell_output = {}

        # Set of external input nets and constants
        known_nets = set()
        for pname, pdata in self.ports.items():
            if pdata['direction'] == 'input':
                for b in pdata['bits']:
                    known_nets.add(b)
        # Constants
        known_nets.update({'0', '1', 0, 1})

        for cname, cdata in self.cells.items():
            conns = cdata['connections']
            out_bit = conns['Y'][0]
            cell_output[cname] = out_bit

            in_bits = [conns['A'][0]]
            if 'B' in conns:
                in_bits.append(conns['B'][0])
            cell_inputs[cname] = in_bits

            for in_b in in_bits:
                dependents[in_b].append(cname)

        for cname, in_bits in cell_inputs.items():
            unresolved = sum(1 for b in in_bits if b not in known_nets)
            in_degree[cname] = unresolved

        queue = deque([cname for cname, deg in in_degree.items() if deg == 0])
        sorted_order = []

        while queue:
            curr = queue.popleft()
            sorted_order.append(curr)
            out_b = cell_output[curr]
            for dep in dependents[out_b]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(sorted_order) != len(self.cells):
            raise RuntimeError(
                f"Topological sort failed: {len(sorted_order)} of {len(self.cells)} gates resolved. Graph may have cycles."
            )

        return sorted_order

    def evaluate_vectorized(self, A_arr, B_arr, mode_val=None):
        """
        Evaluates netlist over arrays of inputs A and B.
        A_arr, B_arr: 1D numpy uint16/uint8 arrays of identical length N.
        mode_val: None or int (0, 1, 2, 3) or 1D array of mode values.
        Returns: 1D numpy array of outputs P (uint32).
        """
        A_arr = np.asarray(A_arr, dtype=np.uint16)
        B_arr = np.asarray(B_arr, dtype=np.uint16)
        N = len(A_arr)

        net_vals = {
            '0': np.zeros(N, dtype=bool),
            '1': np.ones(N, dtype=bool),
            0: np.zeros(N, dtype=bool),
            1: np.ones(N, dtype=bool)
        }

        # Assign input ports
        a_bits = self.ports['A']['bits']
        for i, b in enumerate(a_bits):
            net_vals[b] = ((A_arr >> i) & 1).astype(bool)

        b_bits = self.ports['B']['bits']
        for i, b in enumerate(b_bits):
            net_vals[b] = ((B_arr >> i) & 1).astype(bool)

        if 'mode' in self.ports:
            if mode_val is None:
                raise ValueError("Configurable netlist requires 'mode_val' argument.")
            if isinstance(mode_val, int):
                mode_arr = np.full(N, mode_val, dtype=np.uint8)
            else:
                mode_arr = np.asarray(mode_val, dtype=np.uint8)
            mode_bits = self.ports['mode']['bits']
            for i, b in enumerate(mode_bits):
                net_vals[b] = ((mode_arr >> i) & 1).astype(bool)

        # Evaluate sorted gates
        for cname in self.sorted_cells:
            cdata = self.cells[cname]
            gtype = cdata['type']
            conns = cdata['connections']
            out_bit = conns['Y'][0]
            in_a = net_vals[conns['A'][0]]

            if gtype == '$_NOT_':
                net_vals[out_bit] = ~in_a
            else:
                in_b = net_vals[conns['B'][0]]
                if gtype == '$_AND_':
                    net_vals[out_bit] = in_a & in_b
                elif gtype == '$_NAND_':
                    net_vals[out_bit] = ~(in_a & in_b)
                elif gtype == '$_OR_':
                    net_vals[out_bit] = in_a | in_b
                elif gtype == '$_NOR_':
                    net_vals[out_bit] = ~(in_a | in_b)
                elif gtype == '$_XOR_':
                    net_vals[out_bit] = in_a ^ in_b
                elif gtype == '$_XNOR_':
                    net_vals[out_bit] = ~(in_a ^ in_b)
                elif gtype == '$_ANDNOT_':
                    net_vals[out_bit] = in_a & (~in_b)
                elif gtype == '$_ORNOT_':
                    net_vals[out_bit] = in_a | (~in_b)
                else:
                    raise ValueError(f"Unhandled gate: {gtype}")

        # Assemble output P
        p_bits = self.ports['P']['bits']
        P_res = np.zeros(N, dtype=np.uint32)
        for i, b in enumerate(p_bits):
            P_res |= (net_vals[b].astype(np.uint32) << i)

        return P_res

    def evaluate_all_65536(self, mode_val=None):
        """
        Evaluates all 65,536 (A, B) pairs with A in rows (0..255) and B in columns (0..255).
        Returns a 2D numpy array of shape (256, 256) uint16.
        """
        a_grid, b_grid = np.meshgrid(np.arange(256, dtype=np.uint16), np.arange(256, dtype=np.uint16), indexing='ij')
        a_flat = a_grid.ravel()
        b_flat = b_grid.ravel()

        p_flat = self.evaluate_vectorized(a_flat, b_flat, mode_val=mode_val)
        return p_flat.reshape((256, 256)).astype(np.uint16)

def run_verification():
    print("=" * 80)
    print("PART 1 & PART 2: NETLIST EVALUATOR & VERIFICATION")
    print("=" * 80)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sim_dir = os.path.join(project_root, "results", "simulator")
    lut_dir = os.path.join(project_root, "results", "luts")

    netlists = {
        0: ("Mode 0 (Fixed Exact)", os.path.join(sim_dir, "net_mode0.json"), 366),
        1: ("Mode 1 (Fixed Low Approx)", os.path.join(sim_dir, "net_mode1.json"), 353),
        2: ("Mode 2 (Fixed Med Approx)", os.path.join(sim_dir, "net_mode2.json"), 317),
        3: ("Mode 3 (Fixed High Approx)", os.path.join(sim_dir, "net_mode3.json"), 262),
    }
    cfg_netlist = ("Configurable Runtime Design", os.path.join(sim_dir, "net_configurable.json"), 431)

    print("\n--- CELL COUNT VERIFICATION (Excluding $scopeinfo) ---")
    all_counts_match = True
    for mode, (label, path, expected) in netlists.items():
        evaluator = NetlistEvaluator(path)
        actual = len(evaluator.cells)
        match = (actual == expected)
        all_counts_match = all_counts_match and match
        print(f"[{label}] Cells: {actual} (Expected: {expected}) -> {'PASS' if match else 'FAIL'}")

    cfg_evaluator = NetlistEvaluator(cfg_netlist[1])
    cfg_actual = len(cfg_evaluator.cells)
    cfg_match = (cfg_actual == cfg_netlist[2])
    all_counts_match = all_counts_match and cfg_match
    print(f"[{cfg_netlist[0]}] Cells: {cfg_actual} (Expected: {cfg_netlist[2]}) -> {'PASS' if cfg_match else 'FAIL'}")

    print("\n--- FIXED-MODE NETLIST EVALUATION VS LUTS (65,536 PAIRS) ---")
    print(f"{'Design':<28} | {'Cells':<6} | {'Mismatches':<10} | {'Status':<6} | {'A=255,B=255 Output':<18}")
    print("-" * 78)

    outputs_255 = {}
    for mode in range(4):
        label, path, expected = netlists[mode]
        evaluator = NetlistEvaluator(path)
        eval_matrix = evaluator.evaluate_all_65536()
        
        lut_path = os.path.join(lut_dir, f"lut_mode{mode}.npy")
        lut_matrix = np.load(lut_path)

        mismatches = int(np.sum(eval_matrix != lut_matrix))
        out_255 = int(eval_matrix[255, 255])
        outputs_255[mode] = out_255

        status = "PASS" if mismatches == 0 else "FAIL"
        print(f"{label:<28} | {len(evaluator.cells):<6} | {mismatches:<10} | {status:<6} | {out_255:<18}")

    print("\n--- CONFIGURABLE NETLIST EVALUATION FOR MODES 0-3 (65,536 PAIRS EACH) ---")
    print(f"{'Configured Mode':<28} | {'Mode Bits':<9} | {'Mismatches':<10} | {'Status':<6} | {'A=255,B=255 Output':<18}")
    print("-" * 78)

    for mode in range(4):
        eval_matrix = cfg_evaluator.evaluate_all_65536(mode_val=mode)
        lut_path = os.path.join(lut_dir, f"lut_mode{mode}.npy")
        lut_matrix = np.load(lut_path)

        mismatches = int(np.sum(eval_matrix != lut_matrix))
        out_255 = int(eval_matrix[255, 255])
        status = "PASS" if mismatches == 0 else "FAIL"
        mode_bin = f"2'b{mode:02b}"
        print(f"Mode {mode} on Configurable        | {mode_bin:<9} | {mismatches:<10} | {status:<6} | {out_255:<18}")

    print("\n--- SUMMARY OF A=255, B=255 OUTPUTS ---")
    print(f"Mode 0 (Exact):       {outputs_255[0]} (Expected: 65025)")
    print(f"Mode 1 (Low Approx):  {outputs_255[1]} (Expected: 65015)")
    print(f"Mode 2 (Med Approx):  {outputs_255[2]} (Expected: 64927)")
    print(f"Mode 3 (High Approx): {outputs_255[3]} (Expected: 64383)")

    return all_counts_match

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
