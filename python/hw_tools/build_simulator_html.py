#!/usr/bin/env python3
"""
Generator for results/simulator/simulator.html
Embeds the topologically sorted gate netlists for Modes 0-3 and creates
a self-contained single-file HTML/CSS/JS interactive hardware simulator.
"""

import os
import json
import numpy as np
from netlist_eval import NetlistEvaluator

TYPE_MAP = {
    '$_NOT_': 0,
    '$_AND_': 1,
    '$_NAND_': 2,
    '$_OR_': 3,
    '$_NOR_': 4,
    '$_XOR_': 5,
    '$_XNOR_': 6,
    '$_ANDNOT_': 7,
    '$_ORNOT_': 8
}

def generate_html():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sim_dir = os.path.join(project_root, "results", "simulator")
    lut_dir = os.path.join(project_root, "results", "luts")
    out_html_path = os.path.join(sim_dir, "simulator.html")

    # 1. Compute checksums from LUTs
    checksums = {}
    for m in range(4):
        lut = np.load(os.path.join(lut_dir, f"lut_mode{m}.npy"))
        checksums[m] = {
            'sum': int(np.sum(lut, dtype=np.uint64)),
            'xor': int(np.bitwise_xor.reduce(lut.ravel())),
            'val_255_255': int(lut[255, 255])
        }

    # 2. Extract compact netlists
    netlists_bundle = {}
    for m in range(4):
        json_path = os.path.join(sim_dir, f"net_mode{m}.json")
        ev = NetlistEvaluator(json_path)
        gate_list = []
        max_net = 0
        for cname in ev.sorted_cells:
            c = ev.cells[cname]
            gtype = TYPE_MAP[c['type']]
            out_b = c['connections']['Y'][0]
            in_a = c['connections']['A'][0]
            in_b = c['connections']['B'][0] if 'B' in c['connections'] else 0
            gate_list.append([gtype, out_b, in_a, in_b])
            max_net = max(max_net, out_b, in_a, in_b)

        netlists_bundle[m] = {
            'aBits': ev.ports['A']['bits'],
            'bBits': ev.ports['B']['bits'],
            'pBits': ev.ports['P']['bits'],
            'maxNet': max_net + 1,
            'gateCount': len(gate_list),
            'gates': gate_list
        }

    bundle_json = json.dumps(netlists_bundle)
    checksums_json = json.dumps(checksums)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8x8 Approximate Multiplier Gate-Level Hardware Simulator</title>
<style>
  :root {{
    --bg-primary: #0a0e17;
    --bg-secondary: #131b2e;
    --bg-card: #182238;
    --bg-card-hover: #1f2c48;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-purple: #8b5cf6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border-color: #24324f;
    --border-glow: rgba(59, 130, 246, 0.35);
    --approx-color: #f59e0b;
    --diff-color: #f43f5e;
    --exact-color: #10b981;
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 24px;
    min-height: 100vh;
  }}

  .container {{
    max-width: 1280px;
    margin: 0 auto;
  }}

  header {{
    text-align: center;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
  }}

  header h1 {{
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}

  header p {{
    color: var(--text-secondary);
    font-size: 14px;
    max-width: 800px;
    margin: 0 auto;
  }}

  .badge-bar {{
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 12px;
    flex-wrap: wrap;
  }}

  .badge {{
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #93c5fd;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
    font-family: monospace;
  }}

  /* Top Control Panel */
  .control-panel {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
  }}

  .inputs-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}

  @media (max-width: 768px) {{
    .inputs-grid {{
      grid-template-columns: 1fr;
    }}
  }}

  .input-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
  }}

  .input-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}

  .input-title {{
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }}

  .input-val-box {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .input-num {{
    width: 80px;
    background: #0d1526;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: #60a5fa;
    font-family: monospace;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    padding: 6px;
    outline: none;
  }}

  .input-num:focus {{
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
  }}

  .slider {{
    width: 100%;
    margin: 12px 0;
    accent-color: var(--accent-blue);
    cursor: pointer;
  }}

  .representation-pills {{
    display: flex;
    gap: 12px;
    background: #0d1526;
    padding: 8px 12px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 13px;
    justify-content: space-between;
  }}

  .rep-item {{
    display: flex;
    gap: 6px;
  }}

  .rep-label {{
    color: var(--text-muted);
  }}

  .rep-val {{
    color: var(--accent-cyan);
    font-weight: 600;
  }}

  .actions-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
    flex-wrap: wrap;
    gap: 12px;
  }}

  .btn-group {{
    display: flex;
    gap: 10px;
  }}

  .btn {{
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}

  .btn-primary {{
    background: #2563eb;
    color: white;
  }}

  .btn-primary:hover {{
    background: #1d4ed8;
  }}

  .btn-amber {{
    background: #d97706;
    color: white;
  }}

  .btn-amber:hover {{
    background: #b45309;
  }}

  .btn-outline {{
    background: transparent;
    border-color: var(--border-color);
    color: var(--text-secondary);
  }}

  .btn-outline:hover {{
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
  }}

  .test-status {{
    font-family: monospace;
    font-size: 13px;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 600;
  }}

  .test-status.idle {{
    background: #1e293b;
    color: var(--text-muted);
  }}

  .test-status.pass {{
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
  }}

  .test-status.fail {{
    background: rgba(244, 63, 94, 0.2);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #fb7185;
  }}

  /* Outputs Grid */
  .outputs-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }}

  .mode-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 18px;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}

  .mode-card:hover {{
    border-color: rgba(59, 130, 246, 0.4);
  }}

  .mode-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 8px;
  }}

  .mode-title-group {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .mode-tag {{
    font-size: 12px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    text-transform: uppercase;
  }}

  .mode-tag.m0 {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
  .mode-tag.m1 {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; }}
  .mode-tag.m2 {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
  .mode-tag.m3 {{ background: rgba(244, 63, 94, 0.2); color: #fb7185; }}

  .mode-name {{
    font-size: 15px;
    font-weight: 600;
  }}

  .mode-stats {{
    display: flex;
    align-items: center;
    gap: 16px;
    font-family: monospace;
    font-size: 13px;
  }}

  .mode-dec {{
    font-size: 18px;
    font-weight: bold;
    color: #f8fafc;
  }}

  .stat-badge {{
    background: #0d1526;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
  }}

  .err-exact {{
    color: #10b981;
  }}

  .err-diff {{
    color: #f43f5e;
    font-weight: bold;
  }}

  /* Bit visualizer */
  .bit-strip {{
    display: grid;
    grid-template-columns: repeat(16, minmax(28px, 1fr));
    gap: 4px;
    margin-top: 10px;
  }}

  .bit-cell {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 6px 2px;
    border-radius: 6px;
    background: #0f172a;
    border: 1px solid #1e293b;
    font-family: monospace;
    transition: all 0.15s ease;
  }}

  .bit-idx {{
    font-size: 10px;
    color: var(--text-muted);
    margin-bottom: 2px;
  }}

  .bit-val {{
    font-size: 14px;
    font-weight: bold;
    color: #94a3b8;
  }}

  .bit-cell.val-1 .bit-val {{
    color: #38bdf8;
  }}

  .bit-cell.or-approx {{
    border-color: rgba(245, 158, 11, 0.7);
    background: rgba(245, 158, 11, 0.08);
  }}

  .bit-cell.or-approx .bit-idx {{
    color: #fbbf24;
    font-weight: bold;
  }}

  .bit-cell.bit-mismatch {{
    background: rgba(244, 63, 94, 0.25);
    border-color: #f43f5e;
    box-shadow: 0 0 8px rgba(244, 63, 94, 0.4);
  }}

  .bit-cell.bit-mismatch .bit-val {{
    color: #ff6b81;
  }}

  .bit-legend {{
    display: flex;
    gap: 16px;
    margin-top: 16px;
    font-size: 12px;
    color: var(--text-secondary);
    justify-content: flex-end;
    flex-wrap: wrap;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }}

  footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>8x8 Approximate Multiplier Gate-Level Hardware Simulator</h1>
    <p>Cycle-accurate combinational evaluation of Yosys-synthesized netlists with topological sort. Completely self-contained JavaScript simulation running entirely in the client.</p>
    <div class="badge-bar">
      <span class="badge">Mode 0: 366 Gates (Exact CSA)</span>
      <span class="badge">Mode 1: 353 Gates (OR cols 0..2)</span>
      <span class="badge">Mode 2: 317 Gates (OR cols 0..4)</span>
      <span class="badge">Mode 3: 262 Gates (OR cols 0..6)</span>
    </div>
  </header>

  <!-- Controls -->
  <section class="control-panel">
    <div class="inputs-grid">
      <!-- Operand A -->
      <div class="input-card">
        <div class="input-header">
          <span class="input-title">Operand A (8-bit Multiplicand)</span>
          <div class="input-val-box">
            <input type="number" id="numA" class="input-num" min="0" max="255" value="255">
          </div>
        </div>
        <input type="range" id="sliderA" class="slider" min="0" max="255" value="255">
        <div class="representation-pills">
          <div class="rep-item"><span class="rep-label">DEC:</span><span id="repADec" class="rep-val">255</span></div>
          <div class="rep-item"><span class="rep-label">HEX:</span><span id="repAHex" class="rep-val">0xFF</span></div>
          <div class="rep-item"><span class="rep-label">BIN:</span><span id="repABin" class="rep-val">11111111</span></div>
        </div>
      </div>

      <!-- Operand B -->
      <div class="input-card">
        <div class="input-header">
          <span class="input-title">Operand B (8-bit Multiplier)</span>
          <div class="input-val-box">
            <input type="number" id="numB" class="input-num" min="0" max="255" value="255">
          </div>
        </div>
        <input type="range" id="sliderB" class="slider" min="0" max="255" value="255">
        <div class="representation-pills">
          <div class="rep-item"><span class="rep-label">DEC:</span><span id="repBDec" class="rep-val">255</span></div>
          <div class="rep-item"><span class="rep-label">HEX:</span><span id="repBHex" class="rep-val">0xFF</span></div>
          <div class="rep-item"><span class="rep-label">BIN:</span><span id="repBBin" class="rep-val">11111111</span></div>
        </div>
      </div>
    </div>

    <div class="actions-bar">
      <div class="btn-group">
        <button id="btnRun255" class="btn btn-amber">Run 255 &times; 255</button>
        <button id="btnZero" class="btn btn-outline">Zero (0 &times; 0)</button>
        <button id="btnRandom" class="btn btn-outline">Random Inputs</button>
        <button id="btnSelfTest" class="btn btn-primary">Run 65,536 Exhaustive Self-Test</button>
      </div>
      <div id="testStatus" class="test-status idle">Self-Test: Not run</div>
    </div>
  </section>

  <!-- Bit visualizer legend -->
  <div class="bit-legend">
    <div class="legend-item">
      <span class="legend-dot" style="background:#0f172a; border: 1px solid #1e293b;"></span>
      <span>Exact Bit</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background:rgba(245, 158, 11, 0.4); border: 1px solid #fbbf24;"></span>
      <span>OR-Approximated Bit Zone</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background:rgba(244, 63, 94, 0.6); border: 1px solid #f43f5e;"></span>
      <span>Bit Differs from Mode 0</span>
    </div>
  </div>

  <!-- Multiplier Outputs Grid -->
  <main class="outputs-grid" id="outputsContainer">
    <!-- Rendered dynamically -->
  </main>

  <footer>
    Hardware Synthesis &bull; Structural CSA Baseline &bull; Lower-Part-OR Approximation Schemes &bull; Autonomous Netlist Simulator
  </footer>
</div>

<script>
// Embedded Topologically Sorted Netlists
const NETLISTS = {bundle_json};
const CHECKSUMS = {checksums_json};

// Approximation boundaries: number of lower output bits OR-approximated
const APPROX_BITS = {{
  0: 0, // Mode 0: None
  1: 3, // Mode 1: bits [2:0]
  2: 5, // Mode 2: bits [4:0]
  3: 7  // Mode 3: bits [6:0]
}};

const MODE_META = {{
  0: {{ title: "Mode 0: Exact Structural CSA Multiplier", tag: "m0", subtitle: "Baseline Exact CSA (366 gates)" }},
  1: {{ title: "Mode 1: Low Approximation Multiplier", tag: "m1", subtitle: "Columns 0..2 OR-approximated (353 gates)" }},
  2: {{ title: "Mode 2: Medium Approximation Multiplier", tag: "m2", subtitle: "Columns 0..4 OR-approximated (317 gates)" }},
  3: {{ title: "Mode 3: High Approximation Multiplier", tag: "m3", subtitle: "Columns 0..6 OR-approximated (262 gates)" }}
}};

// Pre-allocate net state arrays for fast evaluation
const netBuffers = {{
  0: new Uint8Array(NETLISTS[0].maxNet + 16),
  1: new Uint8Array(NETLISTS[1].maxNet + 16),
  2: new Uint8Array(NETLISTS[2].maxNet + 16),
  3: new Uint8Array(NETLISTS[3].maxNet + 16)
}};

function evalNetlist(mode, a, b) {{
  const netlist = NETLISTS[mode];
  const buf = netBuffers[mode];
  buf.fill(0);
  buf[1] = 1; // Constant 1 net

  // Assign A
  const aBits = netlist.aBits;
  for (let i = 0; i < 8; i++) {{
    buf[aBits[i]] = (a >> i) & 1;
  }}

  // Assign B
  const bBits = netlist.bBits;
  for (let i = 0; i < 8; i++) {{
    buf[bBits[i]] = (b >> i) & 1;
  }}

  // Gates: [type, out, inA, inB]
  // 0: NOT, 1: AND, 2: NAND, 3: OR, 4: NOR, 5: XOR, 6: XNOR, 7: ANDNOT, 8: ORNOT
  const gates = netlist.gates;
  const numGates = gates.length;
  for (let g = 0; g < numGates; g++) {{
    const gate = gates[g];
    const gt = gate[0];
    const out = gate[1];
    const inA = buf[gate[2]];

    if (gt === 0) {{
      buf[out] = inA ^ 1;
    }} else {{
      const inB = buf[gate[3]];
      switch (gt) {{
        case 1: buf[out] = inA & inB; break;
        case 2: buf[out] = (inA & inB) ^ 1; break;
        case 3: buf[out] = inA | inB; break;
        case 4: buf[out] = (inA | inB) ^ 1; break;
        case 5: buf[out] = inA ^ inB; break;
        case 6: buf[out] = (inA ^ inB) ^ 1; break;
        case 7: buf[out] = inA & (inB ^ 1); break;
        case 8: buf[out] = inA | (inB ^ 1); break;
      }}
    }}
  }}

  // Collect P
  const pBits = netlist.pBits;
  let p = 0;
  for (let i = 0; i < 16; i++) {{
    p |= (buf[pBits[i]] << i);
  }}
  return p;
}}

// DOM Elements
const numA = document.getElementById('numA');
const numB = document.getElementById('numB');
const sliderA = document.getElementById('sliderA');
const sliderB = document.getElementById('sliderB');
const repADec = document.getElementById('repADec');
const repAHex = document.getElementById('repAHex');
const repABin = document.getElementById('repABin');
const repBDec = document.getElementById('repBDec');
const repBHex = document.getElementById('repBHex');
const repBBin = document.getElementById('repBBin');
const btnRun255 = document.getElementById('btnRun255');
const btnZero = document.getElementById('btnZero');
const btnRandom = document.getElementById('btnRandom');
const btnSelfTest = document.getElementById('btnSelfTest');
const testStatus = document.getElementById('testStatus');
const outputsContainer = document.getElementById('outputsContainer');

function updateUI() {{
  const a = parseInt(numA.value, 10) || 0;
  const b = parseInt(numB.value, 10) || 0;

  // Format representations
  repADec.textContent = a;
  repAHex.textContent = '0x' + a.toString(16).toUpperCase().padStart(2, '0');
  repABin.textContent = a.toString(2).padStart(8, '0');

  repBDec.textContent = b;
  repBHex.textContent = '0x' + b.toString(16).toUpperCase().padStart(2, '0');
  repBBin.textContent = b.toString(2).padStart(8, '0');

  // Evaluate all 4 modes
  const results = {{
    0: evalNetlist(0, a, b),
    1: evalNetlist(1, a, b),
    2: evalNetlist(2, a, b),
    3: evalNetlist(3, a, b)
  }};

  const exactP = results[0];

  // Render cards
  outputsContainer.innerHTML = '';
  for (let m = 0; m < 4; m++) {{
    const pVal = results[m];
    const diff = Math.abs(pVal - exactP);
    const relErr = exactP === 0 ? 0.0 : ((diff / exactP) * 100);
    const approxLimit = APPROX_BITS[m];

    const card = document.createElement('div');
    card.className = 'mode-card';
    card.id = `modeCard${{m}}`;

    let errorBadgeHtml = '';
    if (m === 0) {{
      errorBadgeHtml = `<span class="stat-badge err-exact">Exact Baseline (Error: 0)</span>`;
    }} else {{
      const errClass = diff === 0 ? 'err-exact' : 'err-diff';
      errorBadgeHtml = `
        <span class="stat-badge ${{errClass}}">Abs Error: ${{diff}}</span>
        <span class="stat-badge ${{errClass}}">Rel Error: ${{relErr.toFixed(3)}}%</span>
      `;
    }}

    // 16-bit strip (MSB 15 down to LSB 0)
    let bitCellsHtml = '';
    for (let bit = 15; bit >= 0; bit--) {{
      const bitVal = (pVal >> bit) & 1;
      const exactBitVal = (exactP >> bit) & 1;
      const isMismatch = (m !== 0) && (bitVal !== exactBitVal);
      const isApprox = bit < approxLimit;

      let cellClass = `bit-cell val-${{bitVal}}`;
      if (isApprox) cellClass += ' or-approx';
      if (isMismatch) cellClass += ' bit-mismatch';

      bitCellsHtml += `
        <div class="${{cellClass}}" title="Bit ${{bit}}: ${{bitVal}}${{isApprox ? ' (OR-approximated)' : ''}}${{isMismatch ? ' [Differs from Mode 0]' : ''}}">
          <span class="bit-idx">P${{bit}}</span>
          <span class="bit-val">${{bitVal}}</span>
        </div>
      `;
    }}

    card.innerHTML = `
      <div class="mode-header">
        <div class="mode-title-group">
          <span class="mode-tag ${{MODE_META[m].tag}}">Mode ${{m}}</span>
          <div>
            <div class="mode-name">${{MODE_META[m].title}}</div>
            <div style="font-size:12px; color:var(--text-muted);">${{MODE_META[m].subtitle}}</div>
          </div>
        </div>
        <div class="mode-stats">
          <div class="mode-dec" id="mode${{m}}Dec">${{pVal}} <span style="font-size:12px; color:var(--accent-cyan); font-weight:normal;">(0x${{pVal.toString(16).toUpperCase().padStart(4, '0')}})</span></div>
          ${{errorBadgeHtml}}
        </div>
      </div>
      <div class="bit-strip">
        ${{bitCellsHtml}}
      </div>
    `;
    outputsContainer.appendChild(card);
  }}
}}

// Sync handlers
function setA(val) {{
  const clamped = Math.max(0, Math.min(255, parseInt(val, 10) || 0));
  numA.value = clamped;
  sliderA.value = clamped;
  updateUI();
}}

function setB(val) {{
  const clamped = Math.max(0, Math.min(255, parseInt(val, 10) || 0));
  numB.value = clamped;
  sliderB.value = clamped;
  updateUI();
}}

numA.addEventListener('input', e => setA(e.target.value));
sliderA.addEventListener('input', e => setA(e.target.value));
numB.addEventListener('input', e => setB(e.target.value));
sliderB.addEventListener('input', e => setB(e.target.value));

btnRun255.addEventListener('click', () => {{
  setA(255);
  setB(255);
}});

btnZero.addEventListener('click', () => {{
  setA(0);
  setB(0);
}});

btnRandom.addEventListener('click', () => {{
  setA(Math.floor(Math.random() * 256));
  setB(Math.floor(Math.random() * 256));
}});

// 65,536 exhaustive self-test against embedded checksums
btnSelfTest.addEventListener('click', () => {{
  testStatus.className = 'test-status idle';
  testStatus.textContent = 'Running exhaustive 65,536 test...';

  setTimeout(() => {{
    const t0 = performance.now();
    let allPassed = true;
    const stats = {{}};

    for (let m = 0; m < 4; m++) {{
      let sum = 0;
      let xorVal = 0;
      for (let a = 0; a < 256; a++) {{
        for (let b = 0; b < 256; b++) {{
          const p = evalNetlist(m, a, b);
          sum += p;
          xorVal ^= p;
        }}
      }}
      const expected = CHECKSUMS[m];
      const match = (sum === expected.sum) && (xorVal === expected.xor);
      stats[m] = {{ sum, xorVal, match }};
      if (!match) allPassed = false;
    }}
    const elapsed = (performance.now() - t0).toFixed(1);

    if (allPassed) {{
      testStatus.className = 'test-status pass';
      testStatus.textContent = `PASS: All 65,536 pairs & 4 modes (262,144 evals) match LUT checksums in ${{elapsed}} ms`;
    }} else {{
      testStatus.className = 'test-status fail';
      testStatus.textContent = `FAIL: Checksum mismatch. Details in console.`;
      console.error('Self-test mismatches:', stats);
    }}
  }}, 20);
}});

// Initial invocation
updateUI();

if (window.location.search.includes('selftest')) {{
  btnSelfTest.click();
}}
</script>
</body>
</html>
"""

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated self-contained simulator at: {out_html_path}")
    print(f"File size: {os.path.getsize(out_html_path) / 1024:.2f} KB")

if __name__ == "__main__":
    generate_html()
