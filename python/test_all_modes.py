def fa(a, b, cin):
    s = a ^ b ^ cin
    cout = (a & b) | (b & cin) | (a & cin)
    return s, cout

def simulate_approx_multiplier(A, B, mode):
    # Determine K based on mode
    # mode 0: K = 0 (Exact)
    # mode 1: K = 3 (Low)
    # mode 2: K = 5 (Med)
    # mode 3: K = 7 (High)
    if mode == 0:
        K = 0
    elif mode == 1:
        K = 3
    elif mode == 2:
        K = 5
    elif mode == 3:
        K = 7
    else:
        K = 0

    # 1. Raw partial products pp[i][j] = A[j] & B[i]
    pp = [[((A >> j) & 1) & ((B >> i) & 1) for j in range(8)] for i in range(8)]
    
    # 2. Lower-part OR bits for columns 0..6
    col_or = [0] * 7
    for c in range(7):
        bit_or = 0
        for i in range(8):
            for j in range(8):
                if i + j == c:
                    bit_or |= pp[i][j]
        col_or[c] = bit_or

    # 3. CSA inputs: if (i+j < K), zero out for CSA
    pp_csa = [[0 for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(8):
            if (i + j) < K:
                pp_csa[i][j] = 0
            else:
                pp_csa[i][j] = pp[i][j]

    # 4. Carry-Save Addition Tree
    P_csa = [0] * 16
    P_csa[0] = pp_csa[0][0]
    
    S = [pp_csa[0][j] for j in range(8)]
    C = [0] * 8
    
    for stage in range(1, 8):
        next_S = [0] * 8
        next_C = [0] * 8
        for col in range(8):
            in_s = S[col + 1] if col < 7 else 0
            in_pp = pp_csa[stage][col]
            in_c = C[col]
            s_out, c_out = fa(in_s, in_pp, in_c)
            next_S[col] = s_out
            next_C[col] = c_out
        
        P_csa[stage] = next_S[0]
        S = next_S
        C = next_C
        
    # Final Carry-Propagate Adder for bits 8 to 15
    carry = 0
    for col in range(8):
        in_s = S[col + 1] if col < 7 else 0
        in_c = C[col]
        s_out, carry = fa(in_s, in_c, carry)
        P_csa[8 + col] = s_out

    # 5. Output multiplexing
    P = [0] * 16
    for bit in range(16):
        if bit < K:
            P[bit] = col_or[bit]
        else:
            P[bit] = P_csa[bit]
            
    res = 0
    for bit in range(16):
        res |= (P[bit] << bit)
    return res

print("Evaluating all 65,536 combinations for each mode...")
for mode in range(4):
    total = 256 * 256
    exact_matches = 0
    max_err = 0
    total_err = 0
    
    for a in range(256):
        for b in range(256):
            exact_val = a * b
            approx_val = simulate_approx_multiplier(a, b, mode)
            err = abs(exact_val - approx_val)
            if err == 0:
                exact_matches += 1
            else:
                total_err += err
                if err > max_err:
                    max_err = err
                    
    err_cases = total - exact_matches
    err_rate = (err_cases / total) * 100.0
    mae = total_err / total
    print(f"Mode {mode}: Total={total}, Matches={exact_matches}, Errors={err_cases}, "
          f"ErrRate={err_rate:.2f}%, MaxErr={max_err}, MAE={mae:.4f}")
