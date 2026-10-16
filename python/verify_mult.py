def fa(a, b, cin):
    s = a ^ b ^ cin
    cout = (a & b) | (b & cin) | (a & cin)
    return s, cout

def simulate_csa_multiplier(A, B):
    # 1. Partial products pp[i][j] = A[j] & B[i]
    pp = [[((A >> j) & 1) & ((B >> i) & 1) for j in range(8)] for i in range(8)]
    
    P = [0] * 16
    P[0] = pp[0][0]
    
    # S and C vectors
    S = [pp[0][j] for j in range(8)]
    C = [0] * 8
    
    for stage in range(1, 8):
        next_S = [0] * 8
        next_C = [0] * 8
        # We add: {0, S[7:1]}, pp[stage][7:0], C[7:0]
        # bit 0: S[1], pp[stage][0], C[0]
        # ...
        # bit 6: S[7], pp[stage][6], C[6]
        # bit 7: 0,    pp[stage][7], C[7]
        for col in range(8):
            in_s = S[col + 1] if col < 7 else 0
            in_pp = pp[stage][col]
            in_c = C[col]
            s_out, c_out = fa(in_s, in_pp, in_c)
            next_S[col] = s_out
            next_C[col] = c_out
        
        P[stage] = next_S[0]
        S = next_S
        C = next_C
        
    # Final Carry-Propagate Adder for bits 8 to 15
    # Adds {0, S[7:1]} and C[7:0]
    carry = 0
    for col in range(8):
        in_s = S[col + 1] if col < 7 else 0
        in_c = C[col]
        s_out, carry = fa(in_s, in_c, carry)
        P[8 + col] = s_out
        
    res = 0
    for i in range(16):
        res |= (P[i] << i)
    return res

# Test all 65536 combinations
errors = 0
for a in range(256):
    for b in range(256):
        expected = a * b
        actual = simulate_csa_multiplier(a, b)
        if expected != actual:
            errors += 1
            print(f"Error for {a} * {b}: expected {expected}, got {actual}")
            if errors > 5:
                break
    if errors > 5:
        break

if errors == 0:
    print("ALL 65,536 INPUT COMBINATIONS VERIFIED SUCCESSFULLY!")
else:
    print(f"Found {errors} errors.")
