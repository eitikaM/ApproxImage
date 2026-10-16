def compute_column_or(pp, c):
    val = 0
    for i in range(8):
        for j in range(8):
            if i + j == c:
                val |= pp[i][j]
    return val

def test_scheme_A(K):
    # Scheme A: 
    # For columns c < K: P[c] = OR of partial products with i+j == c
    # For columns c >= K: exact sum of partial products with i+j == c (no carry from c < K)
    # When K == 0: exact multiplier
    
    total_cases = 256 * 256
    exact_matches = 0
    max_err = 0
    total_err = 0
    
    for a in range(256):
        for b in range(256):
            exact_p = a * b
            
            # Partial products
            pp = [[((a >> j) & 1) & ((b >> i) & 1) for j in range(8)] for i in range(8)]
            
            if K == 0:
                approx_p = exact_p
            else:
                # Lower K bits from OR
                p_lower = 0
                for c in range(K):
                    p_lower |= (compute_column_or(pp, c) << c)
                    
                # Upper bits: sum of pp with i+j >= K
                p_upper = 0
                for i in range(8):
                    for j in range(8):
                        if i + j >= K:
                            p_upper += (pp[i][j] << (i + j))
                
                approx_p = p_upper | p_lower
                
            err = abs(exact_p - approx_p)
            if err == 0:
                exact_matches += 1
            else:
                total_err += err
                if err > max_err:
                    max_err = err
                    
    mae = total_err / total_cases
    err_rate = (total_cases - exact_matches) / total_cases * 100.0
    print(f"K={K}: Exact Matches={exact_matches}/{total_cases} ({exact_matches/total_cases*100:.2f}%), "
          f"Err Rate={err_rate:.2f}%, Max Err={max_err}, MAE={mae:.4f}")

for k in [0, 2, 3, 4, 5, 6, 7]:
    test_scheme_A(k)
