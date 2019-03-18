from scipy.integrate import quad

def iterate_se(frac_nonzeros, rows_to_columns, var_noise, max_iter=100, tol=1e-7, verbose=1):
    """Iterates state evolution associated to AMP implementation above"""
    
    # Define function to be integrated at each step
    f = lambda A: lambda z: np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi) * \
            ((1 - frac_nonzeros) * prior(A, np.sqrt(A) * z, frac_nonzeros)[1] + \
             frac_nonzeros * prior(A, np.sqrt(A * (1 + A)) * z, frac_nonzeros)[1])
    
    v = np.zeros(max_iter)
    v[0] = frac_nonzeros
    
    for t in range(max_iter - 1):
        A = rows_to_columns / (var_noise + v[t])
        v[t + 1] = quad(f(A), -10, 10)[0]
        
        diff = np.abs(v[t + 1] - v[t])
        if verbose:
            print("t = %d, diff = %g; v = %g" % (t, diff, v[t + 1]))
            
        if diff < tol:
            break
    
    return v[:t + 1]