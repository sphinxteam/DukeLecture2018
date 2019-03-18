def iterate_amp(J, var_noise, x0=None, max_iter=100, tol=1e-7, verbose=1):
    """Iterate AMP to solve J = xx^T, w/ x Rademacher"""
    
    # Some pre-processing
    size_x = J.shape[0]
    
    # Initialize variables
    B = np.zeros(size_x)
    m = np.random.rand(size_x)
    m_old = np.zeros(size_x)
    
    for t in range(max_iter):
        # Perform iteration
        B = (J.dot(m) / np.sqrt(size_x) - m_old * np.mean(1 - m ** 2)) / var_noise
        m_old = np.copy(m)
        m = np.tanh(B)
         
        # Compute metrics
        diff = np.mean(np.abs(m - m_old))
        mse = np.mean((m - x0) ** 2) if x0 is not None else 0
        
        # Print iteration status on screen
        if verbose:
            print("t = %d, diff = %g; mse = %g" % (t, diff, mse))
        
        # Check for convergence
        if diff < tol:
            break
            
    return m