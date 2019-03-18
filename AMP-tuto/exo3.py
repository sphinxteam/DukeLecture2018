def sample_instance(size_x, frac_nonzeros, rows_to_columns, var_noise):
    """Samples F from P(F) and {x, y} from P(x, y | F)"""
    
    # Some pre-processing
    size_nonzeros = int(np.ceil(frac_nonzeros * size_x))
    size_y = int(np.ceil(rows_to_columns * size_x))
    
    # Sample x from P_0(x)
    x0 = np.zeros(size_x)
    nonzeros = np.random.choice(size_x, size_nonzeros, replace=False)
    x0[nonzeros] = np.random.randn(size_nonzeros)
    
    # Generate F and y = Fx + noise
    F = np.random.randn(size_y, size_x) / np.sqrt(size_x)
    noise = np.sqrt(var_noise) * np.random.randn(size_y)
    
    y = F.dot(x0) + noise

    return x0, F, y