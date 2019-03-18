from scipy.special import erfcx

def prior(A, B):
    """Compute f and f' for Rademacher prior"""
    
    a = np.tanh(B)
    c = 1 - a ** 2
    return a, c

def channel(y, w, v, var_noise):
    """Compute g and g' for probit channel"""
    
    phi = -y * w / np.sqrt(2 * (v + var_noise))
    g = 2 * y / (np.sqrt(2 * np.pi * (v + var_noise)) * erfcx(phi))
    dg = -g * (w / (v + var_noise) + g)
    
    return g, dg