import sys
import numpy as np
from numpy import exp, log, sqrt, tanh, cosh, cos
from numpy.linalg import norm
from scipy.linalg import eigh
from collections import defaultdict

def single_test(N, lam, rho):
    alphabet = np.array( [sqrt((1-rho)/rho),-sqrt(rho/(1-rho))] )
    x_star = np.random.choice(alphabet, size=(N,1), replace=True, p=[rho, 1-rho])

    W = np.triu(np.random.randn(N, N), k=1)
    W = W + W.T + np.diag(np.random.randn(N))
    Y = sqrt(lam/N) * x_star @ x_star.T + W
    # Find the largest and smallest eigenvalue and their eigenvector
    eig_val1, eig_vec1 = eigh(Y, eigvals=(0,0))
    eig_val2, eig_vec2 = eigh(Y, eigvals=(N-1,N-1))
    v = eig_vec1 if abs(eig_val1) > abs(eig_val2) else eig_vec2
    if rho == 0.5:
        x_hat = np.sign(v)
        mse = min( norm(x_star-x_hat)**2, norm(x_star+x_hat)**2 ) / N
    else:
        v = eig_vec1 if abs(eig_val1) > abs(eig_val2) else eig_vec2
        x_hat1, x_hat2 = alphabet[1] * np.ones_like(v), alphabet[1] * np.ones_like(v)
        x_hat1[np.argsort(v.reshape(-1))[:int(np.round(N*rho))]] = alphabet[0]
        x_hat2[np.argsort((-v).reshape(-1))[:int(np.round(N*rho))]] = alphabet[0]
        mse = min( norm(x_star-x_hat1)**2, norm(x_star-x_hat2)**2 ) / N
    return mse
    

if __name__ == "__main__":
    rho = round(float(sys.argv[1]), 4) if len(sys.argv) >= 2 else 0.5
    trial = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
    N_list = [10, 20, 50, 100, 200, 500, 1000]
    lam_list = np.linspace(0, 10, 101)

    result = defaultdict(lambda: np.nan * np.ones((len(lam_list), trial)))
    try:
        data = np.load('spectral_rho={0}.npz'.format(rho))
        for k, v in data.items():
            result[k] = v
    except:
        pass

    for N in N_list:
        res = result['N={0}'.format(N)]
        for i, lam in enumerate(lam_list):
            if not any(np.isnan(res[i])): continue
            for j in range(trial):
                if not np.isnan(res[i][j]): continue
                res[i][j] = single_test(N,lam,rho)
            print(N, lam, res[i].mean())
            np.savez('spectral_rho={0}.npz'.format(rho), **result)