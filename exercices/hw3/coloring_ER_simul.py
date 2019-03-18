import numpy as np
from numpy import exp, log, sqrt, tanh
import networkx as nx
from collections import defaultdict
import multiprocessing as mp

def BP_coloring(G, beta, q, init='perturb', schedule='random', damp=1, T=1000, abs_tol=1e-4, report=False):
    theta = 1 - exp(-beta)
    def BP_update(chi, schedule):
        chi_new, diff = chi.copy(), 0
        if schedule == 'parallel':
            for j in G.nodes():
                full_prod = np.ones(q)
                for k in adj_list[j]:
                    full_prod *= 1 - theta * chi[(k, j)]
                for i in adj_list[j]:
                    chi_new[(j, i)] = full_prod / ( 1 - theta * chi[(i, j)] )
                    chi_new[(j, i)] /= chi_new[(j, i)].sum()
                    chi_new[(j, i)] = (1-damp) * chi[(j, i)] + damp * chi_new[(j, i)]
                    diff += abs(chi_new[(j, i)] - chi[(j, i)]).sum()
        elif schedule == 'random':
            perm = list(G.nodes())
            np.random.shuffle(perm)
            for j in perm:
                full_prod = np.ones(q)
                for k in adj_list[j]:
                    full_prod *= 1 - theta * chi_new[(k, j)]
                for i in adj_list[j]:
                    chi_new[(j, i)] = full_prod / ( 1 - theta * chi_new[(i, j)] )
                    chi_new[(j, i)] /= chi_new[(j, i)].sum()
                    chi_new[(j, i)] = (1-damp) * chi[(j, i)] + damp * chi_new[(j, i)]
                    diff += abs(chi_new[(j, i)] - chi[(j, i)]).sum()
        return chi_new, diff / (2*q*len(G.edges()))
    def compute_f_Bethe(chi):
        f = 0
        for i in G.nodes():
            full_prod = np.ones(q)
            for k in adj_list[i]:
                full_prod *= 1 - theta * chi[(k, i)]
            f += log(full_prod.sum())
        for i, j in G.edges():
            f -= log( 1 - theta * (chi[(i,j)] * chi[(j,i)]).sum() )
        return f / len(G.nodes())
        
    # Initialize BP messages randomly
    adj_list, chi, f_Bethe = [list(v.keys()) for k, v in sorted(G.adjacency())], {}, np.array([])
    for i, l in enumerate(adj_list):
        for j in l:
            if init == 'perturb':
                epsilon=0.1
                temp = 1 / q + np.random.uniform(low=-epsilon, high=epsilon, size=q) / q
                chi[(i,j)] = temp / temp.sum()
            elif init == 'first-color':
                chi[(i,j)] = np.array([1] + [0]*(q-1))
            elif init == 'random':
                temp = np.random.uniform(low=0, high=1, size=q)
                chi[(i,j)] = temp / temp.sum()
    for t in range(1,T+1):   
        chi, diff = BP_update(chi, schedule)
        f_Bethe = np.append(f_Bethe, compute_f_Bethe(chi))
        if report and t % report == 0:
            print('iteration {0}, f_Bethe = {1}'.format(t, f_Bethe[-4:]))
        if diff < abs_tol:
            if report:
                print('BP converge in {0} iteration'.format(t))
            break
    return f_Bethe, chi, t

if __name__ == "__main__":
    result = defaultdict(list)
    try:
        data = np.load('coloring_ER_simul.npz')
        for k, v in data.items():
            result[k] = v
    except:
        pass
    beta, q, trial, max_iter = 2, 3, 8, int(1e4)
    c = np.linspace(0.1, 10, 100)

    def single_case(x):
        i, N, cc = x
        G = nx.erdos_renyi_graph(N, cc/(N-1))
        f_Bethe, BP_msg, tt = BP_coloring(G, beta=beta, q=q, T=max_iter, report=False)
        print('trial {0} finished in {1} iterations'.format(i, tt))
        return tt
    pool = mp.Pool(trial)

    for N in [50, 100, 200, 500, 1000, 2000]:
        t = np.nan * np.ones((c.size, trial)) if result.get('N={}'.format(N)) is None else result['N={}'.format(N)]
        for i, cc in enumerate(c):
            if np.any(np.isnan(t[i])):
                print('\n'.join(['#'*80, 'c = {0}, N = {1}'.format(cc, N), '#'*80]))
                t[i] = pool.map(single_case, [(j, N, cc) for j in range(1, trial+1)])
                result['N={0}'.format(N)] = t
                print(i, cc, t[i].tolist())
                np.savez('coloring_ER_simul.npz', **result)
            if np.median(t[i]) == max_iter:
                break