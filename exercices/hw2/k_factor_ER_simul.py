import numpy as np
import networkx as nx
from itertools import combinations
from functools import reduce
from collections import defaultdict
def BP_k_factor(G, beta, k, damp=1, T=3000, rel_tol=1e-4, report=False):
    def cal_prod(i, I):
        # \prod_{l \in I} h^{(il) \to i}
        return reduce(lambda x,y: x*y, [h[(l,i)] for l in I], 1)
    def cal_sum_kappa(j, i, kappa):
        # \sum_{I \subseteq \partial i \ j ,|I|=k} \prod_{l \in I} h^{(il) \to i}
        return sum(cal_prod(i, I) for I in combinations(set(adj_list[i])-set([j]), kappa))
    def BP_update(h):
        h_new = {}
        for j in G.nodes():
            for i in adj_list[j]:
                sum_less_k, sum_eq_k = sum(cal_sum_kappa(j, i, kappa) for kappa in range(k)), cal_sum_kappa(j, i, k)
                h_new[(j, i)] = np.exp(beta) * sum_less_k / (sum_less_k + sum_eq_k)
                h_new[(j, i)] = (1-damp) * h[(j, i)] + damp * h_new[(j, i)]
        return h_new
    def compute_f_Bethe(h):
        f = beta * len(G.edges())
        for i in G.nodes():
            f += np.log( sum(cal_sum_kappa(-1, i, kappa) for kappa in range(k+1)) )
        for i, j in G.edges():
            f -= np.log( np.exp(beta) + h[(i,j)] * h[(j,i)] )
        return f / len(G.nodes())
        
    # Initialize BP messages randomly
    adj_list, h, f_Bethe = [list(v.keys()) for k, v in sorted(G.adjacency())], {}, []
    for i, l in enumerate(adj_list):
        for j in l: h[(i,j)] = 1/2
    for t in range(1,T+1):   
        h = BP_update(h)
        f_Bethe.append(compute_f_Bethe(h))
        if report and t % report == 0:
            print('iteration {0}, f_Bethe = {1:.6f}'.format(t, f_Bethe[-1]))
        if t > 10 and abs((f_Bethe[-1] - f_Bethe[-10])/f_Bethe[-1]) < rel_tol:
            print('beta {0:.3f}: BP converge in {1} iteration, f_Bethe = {2:.6f}'.format(beta, t, f_Bethe[-1]))
            break
    return f_Bethe, h

N = 1000
result, beta = defaultdict(list), np.linspace(-3,3,101)
for c in [3,4,5]:
    G = nx.erdos_renyi_graph(N, c/(N-1))
    for k in [1,2,3]:
        print('#'*80)
        print('k = {0}, c = {1}'.format(k, c))
        print('#'*80)
        for b in beta:
            f = BP_k_factor(G, beta=b, k=k, report=100)[0]
            result['f_Bethe_k={0}_c={1}'.format(k, c)].append(f[-1])
        np.savez('k_factor_ER_simul.npz', **result)