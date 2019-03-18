import os, sys
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def inline_print(text):
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except OSError as e:
        print(e)

class PlantedGraphColoring:
    def __init__(self, N, c, q):
        try:
            iter(c)
            self.c_list = sorted([round(cc, 4) for cc in c])
        except:
            self.c_list = [round(c, 4)]
        self.N, self.c, self.q = N, self.c_list[-1], q
        self.generate_planted_config()
        self.generate_graph()
        self.generate_graph_snapshot()
        
    def generate_planted_config(self):
        # Generate planted configuration
        self.s_star = np.random.randint(low=0, high=self.q, size=self.N)
        
    def generate_graph(self):
        N, c, q, s_star = self.N, self.c, self.q, self.s_star
        # Generate a draft of upper half of adjacency matrix
        valid = np.triu(s_star[None,:] != s_star[:,None])
        mask = (np.random.random(size=(N,N)) < c/(N-1) * q/(q-1))
        A_half = (valid * mask).astype(int)
        self.A = A_half + A_half.T
        # Compute how many edges need to be added/removed to get exactly c*N edges
        delta = (int(c*N+0.5) - self.A.sum()) // 2
        if delta > 0: self.add_edges(delta)
        if delta < 0: self.del_edges(-delta)
        # Generate adjacency list from adjacency matrix
        self.A_link = [np.where(row)[0].tolist() for row in self.A]
        
    def add_edges(self, edge_num):
        # Since c << N, when adding edges, random pick a (row,col) pair 
        # we have high probability to get a zero, then we can set it to be one
        index = []
        while edge_num > 0:
            row, col = np.random.randint(low=0, high=self.N, size=2)
            if row == col or self.A[row,col] > 0: continue
            self.A[row,col] = self.A[col,row] = 1 
            edge_num -= 1
            index.append((row, col))
        return index
            
    def del_edges(self, edge_num):
        N, c, q = self.N, self.c, self.q
        # Since c << N, when removing edges, it is more efficient to list all
        # existing edges and then delete from the edge set
        row, col = np.where(np.triu(self.A) == 1)
        di = np.random.choice(list(range(row.size)), size=edge_num, replace=False)
        self.A[row[di], col[di]] = self.A[col[di], row[di]] = 0
        index = [(row, col) for row, col in zip(row[di], col[di])]
        return index
        
    def dilute(self, delta_c = 0.1):
        edge_num = int(delta_c * self.N // 2 + 0.5)
        self.del_edges(edge_num)
        # Generate adjacency list from adjacency matrix
        self.A_link = [np.where(row)[0].tolist() for row in self.A]
        
    def thicken(self, delta_c = 0.1):
        edge_num = int(delta_c * self.N // 2 + 0.5)
        self.add_edges(edge_num)
        # Generate adjacency list from adjacency matrix
        self.A_link = [np.where(row)[0].tolist() for row in self.A]
        
    def generate_graph_snapshot(self):
        row, col = np.where(np.triu(self.A) == 1)
        perm = list(range(row.size))
        np.random.shuffle(perm)
        
        idx, self.snapshots = 0, {}
        delta_c = [self.c_list[0]] + np.diff(self.c_list).tolist()
        for i in range(len(delta_c)):
            c, edge_num = self.c_list[i], int(delta_c[i] * N / 2 + 0.5)
            self.snapshots[c] = (row[perm[idx:idx+edge_num]], col[perm[idx:idx+edge_num]])
            idx += edge_num
            
    def recover_snapshot(self, c):
        c = round(c, 4)
        if not hasattr(self, 'snapshots'): 
            print('No snapshot')
            return
        if c not in self.snapshots: 
            print('No snapshot at c = {0:.4f}'.format(c))
            return
        self.c, self.A = c, np.zeros((self.N, self.N), dtype=int)
        for cc, (row, col) in self.snapshots.items():
            if cc > c: continue
            self.A[row, col] = self.A[col, row] = 1
        # Generate adjacency list from adjacency matrix
        self.A_link = [np.where(row)[0].tolist() for row in self.A]
            
    def save(self, file_name):
        if hasattr(self, 'A'): del self.A
        np.savez(file_name, G=self)
        

def BP_init(G, mode='perturb'):
    q, A_link, s_star, ep = G.q, G.A_link, G.s_star, 0.001
    normalize= lambda x: x / x.sum()
    init_fun = {
        'random': lambda j,i: normalize(np.random.random(q)),
        'uniform': lambda j,i: np.ones(q) / q,
        'perturb': lambda j,i: normalize((1-ep + 2*ep*np.random.random(q)) / q),
        'planted': lambda j,i: np.eye(1, q, s_star[j]).reshape(-1)
    }
    chi = {(j, i): init_fun[mode](j,i) for j, L in enumerate(A_link) for i in L}
    return chi

def BP_update(G, chi, schedule='random', damping=0.8):
    chi_new, diff, eps = chi.copy(), 0, 1e-32
    normalize= lambda x: x / x.sum()
    if schedule == 'parallel':
        for j in range(G.N):
            full_prod = np.ones(G.q)
            for k in G.A_link[j]:
                full_prod *= 1 - chi[(k, j)] + eps
            for i in G.A_link[j]:
                chi_new[(j, i)] = damping * normalize(full_prod / (1 - chi[(i, j)] + eps)) + (1-damping) * chi_new[(j, i)]
                diff += abs(chi_new[(j, i)] - chi[(j, i)]).sum()
    elif schedule == 'random':
        perm = list(range(G.N))
        np.random.shuffle(perm)
        for j in perm:
            full_prod = np.ones(G.q)
            for k in G.A_link[j]:
                full_prod *= 1 - chi_new[(k, j)] + eps
            for i in G.A_link[j]:
                chi_new[(j, i)] = damping * normalize(full_prod / (1 - chi_new[(i, j)] + eps)) + (1-damping) * chi_new[(j, i)]
                diff += abs(chi_new[(j, i)] - chi[(j, i)]).sum()
    return chi_new, diff / (2*G.q*G.N)

def compute_f_Bethe(G, chi):
    Z_i, Z_ij, eps = 0, 0, 1e-32
    for i in range(G.N):
        full_prod = np.ones(G.q)
        for k in G.A_link[i]:
            full_prod *= 1 - chi[(k, i)] + eps
        Z_i += np.log(full_prod.sum())
    for i in range(G.N):
        for j in G.A_link[i]:
            Z_ij += np.log( 1 - (chi[(i,j)] * chi[(j,i)]).sum() + eps ) / 2
    return (Z_i - Z_ij) / G.N

def compute_Q(G, chi):
    chi_margin, eps = {}, 1e-32
    normalize = lambda x: x / x.sum()
    for i in range(G.N):
        full_prod = np.ones(G.q)
        for k in G.A_link[i]:
            full_prod *= 1 - chi[(k, i)] + eps
        chi_margin[i] = normalize(full_prod)
    Q = 0
    for perm in it.permutations(range(G.q)):
        temp = sum([chi_j[perm[G.s_star[j]]] for j, chi_j in chi_margin.items()])
        Q = max(Q, (temp/N - 1/G.q)/(1-1/G.q) )
    return Q
        
def BP_coloring(G, init='perturb', schedule='random', damping=0.8,
                max_iter=300, abs_tol=1e-3, verbose=False):
    chi_msg, t = BP_init(G, mode=init) if type(init) == str else init, 0
    while t <= max_iter:
        chi_msg, diff = BP_update(G, chi_msg, schedule=schedule, damping=damping)
        if verbose: 
            inline_print('\rIteration {0}: diff={1:.4e}'.format(t+1, diff))
        if diff < abs_tol or np.isnan(diff): break
        t += 1
    f_Bethe, Q = compute_f_Bethe(G, chi_msg), compute_Q(G, chi_msg)
    if verbose: 
        inline_print(', f_Bethe = {0:.4f}, Q = {1:.4f}\n'.format(f_Bethe, Q))
    return f_Bethe, Q, chi_msg, t

if __name__ == "__main__":
	N, q = int(sys.argv[1]), int(sys.argv[2])
	T_max, damping = 1000, 0.95
	if len(sys.argv) >= 4: T_max = int(sys.argv[3])
	if len(sys.argv) >= 5: damping = float(sys.argv[4])

	if q == 3: low, high = 3.5, 4.5
	if q == 4: low, high = 8, 9.7
	if q == 5: low, high = 12.5, 16.5
	c = np.linspace(low, high, round(100*(high-low))+1)	


	simul_file_name = 'planted_graph_coloring_result_q={0}_N={1}.npz'.format(q, N)
	try:
	    res = np.load(simul_file_name)
	    f_perturb, f_planted = res['f_perturb'], res['f_planted']
	    Q_perturb, Q_planted = res['Q_perturb'], res['Q_planted']
	    print('Find previous result')
	except:
	    f_perturb, f_planted = np.nan * np.ones_like(c), np.nan * np.ones_like(c)
	    Q_perturb, Q_planted = np.nan * np.ones_like(c), np.nan * np.ones_like(c)
	    print('New simulation')

	G = PlantedGraphColoring(N, c, q)
	for i, cc in enumerate(G.c_list):
	    if np.isnan(f_perturb[i]) or np.isnan(f_planted[i]):
	        print('Current c:', cc)
	        G.recover_snapshot(cc)
	        if np.isnan(f_perturb[i]):
	            while True:
	                f_perturb[i], Q_perturb[i], chi_msg, t = BP_coloring(G, init='random', max_iter=T_max, damping=damping, verbose=True)
	                if t < T_max: break
	                G = PlantedGraphColoring(N, c, q)
	            np.savez(simul_file_name, f_perturb=f_perturb, f_planted=f_planted, Q_perturb=Q_perturb, Q_planted=Q_planted, c=c)
	        if np.isnan(f_planted[i]):
	            while True:
	                f_planted[i], Q_planted[i], chi_msg, t = BP_coloring(G, init='planted', max_iter=T_max, damping=damping, verbose=True)
	                if t < T_max: break
	                G = PlantedGraphColoring(N, c, q)
	            np.savez(simul_file_name, f_perturb=f_perturb, f_planted=f_planted, Q_perturb=Q_perturb, Q_planted=Q_planted, c=c)
