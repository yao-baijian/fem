import torch, re
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import warnings
warnings.filterwarnings('ignore')

def parse_file(problem_type, filename, index_start=0, graph_type = 'hyper'):
    if problem_type in ['maxcut', 'bmincut', 'modularity', 'vertexcover']:
        if graph_type == 'normal':
            n, m, couplings = read_graph(filename, index_start)
        elif graph_type == 'hyper':
            n, m, couplings, neighbors = read_hypergraph(filename, index_start)
    elif problem_type == 'maxksat':
        n, m, couplings = read_cnf(filename)
    return n, m, couplings


def load_matrix(path:'str',numer_package:'str',store_format:'str') -> 'float':
    """"
    Load the coupling matrix of the Graph instance from '.txt' file to python matrix.
    
    Parameters:

    :param path - The file path of the graph instance, with format of '.txt';
    :param numer_package - Choose the preferred python sci-package, choices = 'scipy' and 'torch';
    :param store_format - The output matrix will be with the store format of 'store_format', choices = 'csr', 'csc' and 'dense'.
    
 
    Returns:
    
    The output coupling matrix of the instance graph.
    
    """ 
    with open(path, "r") as f:
        l = f.readline()
        N, edges = [int(x) for x in l.split(" ") if x != "\n"]
        
    G = pd.read_csv(path,sep=' ',skiprows=[0],index_col=False, header=None,names=['node1','node2','weight'])
    G.fillna({'weight':int(1)},inplace=True)
    shift = G.iloc[0,0]
    ori_graph = np.array([list(np.concatenate([G.iloc[:,0]-shift,G.iloc[:,1]- shift])),
                 list(np.concatenate([G.iloc[:,1]-shift,G.iloc[:,0]- shift])),
                 list(np.concatenate([G.iloc[:,-1],G.iloc[:,-1]]))])
    ori_graph = ori_graph.T[np.lexsort((ori_graph[1,:],ori_graph[0,:])).tolist()].T
    if numer_package == 'scipy':
        J = coo_matrix((ori_graph[2,:].tolist(),
                             (ori_graph[0,:].tolist(),ori_graph[1,:].tolist())), shape=(N, N))
        if J.shape[0] != N:
            print("The shape of J does not match N!")
        if J.data.shape[0]/2 != edges:
            print("The number of elements in J does not match edges!")
        if store_format == 'csr':
            J = csr_matrix(J)
        elif store_format == 'csc':
            J = csc_matrix(J)
        elif store_format == 'dense':
            J = J.todense()
        else:
            print("Error: Input wrong 'store_format'! Please choose from ['csr', 'csc', 'dense'].")
    elif numer_package == 'torch':
        J = torch.sparse_coo_tensor([ori_graph[0,:].tolist(),
                                 ori_graph[1,:].tolist()], 
                                    ori_graph[2,:].tolist(),(N, N))
        if J.shape[0] != N:
            print("The shape of J does not match N!")
        if J._values().shape[0]/2 != edges:
            print("The number of elements in J does not match edges!")
        
        if store_format == 'csr':
            J = J.to_sparse_csr()
        elif store_format == 'csc':
            J = J.to_sparse_csc()
        elif store_format == 'dense':
            J = J.to_dense()
        else:
            print("Error: Input wrong 'store_format'! Please choose from ['csr', 'csc', 'dense'].")
    else:
        print("Error: Input wrong 'numer_package'! Please choose from ['scipy', 'torch'].")
    return J


def load_gset(instance):
    """
    load the weight matrix of Gset, modified from code of Zisong Shen
    """
    # print('loading Gset',instance,'...')
    path = './Gset/' + instance
    G = pd.read_csv(path, sep=' ')
    n_v = int(G.columns[0])
    ori_graph = np.array([list(np.concatenate([G.iloc[:,0]-1,G.iloc[:,1]-1])), list(np.concatenate([G.iloc[:,1]-1,G.iloc[:,0]-1])), list(np.concatenate([G.iloc[:,-1],G.iloc[:,-1]]))])
    ori_graph = ori_graph.T[np.lexsort((ori_graph[1,:],ori_graph[0,:])).tolist()].T
    J = torch.sparse_coo_tensor([ori_graph[0,:].tolist(), ori_graph[1,:].tolist()], ori_graph[2,:].tolist(),(n_v, n_v)).to_sparse_csr() 
    """ using sparse column here """
    with open('targetvalue.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    result = re.findall(".*"+instance+" (.*).*", content)
    target_value = int(result[0]) if result else 0
    # print('N=%d'%(J.shape[0])," c=%.2f"%(torch.count_nonzero(J.to_dense())*2/J.shape[0]),"best_cut:"+ target_value)
    return J.to_dense(), target_value


def read_graph(file, index_start=0):
    """
    function for reading graph files
    the specific format should be n m in the first line, and m following lines
    represent source end weight
    Parameters:
        file: string, the filenmae of the graph to be readed
        index_start: int, specify which is the start index of the graph
    """
    with open(file,"r") as f:
        l = f.readline()
        n, m = [int(x) for x in l.split(" ") if x!="\n"]
        J = torch.zeros([n, n])
        neighbors = [[] for i in range(n)]
        for k in range(m):
            l = f.readline()
            l_split = l.split()
            i, j = [int(x) for x in l_split[:2]]
            if len(l_split) == 2:
                w = 1.0
            elif len(l_split) == 3:
                w = float(l_split[2])
            else:
                raise ValueError("Unkown graph file format")
            i -= index_start
            j -= index_start
            J[i, j], J[j, i] = w, w
            neighbors[i].append(j)
            neighbors[j].append(i)
    return n, m, J

def read_hypergraph(file, index_start=1):  #hgr file
    with open(file, "r") as f:
        # Read first line with number of vertices and hyperedges
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        J = torch.zeros([n, n])
        neighbors = [[] for _ in range(n)]
        
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) for x in l.split() if x != "\n"]
            
            # Create pairwise connections between all vertices in the hyperedge
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    u = vertices[i] - index_start
                    v = vertices[j] - index_start
                    
                    # Add edge with weight 1 (or increment if edge exists)
                    J[u, v] += 1.0
                    J[v, u] += 1.0
                    
                    # Add to neighbors list (avoid duplicates)
                    if v not in neighbors[u]:
                        neighbors[u].append(v)
                    if u not in neighbors[v]:
                        neighbors[v].append(u)
    
    return n, m, J, neighbors


def read_cnf(path):
    with open(path,'r') as f:
        lines = f.readlines()
    k_length_sat_table = {}
    for line in lines:
        l = line.split()
        if l[0] == 'c':  # comment line
            pass
        elif l[0] == 'p': # problem line
            N, M = map(int, l[2:])
        else:
            clause = list(map(int, l[:-1]))
            k = len(clause)
            if k not in k_length_sat_table:
                k_length_sat_table[k] = []
            k_length_sat_table[k].append([])
            k_length_sat_table[k][-1].append(list(map(abs, clause)))
            q_states = []
            for i in range(k):
                if clause[i] > 0:
                    q_states.append(0)    # postive literal
                else:
                    q_states.append(1)     # negative literal
            k_length_sat_table[k][-1].append(q_states)
    sat_table = []
    minimum_index = []
    maximum_index = []
    for key in sorted(k_length_sat_table.keys()):
        k_length_sat_table[key] = np.array(k_length_sat_table[key])
        minimum_index.append(np.min(k_length_sat_table[key][:,0,:]))
        maximum_index.append(np.max(k_length_sat_table[key][:,0,:]))
    max_idx = max(maximum_index)
    min_idx = min(minimum_index)
    assert max_idx - min_idx + 1  == N
    for key in sorted(k_length_sat_table.keys()):
        k_length_sat_table[key][:,0,:] -= min_idx
        k_length_sat_table[key] = k_length_sat_table[key].tolist()
        sat_table += k_length_sat_table[key]
    real_M = len(sat_table)
    assert real_M  == M
    max_k = max(k_length_sat_table.keys())
    min_k = min(k_length_sat_table.keys())
    if max_k != min_k:
        raise ValueError("This is not a max-ksat instances.")
    mask_tensor = clause_mask_tensor(N, M, sat_table)
    return N, M, mask_tensor

def clause_mask_tensor(N, M, sat_table):
    clause = []
    for ii in range(M):
        k = len(sat_table[ii][0])
        clause_m = torch.sparse_coo_tensor(
            [sat_table[ii][0], sat_table[ii][1]], 
            [1] * k,
            (N, 2)
        ).to_dense().unsqueeze(0)
        clause.append(clause_m.unsqueeze(0))
    clause_batch = torch.cat(clause, dim=0)    #[M, batch, N, q]  # sparse tensor values = k * M * batch
    return clause_batch 