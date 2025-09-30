import torch, re
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import warnings
from collections import defaultdict
import itertools
warnings.filterwarnings('ignore')

def parse_file(problem_type, filename, index_start=0, map_type = 'normal'):
    if problem_type in ['maxcut', 'bmincut', 'modularity', 'vertexcover']:
        n, m, J = read_graph(filename, index_start)
    elif problem_type == 'hyperbmincut':
        if map_type == 'normal':
            n, m, J = read_hypergraph_normal(filename, index_start)
        elif map_type == 'star':
            n, m, J = read_hypergraph_star(filename, index_start)
        elif map_type == 'clique':
            n, m, J = read_hypergraph_clique(filename, index_start)
        elif map_type == 'weighted_clique':
            n, m, J = read_hypergraph_weighted_clique(filename, index_start, alpha=0.5)
        elif map_type == 'bisecgraph':
            n, m, J = read_hypergraph_bisecgraph(filename, index_start)
    elif problem_type == 'maxksat':
        n, m, J = read_cnf(filename)
    return n, m, J


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

def read_hypergraph_normal(file, index_start=1):
    with open(file, "r") as f:
        # Read first line with number of vertices and hyperedges
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        # Collect all hyperedges
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)
    
    # Create all pairwise combinations
    all_pairs = []
    for vertices in hyperedges:
        if len(vertices) >= 2:
            # Generate all combinations of 2 vertices
            pairs = torch.combinations(torch.tensor(vertices), 2)
            all_pairs.append(pairs)
    
    # Concatenate all pairs
    indices = torch.cat(all_pairs, dim=0)
    
    # Create symmetric indices (u,v) and (v,u)
    indices_symmetric = torch.cat([indices, indices.flip(1)], dim=0)
    
    # Create values (all ones)
    values = torch.ones(indices_symmetric.shape[0])
    
    # Create sparse tensor and convert to dense
    J_sparse = torch.sparse_coo_tensor(indices_symmetric.t(), values, (n, n))
    J = J_sparse.to_dense()
    
    return n, m, J

def read_hypergraph_star(file, index_start=1):
    with open(file, "r") as f:
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)
    
    new_n = n + m
    all_pairs = []
    
    for he_idx, vertices in enumerate(hyperedges):
        center_node_id = n + he_idx
        
        for node in vertices:
            pair = torch.tensor([center_node_id, node])
            all_pairs.append(pair.unsqueeze(0))
            pair_reverse = torch.tensor([node, center_node_id])
            all_pairs.append(pair_reverse.unsqueeze(0))
    
    indices = torch.cat(all_pairs, dim=0)
    values = torch.ones(indices.shape[0])
    
    J_sparse = torch.sparse_coo_tensor(indices.t(), values, (new_n, new_n))
    J = J_sparse.to_dense()
    
    return new_n, m, J

def read_hypergraph_clique(file, index_start=1):
    with open(file, "r") as f:
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)
    
    # 收集所有边对
    all_pairs = []
    all_weights = []
    
    for vertices in hyperedges:
        if len(vertices) < 2:
            continue
            
        k = len(vertices)
        weight = 1.0 / (k - 1) if k > 1 else 1.0
        
        # 使用 combinations 生成所有边对
        pairs = torch.combinations(torch.tensor(vertices), 2)
        weights = torch.full((pairs.shape[0],), weight)
        
        all_pairs.append(pairs)
        all_weights.append(weights)
    
    # 合并所有边对
    all_pairs_tensor = torch.cat(all_pairs, dim=0)
    all_weights_tensor = torch.cat(all_weights, dim=0)
    
    # 对相同的边进行权重聚合
    # 先将边标准化为 (min, max) 形式
    normalized_pairs = torch.stack([
        torch.min(all_pairs_tensor, dim=1).values,
        torch.max(all_pairs_tensor, dim=1).values
    ], dim=1)
    
    # 使用 scatter_add 进行权重聚合
    # 将边对映射到唯一的索引
    unique_pairs, inverse_indices = torch.unique(normalized_pairs, dim=0, return_inverse=True)
    aggregated_weights = torch.zeros(unique_pairs.shape[0])
    aggregated_weights.scatter_add_(0, inverse_indices, all_weights_tensor)
    
    # 构建对称的邻接矩阵索引
    indices_u = torch.cat([unique_pairs[:, 0], unique_pairs[:, 1]])
    indices_v = torch.cat([unique_pairs[:, 1], unique_pairs[:, 0]])
    values = torch.cat([aggregated_weights, aggregated_weights])
    
    indices = torch.stack([indices_u, indices_v])
    
    J_sparse = torch.sparse_coo_tensor(indices, values, (n, n))
    J = J_sparse.to_dense()
    
    return n, m, J

def read_hypergraph_weighted_clique(file, index_start=1, alpha=0.5):
    with open(file, "r") as f:
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)
    
    hyperedge_weights = [1.0] * m
    
    # 使用张量计算节点度
    node_degrees = torch.zeros(n, dtype=torch.float)
    for he_idx, vertices in enumerate(hyperedges):
        weight = hyperedge_weights[he_idx]
        if vertices:  # 确保超边不为空
            node_degrees[torch.tensor(vertices)] += weight
    
    # 收集所有边对和对应的权重
    all_edges = []
    all_raw_weights = []
    
    for he_idx, vertices in enumerate(hyperedges):
        if len(vertices) < 2:
            continue
            
        he_weight = hyperedge_weights[he_idx]
        vertices_tensor = torch.tensor(vertices)
        
        # 生成所有边对组合
        pairs = torch.combinations(vertices_tensor, 2)
        
        # 计算每对边的初始权重
        for pair in pairs:
            u, v = pair[0].item(), pair[1].item()
            if u > v:  # 标准化
                u, v = v, u
            
            deg_u = node_degrees[u].item()
            deg_v = node_degrees[v].item()
            
            if deg_u > 0 and deg_v > 0:
                weight = he_weight * (1.0 / (deg_u * deg_v)) ** alpha
            else:
                weight = he_weight
            
            all_edges.append((u, v))
            all_raw_weights.append(weight)
    
    # 聚合相同边的权重
    edge_dict = defaultdict(float)
    for edge, weight in zip(all_edges, all_raw_weights):
        edge_dict[edge] += weight
    
    # 构建最终的张量
    edges = list(edge_dict.keys())
    weights = list(edge_dict.values())
    
    # 创建索引张量
    edges_tensor = torch.tensor(edges, dtype=torch.long)  # shape: [num_edges, 2]
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    
    # 构建对称的邻接矩阵
    indices_forward = edges_tensor.t()  # shape: [2, num_edges]
    indices_backward = torch.flip(edges_tensor, dims=[1]).t()  # 反向边
    
    indices = torch.cat([indices_forward, indices_backward], dim=1)
    values = torch.cat([weights_tensor, weights_tensor])
    
    J_sparse = torch.sparse_coo_tensor(indices, values, (n, n))
    J = J_sparse.to_dense()
    
    return n, m, J

def read_hypergraph_bisecgraph(file, index_start=1):
    with open(file, "r") as f:
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)
    
    n_new = n + m
    
    # 使用列表推导式批量生成边
    edge_pairs = []
    for he_idx, vertices in enumerate(hyperedges):
        hyperedge_node_id = n + he_idx
        # 为每个原节点生成双向边
        for node in vertices:
            edge_pairs.extend([(node, hyperedge_node_id), (hyperedge_node_id, node)])
    
    # 转换为张量（批量操作）
    edges_tensor = torch.tensor(edge_pairs, dtype=torch.long)
    indices = edges_tensor.t()
    values = torch.ones(indices.shape[1])
    
    J_sparse = torch.sparse_coo_tensor(indices, values, (n_new, n_new))
    J = J_sparse.to_dense()
    
    
    # 使用集合来避免重复，然后排序
    neighbor_sets = [set() for _ in range(n_new)]
    for u, v in edge_pairs:
        neighbor_sets[u].add(v)
    
    return n_new, m, J

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