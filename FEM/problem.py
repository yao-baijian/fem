import torch
import torch.nn.functional as Func
import numpy as np

def infer_qubo(J, p):
    """
    determine the configuration by mariginal probility and calculate the value
    of QUBO function
    """
    config = p.round()
    return config, expected_qubo(J, config)

def expected_qubo(J, p):
    """
    QUBO function of weights and mariginal probability p
    Parameters:
        J: torch.Tensor, shape: (N, N), weight matrix of the QUBO problem
        p: torch.Tensor, shape: (batch, N), mariginal probability of QUBO variables
    """
    return torch.bmm(
        (p @ J).reshape(-1, 1, J.shape[1]),
        p.reshape(-1, p.shape[1], 1)
    ).reshape(-1)

def manual_grad_qubo(J, p):
    """
    gradients of QUBO function
    """
    grad = 2 * (p*(1-p) * p @ J)
    # grad = 2 * (p*(1-p) * (p > 0.5).to(J.dtype) @ J)
    return grad

def infer_bmincut(J, p):
    """
    J: weight matrix, with shape [N,N], better with the csc format
    p: the marginal matrix, with shape [batch, N, q], q is the number of groups
    config is the configuration for n variables, with shape [batch, N, q]
    return the cut size, i.e. outer weights.
    """
    config = Func.one_hot(p.view(-1,J.shape[0],p.shape[-1]).argmax(dim=2), num_classes=p.shape[-1]).to(J.dtype)
    return config, expected_bmincut(J, config) / 2


def expected_bmincut(J,p):
    """
    p is the marginal matrix, with shape [batch, N,q], q is the number of groups
    config is the configuration for n variables, with shape [batch, N,q]
    return TWICE the expected cut size, i.e. outer weights. 
    """
    return ((J @ p) * (1-p)).sum((1, 2))


def manual_grad_bmincut(J, p, imba):
    temp = 1 - 2 * p
    tp = J @ temp + imba*(2 * p.sum(1,keepdim=True)- 2*p)
    h_grad = (tp  - (tp * p).sum(2,keepdim=True).expand(tp.shape))*p
    return h_grad


def balance_constrain(J, p, U_max, L_min):
    # relu hard regu
    # S_k = p.sum(dim=1)
    # config, result = infer_hyperbmincut(J, p)
    # optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    # best_config = config[optimal_inds[0]]
    # group_assignment = best_config.argmax(dim=1).cpu().numpy()
    # group_counts = np.bincount(group_assignment, minlength=4)

    # batch_size, n_nodes, n_clusters = p.shape
    
    # probabilities = torch.softmax(p, dim=2)
    # assignments = torch.argmax(probabilities, dim=2)
    # one_hot = torch.nn.functional.one_hot(assignments, num_classes=n_clusters)
    # S_k = one_hot.sum(dim=1).float()

    # print(f"group_counts: {group_counts}")
    # print(f"S_k: {S_k}")
    # upper_violation = torch.relu(S_k - U_max)
    # lower_violation = torch.relu(L_min - S_k)
    # balance_loss = upper_violation.sum(dim=1)
    # balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)


    # softplus 
    # S_k = p.sum(dim=1)
    # softplus(x) = log(1 + exp(beta * x)) / beta
    # upper_violation = torch.log(1 + torch.exp(beta * (S_k - U_max))) / beta
    # lower_violation = torch.log(1 + torch.exp(beta * (L_min - S_k))) / beta

    # upper_violation = torch.nn.functional.softplus(beta *(S_k - U_max))
    # lower_violation = torch.nn.functional.softplus(beta *(L_min - S_k))

    # upper_violation = torch.where(
    #     x > 20,
    #     x / beta,
    #     torch.log1p(torch.exp(x)) / beta 
    # )

    # balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)
    # print(f"balance_loss: {balance_loss}")
    # return balance_loss


    batch_size, n_nodes, n_clusters = p.shape
    
    # 使用概率和作为分组大小的可微近似
    S_k = p.sum(dim=1)  # [batch, n_clusters] - 完全可微
    
    # 仅用于打印实际分组情况（不影响梯度）
    # with torch.no_grad():
    #     probabilities = torch.softmax(p, dim=2)
    #     assignments = torch.argmax(probabilities, dim=2)
    #     actual_counts = torch.nn.functional.one_hot(assignments, n_clusters).sum(dim=1)
    #     print(f"actual partition: {actual_counts.tolist()[0]} soft partition: {S_k.tolist()[0]}")
    
    upper_violation = torch.relu(S_k - U_max)
    lower_violation = torch.relu(L_min - S_k)

    balance_loss = upper_violation.sum(dim=1) + lower_violation.sum(dim=1)
    # print(f"balance_loss: {balance_loss}")
    return balance_loss

def infer_hyperbmincut(J, p):
    config = Func.one_hot(p.view(-1,J.shape[0],p.shape[-1]).argmax(dim=2), num_classes=p.shape[-1]).to(J.dtype)
    return config, expected_bmincut(J, config) / 2

def expected_hyperbmincut(J, p,  hyperedges):
    # return ((J @ p) * (1-p)).sum((1, 2))
    # n_hyperedges = J.shape
    # n_groups = p.shape[1]
    
    # log_prob_no_node = torch.zeros(n_hyperedges, n_groups, device=p.device)
    
    # for e in range(n_hyperedges):
    #     nodes_in_e = torch.where(J[e] > 0)[0]
    #     if len(nodes_in_e) > 0:
    #         log_one_minus_p = torch.log(1 - p[nodes_in_e] + 1e-10) 
    #         log_prob_no_node[e] = torch.sum(log_one_minus_p, dim=0)
    
    # prob_no_node = torch.exp(log_prob_no_node)
    # p_ek = 1 - prob_no_node
    # expected_lambda = p_ek.sum(dim=1)
    # expected_cut_value = (expected_lambda - 1).sum(1)
    
    # return expected_cut_value



    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
    #     expected_nodes_per_cluster = torch.sum(he_probs, dim=1)  # [batch, m]
    #     max_expected_nodes = torch.max(expected_nodes_per_cluster, dim=1)[0]
    #     cut_value = 1 - (max_expected_nodes / k)
    #     total_cut_value = total_cut_value + cut_value
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value


    total_cut_value = 0.0
    
    for he_idx, he in enumerate(hyperedges):
        weight = 1.0
        k = len(he)
        m = p.shape[2]

        he_probs = p[:, he, :]  # [batch, k, num_clusters]
        expected_nodes_per_cluster = torch.sum(he_probs, dim=1)  # [batch, m]

        # 最简单的连续映射: 跨区数 = m - (m-1) * max_ratio
        max_ratio = torch.max(expected_nodes_per_cluster, dim=1)[0] / k
        expected_crossing = m * (1 - max_ratio)
        
        total_cut_value += expected_crossing
    
    # print(f"Weighted Cut value: {total_cut_value}")
    return total_cut_value



    # total_cut_value = 0.0
    # m = 4  # 簇数
    
    # # 预定义所有组合
    # pair_masks = torch.tensor([
    #     [1, 1, 0, 0],  # (0,1)
    #     [1, 0, 1, 0],  # (0,2) 
    #     [1, 0, 0, 1],  # (0,3)
    #     [0, 1, 1, 0],  # (1,2)
    #     [0, 1, 0, 1],  # (1,3)
    #     [0, 0, 1, 1],  # (2,3)
    # ], dtype=torch.float32, device=p.device)
    
    # triple_masks = torch.tensor([
    #     [1, 1, 1, 0],  # (0,1,2)
    #     [1, 1, 0, 1],  # (0,1,3)
    #     [1, 0, 1, 1],  # (0,2,3)
    #     [0, 1, 1, 1],  # (1,2,3)
    # ], dtype=torch.float32, device=p.device)
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, m]
    #     batch_size = he_probs.shape[0]
        
    #     # 1. 计算 P(跨区数=1) - 向量化
    #     prob_single_cluster = torch.sum(torch.prod(he_probs, dim=1), dim=1)  # [batch]
        
    #     # 2. 计算 P(跨区数=2) - 向量化
    #     # 扩展维度用于广播 [6, 4] -> [batch, 6, k, 4]
    #     pair_masks_expanded = pair_masks.view(1, 6, 1, 4).expand(batch_size, 6, k, 4)
    #     he_probs_expanded = he_probs.unsqueeze(1).expand(batch_size, 6, k, 4)
        
    #     # 计算每个组合的概率 [batch, 6]
    #     pair_probs = torch.prod(
    #         torch.sum(he_probs_expanded * pair_masks_expanded, dim=3), 
    #         dim=2
    #     )
    #     sum_2comb = torch.sum(pair_probs, dim=1)  # [batch]
    #     prob_2_clusters = sum_2comb - 2 * prob_single_cluster
        
    #     # 3. 计算 P(跨区数=3) - 向量化
    #     triple_masks_expanded = triple_masks.view(1, 4, 1, 4).expand(batch_size, 4, k, 4)
    #     he_probs_expanded_triple = he_probs.unsqueeze(1).expand(batch_size, 4, k, 4)
        
    #     triple_probs = torch.prod(
    #         torch.sum(he_probs_expanded_triple * triple_masks_expanded, dim=3), 
    #         dim=2
    #     )
    #     sum_3comb = torch.sum(triple_probs, dim=1)  # [batch]
    #     prob_3_clusters = sum_3comb - 2 * sum_2comb + 3 * prob_single_cluster
        
    #     # 4. 计算 P(跨区数=4)
    #     prob_4_clusters = 1 - prob_single_cluster - prob_2_clusters - prob_3_clusters
        
    #     epsilon = 1e-3
    #     prob_2 = torch.log(prob_2_clusters + epsilon)
    #     prob_3 = torch.log(prob_3_clusters + epsilon) 
    #     prob_4 = torch.log(prob_4_clusters + epsilon)

    #     # prob_2 = torch.log(torch.sqrt(prob_2_clusters))
    #     # prob_3 = torch.log(torch.sqrt(prob_3_clusters))
    #     # prob_4 = torch.sqrt(prob_4_clusters) - epsilon
        
    #     total_cut_value += prob_2 + 2 * prob_3 + 3 * prob_4
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value

        # total_cut_value = total_cut_value + prob_2_clusters + prob_3_clusters * 2 + prob_4_clusters * 3
        
        # if he_idx == 0:
        #     print(f"超边{k}: P(1簇)={prob_single_cluster[0]:.6f}, P(2簇)={prob_2_clusters[0]:.6f}, "
        #           f"P(3簇)={prob_3_clusters[0]:.6f}, P(4簇)={prob_4_clusters[0]:.6f}, "
        #           f"Cut期望={cut_expectation[0]:.6f}")
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value

    # threshold=15
    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
    #     batch_size, _, num_clusters = he_probs.shape
        
    #     if k <= threshold:
    #         # 对于小k，使用精确计算
    #         prob_single_cluster = torch.zeros(batch_size, device=p.device)
    #         for cluster_j in range(num_clusters):
    #             prob_all_in_j = torch.prod(he_probs[:, :, cluster_j], dim=1)
    #             prob_single_cluster += prob_all_in_j
            
    #         cut_expectation = 1 - torch.clamp(prob_single_cluster, 0.0, 1.0)
            
    #     else:
    #         # 对于大k，使用成对近似
    #         # 计算节点分配的"集中度"
    #         cluster_weights = torch.sum(he_probs, dim=1) / k  # [batch, num_clusters]
    #         max_concentration = torch.max(cluster_weights, dim=1)[0]  # 最大簇的节点比例
            
    #         # 集中度越高，越不可能被切割
    #         cut_expectation = 1 - max_concentration
        
    #     cut_value = cut_expectation * weight
    #     total_cut_value = total_cut_value + cut_value
        
    #     # if he_idx < 3:
    #     #     method = "精确" if k <= threshold else "近似"
    #     #     print(f"超边{he_idx}(k={k}, {method}): cut期望={cut_expectation[0]:.6f}")
    
    # print(f"Hybrid Cut value: {total_cut_value}")
    # return total_cut_value

    
    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
    #     expected_nodes_per_cluster = torch.sum(he_probs, dim=1)
    #     e = expected_nodes_per_cluster / k  # 归一化的期望节点数
    #     p_used = 1 - torch.exp(-k * e)
        
    #     # 期望跨区数
    #     expected_crossing = torch.sum(p_used, dim=1)
        
    #     # 近似方差（假设独立）
    #     variance = torch.sum(p_used * (1 - p_used), dim=1)
        
    #     # 使用正态近似计算 P(跨区数 ≥ 2)
    #     # 但更简单：cut_value = max(0, expected_crossing - 1)
    #     cut_value = torch.relu(expected_crossing - 1) * weight
    #     total_cut_value = total_cut_value + cut_value
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value


    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
    #     expected_nodes = torch.sum(he_probs, dim=1)  # [batch, m]
        
    #     # 更好的近似：考虑概率的方差
    #     p_used = torch.zeros_like(expected_nodes)
        
    #     for cluster_j in range(he_probs.shape[2]):
    #         probs_j = he_probs[:, :, cluster_j]  # [batch, k]
    #         e_j = expected_nodes[:, cluster_j]   # 期望节点数
            
    #         # 使用一阶泰勒展开近似
    #         # log(P(无节点)) = ∑ log(1-p_i,j) ≈ -∑ p_i,j - 0.5∑ p_i,j^2
    #         sum_p = torch.sum(probs_j, dim=1)
    #         sum_p2 = torch.sum(probs_j**2, dim=1)
            
    #         # P(无节点) ≈ exp(-sum_p - 0.5*sum_p2)
    #         p_no_nodes = torch.exp(-sum_p - 0.5 * sum_p2)
    #         p_used[:, cluster_j] = 1 - p_no_nodes
        
    #     expected_crossing = torch.sum(p_used, dim=1)
    #     cut_value = torch.relu(expected_crossing - 1) * weight
    #     total_cut_value = total_cut_value + cut_value
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value

    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, m]
        
    #     # 直接计算每个簇被使用的概率
    #     p_used = 1 - torch.prod(1 - he_probs, dim=1)  # [batch, m]
        
    #     # 期望跨区数
    #     expected_crossing = torch.sum(p_used, dim=1)
        
    #     # 或者直接计算Kahypar cut期望
    #     prob_single_cluster = torch.sum(torch.prod(he_probs, dim=1), dim=1)
    #     cut_value = (1 - prob_single_cluster) * weight
        
    #     total_cut_value = total_cut_value + cut_value
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value
    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
    #     batch_size, _, num_clusters = he_probs.shape
        
    #     # 使用log空间计算避免数值下溢
    #     log_prob_single_cluster = None
        
    #     for cluster_j in range(num_clusters):
    #         # log(P(所有节点在簇j)) = ∑ log(p_i,j)
    #         log_probs = torch.log(he_probs[:, :, cluster_j] + 1e-12)
    #         log_prob_all_in_j = torch.sum(log_probs, dim=1)
            
    #         if log_prob_single_cluster is None:
    #             log_prob_single_cluster = log_prob_all_in_j
    #         else:
    #             # log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    #             max_log = torch.max(log_prob_single_cluster, log_prob_all_in_j)
    #             log_prob_single_cluster = max_log + torch.log(
    #                 torch.exp(log_prob_single_cluster - max_log) + 
    #                 torch.exp(log_prob_all_in_j - max_log) + 1e-12
    #             )
        
    #     prob_single_cluster = torch.exp(log_prob_single_cluster)
        
    #     # 确保概率在合理范围内
    #     prob_single_cluster = torch.clamp(prob_single_cluster, 0.0, 1.0)
    #     cut_expectation = 1 - prob_single_cluster
        
    #     cut_value = cut_expectation * weight
    #     total_cut_value = total_cut_value + cut_value
        
    #     if he_idx < 2:  # 只打印前两个超边
    #         print(f"超边{he_idx}(k={k}): P(单簇)={prob_single_cluster[0]:.8f}")
    
    # print(f"Total Cut value: {total_cut_value}")
    # return total_cut_value
    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
    #     m = p.shape[2]
        
    #     he_probs = p[:, he, :]  # [batch, k, num_clusters]
        
    #     # 计算一阶矩：E[X_j]
    #     p_used = 1 - torch.prod(1 - he_probs, dim=1)  # [batch, m]
    #     mean_crossing = torch.sum(p_used, dim=1)
        
    #     # 计算二阶矩：E[X_j X_k]
    #     second_moment = 0.0
    #     for j in range(m):
    #         for l in range(j+1, m):
    #             # P(簇j和簇l都被使用) = 1 - P(不用j) - P(不用l) + P(都不用j和l)
    #             p_no_j = torch.prod(1 - he_probs[:, :, j], dim=1)
    #             p_no_l = torch.prod(1 - he_probs[:, :, l], dim=1)
    #             p_no_jl = torch.prod(1 - he_probs[:, :, j] - he_probs[:, :, l], dim=1)
    #             p_both_used = 1 - p_no_j - p_no_l + p_no_jl
    #             second_moment += p_both_used
        
    #     # 使用切比雪夫不等式近似 P(跨区数 ≥ 2)
    #     variance = second_moment + mean_crossing - mean_crossing**2
    #     # P(跨区数 ≥ 2) ≈ 1 - P(跨区数 ≤ 1)
    #     # 但更实用：cut_value = (mean_crossing - 1) 当 mean_crossing > 1.5，否则用概率近似
    #     cut_prob = torch.sigmoid((mean_crossing - 1.5) * 10)  # 平滑的阶跃函数
    #     cut_value = cut_prob * weight
    #     total_cut_value = total_cut_value + cut_value
    
    # print(f"Weighted Cut value: {total_cut_value}")
    # return total_cut_value

    # total_cut_value = 0.0
    
    # for he_idx, he in enumerate(hyperedges):
    #     weight = 1.0
    #     k = len(he)
        
    #     he_probs = p[:, he, :]
    #     expected_nodes_per_cluster = torch.sum(he_probs, dim=1)
        
    #     temperature = 0.1
    #     weights = torch.softmax(expected_nodes_per_cluster / temperature, dim=1)
    #     weighted_max = torch.sum(weights * expected_nodes_per_cluster, dim=1)
        
    #     cut_value = 1 - (weighted_max / k)
    #     total_cut_value = total_cut_value + cut_value * weight
        
    # return total_cut_value



def manual_grad_hyperbmincut(J, p, U_max, L_min, n, h, imbalance_weight, q):
    
    # 1. 计算概率分布 (需要softmax)
    # p = torch.softmax(h, dim=-1)  # [batch, n_nodes, q]
    
    group_sizes = p.sum(dim=1)  # [batch, q] - 每个分组的"软节点数"
    
    temperature = 0.1
    indicator_upper = torch.sigmoid((group_sizes - U_max) / temperature)  
    indicator_lower = torch.sigmoid((L_min - group_sizes) / temperature)  
    
    balance_grad =  imbalance_weight * (indicator_upper - indicator_lower)  # [batch, q]
    
    # 扩展到每个节点：balance_grad_expanded[i,k] = balance_grad[batch,k]
    balance_grad_expanded = balance_grad.unsqueeze(1).expand(-1, n, -1)  # [batch, n_nodes, q]
    
    cut_grad = torch.zeros_like(h)
    for k in range(q):
        p_k = p[:, :, k]  # [batch, n_nodes]
        for b in range(batch_size):
            cut_grad[b, :, k] = torch.matmul(J, 1 - 2 * p_k[b])
    
    total_grad = cut_grad + balance_grad_expanded
    
    return total_grad


def infer_maxcut(J, p):
    """
    J: weight matrix, with shape [N, N]
    p: the marginal matrix, with shape [batch, N], p[:, x] represent the
        probability of x variable to be 1 
    config is the configuration for N variables, with shape [batch, N]
    return the cut size, i.e. outer weights.
    """
    config = p.round()
    return config, expected_cut(J, config) / 2

def expected_cut(J, p):
    """
    p is the marginal matrix, with shape [batch, N]
    config is the configuration for n variables, with shape [batch, N]
    return TWICE the expected cut size, i.e. outer weights. 
    """
    return 2 * ((p @ J) * (1-p)).sum(1)

def manual_grad_maxcut(J, p, discretization=False):
    p_prime = p.round() if discretization else p
    h_grad = (2 * p_prime - 1) @ J * (1-p) * p
    return h_grad

def manual_grad_modularity(J, p, m, d):
    temp = d * p
    tp = -J @ p + d * ((temp).sum(1,keepdim=True)-temp)/m
    h_grad = (tp  - (tp * p).sum(2,keepdim=True).expand(tp.shape))*p
    return h_grad

def imbalance_penalty(p):
    """
    p is the marginal matrix, with shape [batch, N,q], q is the number of groups
    config is the configuration for n variables, with shape [batch, N,q]
    return an anti-ferromagnetic all-to-all interaction panelty which equals to
    #   \sum_i\sum_{s_i}\sum_{j\neq i}p_i(s_i)p_j(s_i)
    # = \sum_{s_i}\sum_ip_i(s_i)\sum_jp_j(s_i) - \sum_i\sum_{s_i}p_i(s_i)*p_i(s_i)
    # = \sum_{s_i}(\sum_{i}p_i(s_i))**2 - \sum_i\sum_{s_i}p_i(s_i)*p_i(s_i)

    """
    return ((p.sum(1))**2).sum(1) - (p*p).sum(2).sum(1)

def expected_inner_weight(J,p):
    return 0.5*(J.sum() - expected_cut(J,p))

def expected_inner_weight_configmodel(J,p):
    """
    \frac{1}{2m}\sum_i\sum_j\frac{d_i*d_j}\delta(s_i,s_j)
    =\frac{1}{2m}\sum_i\sum_{s_i}d_i\sum_{j\neq i}d_jp_i(s_i)p_j(s_i)
    =\frac{1}{2m}\sum_i\sum_{s_i}d_ip_i(s_i)\sum_{j\neq i}d_jp_j(s_i)
    =\frac{1}{2m}(\sum_{s_i}\sum_id_ip_i(s_i)\sum_{j}d_jp_j(s_i) - \sum_id_id_i\sum_{s_i}p_i(s_i)*p_i(s_i))
    =\frac{1}{2m}(\sum_{s_i}(\sum_id_ip_i(s_i))**2 - \sum_i\sum_{s_i}d_i*d_i*p_i(s_i)*p_i(s_i))
    """
    d = J.to_dense().sum(1).reshape([1,p.shape[1],1]).expand(p.shape)
    m2 = J.sum()
    return (((d*p).sum(1)**2).sum(1) - (d*p*d*p).sum(2).sum(1)) / m2

def manual_grad_maxksat(clause_batch, p):
    M, batch = clause_batch.shape[:2]
    minus_p = 1 - 0.99999 * p
    prod = clause_batch * minus_p
    value = prod._values().reshape(M, batch, -1)   # # values = k * M * batch
    value_prod = prod._values().reshape(M, batch, -1).prod(-1,keepdim=True)
    grad = torch.sparse_coo_tensor(
        prod.coalesce().indices(),
        (-value_prod/value).reshape([1,-1]).squeeze(0), 
        prod.shape
    ).sum(0,keepdim=True).to_dense()
    h_grad = (grad  - (grad * p).sum(3,keepdim=True).expand(grad.shape))*p
    return h_grad

def expected_maxksat(clause_batch, p):
    M, batch = clause_batch.shape[:2]
    minus_p = 1 - p
    prod = clause_batch * minus_p
    value_prod = prod._values().reshape(M, batch, -1).prod(-1,keepdim=True)
    energy = value_prod.sum(0).reshape(1,-1).squeeze(0)
    return energy
    
def infer_maxksat(clause_batch, p):
    config = Func.one_hot(p.view(1,clause_batch.shape[1],clause_batch.shape[2],-1).argmax(dim=3), num_classes=p.shape[-1]).to(clause_batch.dtype)
    return config, expected_maxksat(clause_batch, config)



# coords functions
def get_site_coordinates_from_index(expected_site_index, area_width):
    x_coords = expected_site_index % area_width
    y_coords = expected_site_index // area_width
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords.float()

def get_site_coordinates_from_px_py(p_x, p_y):
    x = torch.argmax(p_x, dim=2)
    y = torch.argmax(p_y, dim=2)
    return torch.stack([x, y], dim=2)

def get_site_distance_matrix(coords):
    coords_i = coords.unsqueeze(1)
    coords_j = coords.unsqueeze(0)
    distances = torch.sum(torch.abs(coords_i - coords_j), dim=2)
    return distances

def get_expected_placements_from_index(p, site_coords_matrix):
    expected_coords = torch.matmul(p, site_coords_matrix)
    return expected_coords

def get_site_coords_all(num_locations, area_width):
    indices = torch.arange(num_locations, dtype=torch.float32)
    x_coords = indices % area_width
    y_coords = indices // area_width
    return torch.stack([x_coords, y_coords], dim=1)

def get_grid_coords_xy(area_width):
    grid_x_coords = torch.arange(area_width, dtype=torch.float32)
    grid_y_coords = torch.arange(area_width, dtype=torch.float32)
    return grid_x_coords, grid_y_coords

def get_hard_placements_from_index(p, site_coords_matrix):
    site_indices = torch.argmax(p, dim=2)
    hard_coords = site_coords_matrix[site_indices]  # [batch_size, num_instances, 2]
    return hard_coords

def get_placements_from_index_st(p, site_coords_matrix):
    with torch.no_grad():
        site_indices = torch.argmax(p, dim=2)
        hard_coords = site_coords_matrix[site_indices]
    
    expected_coords = torch.matmul(p, site_coords_matrix)
    straight_coords = expected_coords + (hard_coords - expected_coords).detach()
    
    return straight_coords

def get_placements_from_xy_st(p_x, p_y, grid_x_coords, grid_y_coords):
    with torch.no_grad():
        x_hard = grid_x_coords[torch.argmax(p_x, dim=2)] 
        y_hard = grid_y_coords[torch.argmax(p_y, dim=2)] 
        hard_coords = torch.stack([x_hard, y_hard], dim=2)
    
    expected_x = torch.matmul(p_x, grid_x_coords.unsqueeze(1)).squeeze(2) 
    expected_y = torch.matmul(p_y, grid_y_coords.unsqueeze(1)).squeeze(2)
    expected_coords = torch.stack([expected_x, expected_y], dim=2)
    straight_coords = expected_coords + (hard_coords - expected_coords).detach()
    return straight_coords

# hpwl loss
def get_hpwl_loss(J, p, site_coords_matrix):
    expected_coords = get_placements_from_index_st(p, site_coords_matrix)

    # expected_coords = get_expected_placements_from_index(p, site_coords_matrix)
    
    coords_i = expected_coords.unsqueeze(2)  # [batch_size, num_instances, 1, 2]
    coords_j = expected_coords.unsqueeze(1)  # [batch_size, 1, num_instances, 2]

    manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
    
    weighted_dist = manhattan_dist * J.unsqueeze(0)  # [batch_size, num_instances, num_instances]
    
    triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
    weighted_dist_triu = weighted_dist[:, triu_mask]  # [batch_size, num_pairs]
    
    total_wirelength = torch.sum(weighted_dist_triu, dim=1)  # [batch_size]
    return total_wirelength

def get_hpwl_loss_from_net_tensor_lse(p, net_tensor, site_coords_matrix, gamma=0.1):
    batch_size, num_instances, num_sites = p.shape
    num_nets = net_tensor.shape[0]
    
    # 计算期望坐标
    expected_coords = torch.matmul(p, site_coords_matrix)
    expected_x = expected_coords[..., 0]
    expected_y = expected_coords[..., 1]
    
    # 扩展用于批量计算
    net_tensor_expanded = net_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    net_x = expected_x.unsqueeze(1).expand(-1, num_nets, -1)
    net_y = expected_y.unsqueeze(1).expand(-1, num_nets, -1)
    
    small_value = -1e6
    
    def compute_boundaries(coords, net_mask):
        coords_masked = torch.where(net_mask, coords, torch.tensor(small_value, device=p.device))
        coords_neg_masked = torch.where(net_mask, -coords, torch.tensor(small_value, device=p.device))
        
        coord_max = gamma * torch.logsumexp(coords_masked / gamma, dim=2)
        coord_min = -gamma * torch.logsumexp(coords_neg_masked / gamma, dim=2)
        
        return coord_max, coord_min
    
    # 计算x和y方向的边界
    x_max, x_min = compute_boundaries(net_x, net_tensor_expanded)
    y_max, y_min = compute_boundaries(net_y, net_tensor_expanded)
    
    # 计算HPWL
    net_hpwl = (x_max - x_min) + (y_max - y_min)
    total_hpwl = torch.sum(net_hpwl, dim=1)
    
    return total_hpwl



    # batch_size, num_instance, num_sites = p.shape
    # penalty_strength = 2
    # site_usage = torch.sum(p, dim=1)
    # overuse = torch.relu(site_usage - num_instance / num_sites)
    # penalty = torch.sum(torch.exp(overuse * penalty_strength) - 1.0, dim=1)
    # # penalty = torch.sum(overuse ** 3, dim=1)
    # return penalty

    # _, num_instance, num_sites = p.shape
    # site_usage = torch.sum(p, dim=1)
    # log_penalty = torch.sum(torch.log1p(Func.relu(site_usage - num_instance / num_sites)), dim=1)
    # return log_penalty

def get_hpwl_loss_xy_simple(J, p_x, p_y):
    
    batch_size, num_instances, num_x = p_x.shape

    grid_x_coords, grid_y_coords = get_grid_coords_xy(num_x)

    expected_coords = get_placements_from_xy_st(p_x, p_y, grid_x_coords, grid_y_coords)

    coords_i = expected_coords.unsqueeze(2)  # [batch_size, num_instances, 1, 2]
    coords_j = expected_coords.unsqueeze(1)  # [batch_size, 1, num_instances, 2]

    manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
    
    weighted_dist = manhattan_dist * J.unsqueeze(0)
    
    triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
    weighted_dist_triu = weighted_dist[:, triu_mask]
    
    total_wirelength = torch.sum(weighted_dist_triu, dim=1)
    return total_wirelength

# timing loss
def get_timing_loss(design, timer):
    wirelength = get_hpwl_loss(design)
    return max(0, 1000 - wirelength / 1000)

# position constraints loss
def get_constraints_loss(p):
    _, num_instance, num_sites = p.shape
    # p_normalized = Func.softmax(p, dim=-1)
    site_usage = torch.sum(p, dim=1)
    site_constraint_softplus = torch.sum(Func.softplus(10 * (site_usage - num_instance / num_sites))**2, dim=1)
    return site_constraint_softplus

def get_constraints_loss_electric_field(p, site_coords_matrix, push_strength=10.0):
    """
    电场力式约束：基于密度梯度产生推力
    """
    batch_size, num_instances, num_sites = p.shape
    site_usage = torch.sum(p, dim=1)  # [batch_size, num_sites]
    
    # 创建密度图 [batch_size, layout_size, layout_size]
    layout_size = int(num_sites ** 0.5)
    density_map = site_usage.reshape(batch_size, layout_size, layout_size)
    
    # 计算密度梯度（电场方向）
    # 使用Sobel算子计算密度梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=p.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=p.device).view(1, 1, 3, 3)
    
    # 计算密度梯度 [batch_size, 2, layout_size, layout_size]
    density_grad_x = Func.conv2d(density_map.unsqueeze(1), sobel_x, padding=1)
    density_grad_y = Func.conv2d(density_map.unsqueeze(1), sobel_y, padding=1)
    density_grad = torch.stack([density_grad_x.squeeze(1), density_grad_y.squeeze(1)], dim=1)
    
    # 计算每个实例受到的推力
    expected_coords = torch.matmul(p, site_coords_matrix)  # [batch_size, num_instances, 2]
    
    # 将连续坐标映射到网格索引
    grid_coords = (expected_coords * (layout_size - 1)).long()  # [batch_size, num_instances, 2]
    
    total_push_force = torch.zeros(batch_size, device=p.device)
    
    for b in range(batch_size):
        for i in range(num_instances):
            x, y = grid_coords[b, i, 0], grid_coords[b, i, 1]
            
            # 确保在边界内
            x = torch.clamp(x, 0, layout_size-1)
            y = torch.clamp(y, 0, layout_size-1)
            
            # 获取该位置的密度梯度（推力方向）
            force_direction = density_grad[b, :, y, x]  # [2]
            
            # 推力大小与密度成正比
            local_density = density_map[b, y, x]
            force_magnitude = local_density * push_strength
            
            # 推力损失：实例应该沿着梯度反方向移动（从高密度到低密度）
            # 我们惩罚实例不沿着推力方向移动的程度
            instance_force = force_magnitude * torch.norm(force_direction)
            total_push_force[b] += instance_force
    
    return total_push_force

def get_constraints_loss_max_prob(p, threshold=0.7):
    batch_size, num_instances, num_sites = p.shape
    max_probs, max_indices = torch.max(p, dim=2)
    max_prob_mask = torch.zeros_like(p)
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_instances)
    instance_indices = torch.arange(num_instances).unsqueeze(0).expand(batch_size, -1)
    max_prob_mask[batch_indices, instance_indices, max_indices] = max_probs
    
    site_usage = torch.sum(max_prob_mask, dim=1) 
    target_usage = num_instances / num_sites
    excess_usage = Func.relu(site_usage - target_usage)
    constraint_loss = torch.sum(excess_usage ** 2, dim=1)
    return constraint_loss

def get_constraints_loss_hard_exclusion_st(p, site_coords_matrix, exclusion_radius=1.0, strength=50.0):
    batch_size, num_instances, num_sites = p.shape
    
    # 硬坐标（用于前向计算）
    with torch.no_grad():
        site_indices = torch.argmax(p, dim=2)
        if site_coords_matrix.dim() == 2:
            hard_coords = site_coords_matrix[site_indices]
        else:
            batch_indices = torch.arange(batch_size, device=p.device).unsqueeze(1).expand(-1, num_instances)
            hard_coords = site_coords_matrix[batch_indices, site_indices, :]
    
    # 软坐标（用于梯度）
    expected_coords = torch.matmul(p, site_coords_matrix)
    
    # 硬重叠计算（前向）
    hard_overlap_penalty = torch.zeros(batch_size, device=p.device)
    for b in range(batch_size):
        overlap_count = 0
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                distance = torch.norm(hard_coords[b, i] - hard_coords[b, j])
                if distance < exclusion_radius:
                    overlap_count += 1
                    overlap_penalty = (exclusion_radius - distance) ** 2
                    hard_overlap_penalty[b] += overlap_penalty
        
        if overlap_count > 0:
            hard_overlap_penalty[b] += overlap_count * 10.0
    
    # 软重叠计算（梯度）
    soft_overlap_penalty = torch.zeros(batch_size, device=p.device)
    for b in range(batch_size):
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                distance = torch.norm(expected_coords[b, i] - expected_coords[b, j])
                if distance < exclusion_radius:
                    overlap_penalty = (exclusion_radius - distance) ** 2
                    soft_overlap_penalty[b] += overlap_penalty
    
    # Straight-Through
    final_penalty = soft_overlap_penalty + (hard_overlap_penalty - soft_overlap_penalty).detach()
    
    return final_penalty * strength

def get_constraints_loss_centering(p, site_coords_matrix, center_strength=5.0):
    batch_size, num_instances, num_sites = p.shape

    expected_coords = torch.matmul(p, site_coords_matrix)
    center = torch.mean(site_coords_matrix, dim=0)
    
    total_centering_loss = torch.zeros(batch_size, device=p.device)
    
    for b in range(batch_size):
        # 计算所有实例到中心的平均距离
        distances_to_center = torch.norm(expected_coords[b] - center, dim=1)
        avg_distance = torch.mean(distances_to_center)
        total_centering_loss[b] = avg_distance * center_strength
    
    return total_centering_loss

def get_constraints_loss_site_exclusivity_st(p, strength=10.0):
    batch_size, num_instances, num_sites = p.shape
    
    with torch.no_grad():
        site_indices = torch.argmax(p, dim=2)
    
    total_overlap_penalty = torch.zeros(batch_size)
    
    for b in range(batch_size):
        # 使用bincount统计每个站点的实例数量
        site_counts = torch.bincount(site_indices[b], minlength=num_sites)
        
        # 惩罚有多个实例的站点
        overlapping_sites = site_counts > 1
        overlap_degree = torch.sum(site_counts[overlapping_sites] - 1)
        total_overlap_penalty[b] = overlap_degree * strength
    
    site_usage = torch.sum(p, dim=1)  # [batch_size, num_sites]
    
    soft_overlap = Func.softplus(site_usage - num_instances / num_sites)
    soft_overlap_penalty = torch.sum(soft_overlap, dim=1) * strength
    
    final_penalty = soft_overlap_penalty + (total_overlap_penalty - soft_overlap_penalty).detach()
    
    return final_penalty

def get_constraints_loss_enhanced(p, bin_sizes=[4, 8, 16], weights=[1.0, 0.5, 0.2]):
    batch_size, num_instances, num_sites = p.shape
    site_usage = torch.sum(p, dim=1)  # [batch_size, num_sites]
    
    # 假设布局是方形的
    layout_size = int(num_sites ** 0.5)
    density_map = site_usage.view(batch_size, layout_size, layout_size)
    
    total_constraint = 0.0
    
    for bin_size, weight in zip(bin_sizes, weights):
        # 使用平均池化计算局部密度
        kernel = torch.ones(1, 1, bin_size, bin_size, device=p.device) / (bin_size * bin_size)
        local_density = Func.conv2d(
            density_map.unsqueeze(1), 
            kernel, 
            padding=bin_size//2, 
            stride=1
        ).squeeze(1)
        
        # 更强的惩罚：目标密度更低，惩罚系数更高
        target_density = num_instances / num_sites * 0.6  # 更严格的目标
        density_excess = Func.relu(local_density - target_density)
        
        # 使用更激进的惩罚函数
        bin_constraint = torch.sum(density_excess ** 3, dim=(1, 2))  # 三次方惩罚
        total_constraint += weight * bin_constraint
    
    return total_constraint

def get_constraints_loss_xy(p_x, p_y):
    batch_size, num_instances, num_x = p_x.shape
    _, _, num_y = p_y.shape

    usage = num_instances / (num_x * num_y)
    p_x_expanded = p_x.unsqueeze(3)  # [batch_size, num_instances, num_x, 1]
    p_y_expanded = p_y.unsqueeze(2)  # [batch_size, num_instances, 1, num_y]
    p_instance_xy = p_x_expanded * p_y_expanded  # [batch_size, num_instances, num_x, num_y]

    p_position_usage = torch.sum(p_instance_xy, dim=1)  # [batch_size, num_x, num_y]
    p_position_usage_flat = p_position_usage.flatten(start_dim=1)  # [batch_size, num_x * num_y]
    constraint_loss = torch.sum(Func.softplus(10 * (p_position_usage_flat - usage) ** 2 ), dim=1)  # [batch_size, num_x, num_y]
    return constraint_loss

# expected placement loss
def expected_fpga_placement(J, p, io_site_connect_matrix, site_coords_matrix, net_sites_tensor, best_hpwl):
    total_energy = 0
    hpwl_weight, timing_weight, constrain_weight = 1, 1, 10

    current_hpwl = get_hpwl_loss(J, p, site_coords_matrix)
    # current_hpwl = get_hpwl_loss_lse(J, p, bbox_length, site_coords_matrix)

    # current_hpwl = get_hpwl_loss_from_net_tensor_lse(p, net_sites_tensor, site_coords_matrix)

    improved_mask = current_hpwl < best_hpwl
    best_hpwl[improved_mask] = current_hpwl[improved_mask]

    # constrain_loss = get_reward_based_constraints_loss(p, current_hpwl, best_hpwl, constrain_weight)
    # constrain_loss = get_constraints_loss(p)

    constrain_loss = get_constraints_loss_site_exclusivity_st(p) + get_constraints_loss_centering(p, site_coords_matrix)
    
    # total_energy += hpwl_weight *  hpwl_loss + constrain_weight * constrain_loss
    
    return hpwl_weight * current_hpwl, constrain_weight * constrain_loss

def expected_fpga_placement_xy(J, p_x, p_y):
    hpwl_weight, constrain_weight = 1, 1
    current_hpwl = get_hpwl_loss_xy_simple(J, p_x, p_y)
    constrain_loss = get_constraints_loss_xy(p_x, p_y)
    return hpwl_weight * current_hpwl, constrain_weight * constrain_loss


# infer placment
def infer_site_coordinates(expected_site_index, area_width):
    x_coords = expected_site_index % area_width
    y_coords = expected_site_index // area_width
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords.float()

def infer_placements(J, p, area_width, site_coords_matrix):
    site_indices = torch.argmax(p, dim=2)
    site_coords = get_site_coordinates_from_index(site_indices, area_width)
    result = get_hpwl_loss(J, p, site_coords_matrix)
    return site_coords, result

def infer_placements_xy(J, p_x, p_y):
    site_coords = get_site_coordinates_from_px_py(p_x, p_y)
    result = get_hpwl_loss_xy_simple(J, p_x, p_y)
    return site_coords, result


class OptimizationProblem:
    """
    Optimization problem class
    """
    def __init__(
            self, 
            num_nodes, num_interactions,
            coupling_matrix, 
            problem_type, 
            imbalance_weight=5.0, 
            epsilon=0.03,
            q=2,
            io_site_connect_matrix=None,
            hyperedge=None,
            fpga_wrapper=None,
            discretization=False,
            customize_expected_func=None,
            customize_grad_func=None,
            customize_infer_func=None
        ) -> None:
        self.num_nodes = num_nodes
        self.num_interactions = num_interactions
        self.coupling_matrix = coupling_matrix
        self.problem_type = problem_type
        self.imbalance_weight = imbalance_weight
        self.epsilon = epsilon
        self.q = q
        self.hyperedge = hyperedge

        self.fpga_wrapper = fpga_wrapper # rapidwright design object
        self.io_site_connect_matrix = io_site_connect_matrix
        self.site_coords_matrix = None

        self.discretization = discretization
        self.constant = 0
        self.customize_expected_func = customize_expected_func
        self.customize_grad_func = customize_grad_func
        self.customize_infer_func = customize_infer_func

        self.net_sites_tensor = fpga_wrapper.net_to_slice_sites_tensor
        pass

    def extra_preparation(self, num_trials=1, sparse=False):
        if self.problem_type == 'maxcut':
            self.c = 1 / torch.abs(self.coupling_matrix).sum(1)
        if self.problem_type == 'bmincut':
            self.w2 = self.coupling_matrix.square().sum()
            self.imbalance_weight = self.imbalance_weight * self.w2 / (self.num_nodes**2)
        if self.problem_type == 'hyperbmincut':
            # TODO improved weight here
            total_weight = self.coupling_matrix.sum()
            num_edges = (self.coupling_matrix > 0).float().sum()
            avg_weight = total_weight / num_edges if num_edges > 0 else 1.0
            self.imbalance_weight = 0.5
            # self.imbalance_weight = 0.5 * (avg_weight ** 0.5) * (self.num_nodes ** 0.25)
            self.U_max = int((1 + self.epsilon) * self.num_nodes / self.q)
            self.L_min = int((1 - self.epsilon) * self.num_nodes / self.q)
            print(f"Imbalance weight: {self.imbalance_weight}, U_max: {self.U_max}, L_min: {self.L_min}")
        if self.problem_type == 'modularity':
            self.d = self.coupling_matrix.sum(1).reshape([1, self.num_nodes, 1])
            self.m = self.coupling_matrix.sum() / 2
        if self.problem_type == 'vertexcover':
            degrees = self.coupling_matrix.sum(1)
            self.coupling_matrix *= self.imbalance_weight / 2
            self.coupling_matrix[range(self.num_nodes), range(self.num_nodes)] = \
                1 - degrees * self.imbalance_weight
            self.constant = self.num_interactions * self.imbalance_weight
        if self.problem_type == 'maxksat':
            self.coupling_matrix = self.coupling_matrix.repeat(1, num_trials, 1, 1)

        if self.problem_type == 'fpga_placement':
            self.bbox_length = self.fpga_wrapper.bbox['area_length']
            self.site_coords_matrix = get_site_coords_all(self.fpga_wrapper.num_of_sites, self.bbox_length)
            self.best_hpwl = torch.full((10,), float('inf'))
            return
        
        if sparse:
            self.coupling_matrix = self.coupling_matrix.to_sparse()

    def set_up_couplings_status(self, dev, dtype):
        self.coupling_matrix = self.coupling_matrix.to(dtype).to(dev)
    
    def expectation(self, p):
        if self.problem_type == 'maxcut':
            return -expected_cut(self.coupling_matrix/2, p)
        elif self.problem_type == 'bmincut':
            return expected_bmincut(self.coupling_matrix, p) + \
                self.imbalance_weight * imbalance_penalty(p)
        elif self.problem_type == 'hyperbmincut':
            # print(f"expected_hyperbmincut: {expected_hyperbmincut(self.coupling_matrix, p)}")
            # print(f"Balance loss: {self.imbalance_weight * balance_constrain_1(p, self.U_max, self.L_min)}")
            # factor = (step_max - step )/ step_max
            # rev_factor = ( step )/ step_max
            expect_loss = expected_hyperbmincut(self.coupling_matrix, p, self.hyperedge)
            balance_loss = self.imbalance_weight * balance_constrain(self.coupling_matrix, p, self.U_max, self.L_min)
            return expect_loss, balance_loss
        elif self.problem_type == 'modularity':
            return -expected_inner_weight(self.coupling_matrix, p) + \
                expected_inner_weight_configmodel(self.coupling_matrix, p)
        elif self.problem_type == 'vertexcover':
            return expected_qubo(self.coupling_matrix, p)
        elif self.problem_type == 'maxksat':
            return expected_maxksat(self.coupling_matrix, p.unsqueeze(0))
        elif self.problem_type == 'fpga_placement':
            return expected_fpga_placement_xy(self.coupling_matrix, p_x=p[0], p_y=p[1])
            # return expected_fpga_placement(self.coupling_matrix, p, self.io_site_connect_matrix, self.site_coords_matrix, self.net_sites_tensor, self.best_hpwl)
        
        elif self.problem_type == 'customize':
            return self.customize_expected_func(self.coupling_matrix, p)
    
    def manual_grad(self, p):
        if self.problem_type == 'maxcut':
            return manual_grad_maxcut(self.c * self.coupling_matrix, p, self.discretization)
        elif self.problem_type == 'bmincut':
            return manual_grad_bmincut(self.coupling_matrix, p, self.imbalance_weight)
        elif self.problem_type == 'hyperbmincut':
            return manual_grad_hyperbmincut(self.coupling_matrix, p, self.U_max, self.L_min, self.imbalance_weight)
        elif self.problem_type == 'modularity':
            return manual_grad_modularity(
                self.coupling_matrix, p, self.m, self.d.expand(p.shape)
            )
        elif self.problem_type == 'vertexcover':
            return manual_grad_qubo(self.coupling_matrix, p)
        elif self.problem_type == 'maxksat':
            return manual_grad_maxksat(self.coupling_matrix, p.unsqueeze(0)).squeeze(0)
        
        elif self.problem_type == 'fpga_placement':
            return
        
        elif self.problem_type == 'customize':
            return self.customize_grad_func(self.coupling_matrix, p)
            
    
    def inference_value(self, p):
        if (self.problem_type != 'fpga_placement'):
            p = torch.vstack([pi for pi in p if torch.isnan(pi).sum() == 0])
        if self.problem_type == 'maxcut':
            config, result = infer_maxcut(self.coupling_matrix, p)
        elif self.problem_type == 'bmincut':
            config, result = infer_bmincut(self.coupling_matrix, p)
        elif self.problem_type == 'hyperbmincut':
            config, result = infer_hyperbmincut(self.coupling_matrix, p)
        elif self.problem_type == 'vertexcover':
            config, result = infer_qubo(self.coupling_matrix, p)
        elif self.problem_type == 'maxksat':
            config, result = infer_maxksat(self.coupling_matrix, p.unsqueeze(0))

        elif self.problem_type == 'fpga_placement':
            config, result = infer_placements_xy(self.coupling_matrix, p_x=p[0], p_y=p[1])
            # config, result = infer_placements(self.coupling_matrix, p, self.bbox_length, self.site_coords_matrix)

        elif self.problem_type == 'customize':
            return self.customize_infer_func(self.coupling_matrix, p)
        result += self.constant
        return config, result