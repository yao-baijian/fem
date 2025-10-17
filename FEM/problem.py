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


# def expected_maxcut(J, p):
#     return -torch.bmm(
#         (p @ J).reshape(-1, 1, J.shape[1]),
#         p.reshape(-1, p.shape[1], 1)
#     ).reshape(-1)


# def manual_grad_maxcut(J, p):
#     temp = 1 - 2 * p
#     tp = -J @ temp
#     h_grad = (tp  - (tp * p).sum(2,keepdim=True).expand(tp.shape))*p
#     return h_grad
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




def expected_fpga_placement(clause_batch, p):
    return
    
def infer_fpga_placement(clause_batch, p):
    return






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
            hyperedge=None,
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

        self.discretization = discretization
        self.constant = 0
        self.customize_expected_func = customize_expected_func
        self.customize_grad_func = customize_grad_func
        self.customize_infer_func = customize_infer_func
        pass

    def extra_preparation(self, num_trials=1, sparse=False):
        if self.problem_type == 'maxcut':
            self.c = 1 / torch.abs(self.coupling_matrix).sum(1)
        if self.problem_type == 'bmincut':
            self.w2 = self.coupling_matrix.square().sum()
            self.imbalance_weight = self.imbalance_weight * self.w2 / (self.num_nodes**2)


        if self.problem_type == 'hyperbmincut':


            # self.w2 = self.coupling_matrix.square().sum()
            # self.imbalance_weight = self.imbalance_weight * self.w2 / (self.num_nodes**2)

            # TODO improved weight here

            total_weight = self.coupling_matrix.sum()
            num_edges = (self.coupling_matrix > 0).float().sum()
            avg_weight = total_weight / num_edges if num_edges > 0 else 1.0
            self.imbalance_weight = 0.5
            # self.imbalance_weight = 0.5 * (avg_weight ** 0.5) * (self.num_nodes ** 0.25)

            # self.w2 = self.coupling_matrix.sum()
            # self.imbalance_weight = self.imbalance_weight * self.w2 / (self.num_nodes)

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
            # self.coupling_matrix = self.coupling_matrix.repeat(1, num_trials, 1, 1)

        if sparse:
            self.coupling_matrix = self.coupling_matrix.to_sparse()

    def set_up_couplings_status(self, dev, dtype):
        self.coupling_matrix = self.coupling_matrix.to(dtype).to(dev)
    
    def expectation(self, p):
        if self.problem_type == 'maxcut':
            return -expected_cut(self.coupling_matrix/2, p)
        elif self.problem_type == 'bmincut':

            # print(f"Loss: {self.imbalance_weight * imbalance_penalty(p)}")

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
                # self.imbalance_weight * balance_constrain(self.coupling_matrix, p, self.U_max, self.L_min)
                

        elif self.problem_type == 'modularity':
            return -expected_inner_weight(self.coupling_matrix, p) + \
                expected_inner_weight_configmodel(self.coupling_matrix, p)
        elif self.problem_type == 'vertexcover':
            return expected_qubo(self.coupling_matrix, p)
        elif self.problem_type == 'maxksat':
            return expected_maxksat(self.coupling_matrix, p.unsqueeze(0))
        
        elif self.problem_type == 'fpga_placement':
            return expected_fpga_placement()
        
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
            return manual_grad_fpga_placement(self.coupling_matrix, p.unsqueeze(0)).squeeze(0)

        elif self.problem_type == 'customize':
            return self.customize_grad_func(self.coupling_matrix, p)
            
    
    def inference_value(self, p):
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
            config, result = infer_fpga_placement(self.coupling_matrix, p.unsqueeze(0))

        elif self.problem_type == 'customize':
            return self.customize_infer_func(self.coupling_matrix, p)
        result += self.constant
        return config, result