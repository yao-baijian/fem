import torch
import torch.nn.functional as Func

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


def balance_constrain_1(p, U_max, L_min):
    # inbalance with constrain (relu version)
    S_k = p.sum(dim=0)
    balance_loss = torch.sum(torch.relu(S_k - U_max)) + torch.sum(torch.relu(L_min - S_k))
    return balance_loss


def balance_constrain_2(p, U_max, L_min, q):
    # inbalance with constrain (softplus version)
    group_sizes = p.sum(dim=1)  # [batch, q]
    balance_loss = 0
    for k in range(q):
        size_k = group_sizes[:, k]
        upper_violation = torch.nn.functional.softplus(size_k - U_max)
        lower_violation = torch.nn.functional.softplus(L_min - size_k)
        balance_loss += (upper_violation + lower_violation).mean()
    return balance_loss


    # optimizer.zero_grad()
    # loss, p = balanced_mincut_loss_autodiff(h, W, lambda_penalty, epsilon, q, beta)
    # loss.backward()
    # optimizer.step()
    # return loss, p

def infer_hyperbmincut(J, p):
    config = Func.one_hot(p.view(-1,J.shape[0],p.shape[-1]).argmax(dim=2), num_classes=p.shape[-1]).to(J.dtype)
    return config, expected_bmincut(J, config) / 2

def expected_hyperbmincut(J,p):
    return ((J @ p) * (1-p)).sum((1, 2))


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
            self.imbalance_weight = 0.5 * (avg_weight ** 0.5) * (self.num_nodes ** 0.25)

            self.U_max = (1 + self.epsilon) * self.num_nodes / self.q
            self.L_min = (1 - self.epsilon) * self.num_nodes / self.q

            print(f"Set imbalance weight to {self.imbalance_weight}, U_max: {self.U_max}, L_min: {self.L_min}")

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
            return expected_hyperbmincut(self.coupling_matrix, p) + \
                self.imbalance_weight * balance_constrain_1(p, self.U_max, self.L_min)
        

        elif self.problem_type == 'modularity':
            return -expected_inner_weight(self.coupling_matrix, p) + \
                expected_inner_weight_configmodel(self.coupling_matrix, p)
        elif self.problem_type == 'vertexcover':
            return expected_qubo(self.coupling_matrix, p)
        elif self.problem_type == 'maxksat':
            return expected_maxksat(self.coupling_matrix, p.unsqueeze(0))
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
        elif self.problem_type == 'customize':
            return self.customize_infer_func(self.coupling_matrix, p)
        result += self.constant
        return config, result