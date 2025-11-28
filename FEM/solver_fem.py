import torch
from .problem import *
from math import log
from utils import *

def entropy_q(p):
    """
    p is the probabilities for each group, shape [batch, N, q], with q denoting the number of groups
    return - \sum_{i=1}^N sum_{t=1}^q p(t)*\log p(t)
    """
    return - (p*torch.log(p)).sum(2).sum(1)

def entropy_grad_q(p):
    return -p * (torch.log(p) - (p*torch.log(p)).sum(2,keepdim=True).expand(p.shape))

def entropy_binary(p):
    return - ((p*torch.log(p)) + (1-p)*torch.log(1-p)).sum(1)

def entropy_grad_binary(p):
    grad = - (p * (1-p) * (p.log() - (1-p).log()))
    return grad


class Solver:
    def __init__(
            self, 
            problem, num_trials, num_steps, betamin=0.01, betamax=0.5, 
            anneal='inverse', optimizer='adam', learning_rate=0.1, dev='cpu', 
            dtype=torch.float32, seed=1, q=2, manual_grad=False, 
            h_factor=0.01, sparse=False, drawer = None
        ):
        self.dtype = dtype
        self.dev = dev
        if anneal == 'lin':
            betas = torch.linspace(betamin, betamax, num_steps)
        elif anneal == 'exp':
            betas = torch.exp(torch.linspace(log(betamin), log(betamax),num_steps))
        elif anneal == 'inverse':
            betas = 1 / torch.linspace(betamax, betamin, num_steps)
        self.betas = betas.to(self.dtype).to(self.dev) 
        self.num_trials = num_trials
        self.seed = seed
        self.q = q
        self.manual_grad = manual_grad
        self.h_factor = h_factor
        self.problem = problem
        self.problem.set_up_couplings_status(dev, dtype)
        self.problem.extra_preparation(num_trials, sparse)
        self.binary = True if self.problem.problem_type in ['maxcut', 'vertexcover'] else False
        if self.binary:
            assert self.q == 2
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.drawer = drawer

        self.length = self.problem.fpga_wrapper.bbox['area_length']

    def initialize(self):
        torch.manual_seed(self.seed)
        if self.binary:
            h = self.h_factor * torch.randn(
                [self.num_trials, self.problem.num_nodes], 
                device=self.dev, dtype=self.dtype
            )
        else:
# **********************************   Start   *********************************** #
            if self.problem.problem_type == 'fpga_placement':
                # num_trials = self.num_trials
                # num_nodes = self.problem.num_nodes
                # center_site_idx = self.q // 2
                # h = torch.zeros([num_trials, num_nodes, self.q], device=self.dev, dtype=self.dtype)
                # center_strength = 2.0
                # peripheral_strength = 1.0
                
                # h[:, :, center_site_idx] = center_strength
                # peripheral_mask = torch.ones(self.q, dtype=torch.bool, device=self.dev)
                # peripheral_mask[center_site_idx] = False
                # h[:, :, peripheral_mask] = peripheral_strength
                # h += self.h_factor * 0.1 * torch.randn_like(h)

                # Initialize separate potentials for X and Y coordinates in FPGA placement
                # Each coordinate dimension gets its own probability distribution over site positions
                h_x = self.h_factor * torch.randn(
                    [self.num_trials, self.problem.num_nodes,  self.length], 
                    device=self.dev, dtype=self.dtype
                )

                h_y = self.h_factor * torch.randn(
                    [self.num_trials, self.problem.num_nodes,  self.length], 
                    device=self.dev, dtype=self.dtype
                )

                h_x.requires_grad=True
                h_y.requires_grad=True
                return h_x, h_y
# **********************************    END    *********************************** #
            else:
                h = self.h_factor * torch.randn(
                    [self.num_trials, self.problem.num_nodes, self.q], 
                    device=self.dev, dtype=self.dtype
                )

            # h = self.h_factor * torch.randn(
            #     [self.num_trials, self.problem.num_nodes, self.q], 
            #     device=self.dev, dtype=self.dtype
            # )

            # # 改进的多分类初始化
            # n_trials = self.num_trials
            # n_nodes = self.problem.num_nodes
            # q = self.q
            
            # # 方法1：为每个节点随机选择一个主簇并增强
            # h = torch.randn(n_trials, n_nodes, q, device=self.dev, dtype=self.dtype) * 0.2
            
            # # 为每个节点随机选择主簇
            # main_clusters = torch.randint(0, q, (n_trials, n_nodes), device=self.dev)
            
            # # 使用scatter_高效地增强主簇
            # batch_indices = torch.arange(n_trials, device=self.dev).unsqueeze(1).expand(-1, n_nodes)
            # node_indices = torch.arange(n_nodes, device=self.dev).unsqueeze(0).expand(n_trials, -1)
            
            # h[batch_indices, node_indices, main_clusters] += 3.0
            
            # # 乘以h_factor保持原有缩放
            # h = self.h_factor * h

        if self.manual_grad:
            h.requires_grad=False
        else:
            h.requires_grad=True
        return h
    
    def set_up_optimizer(self, params):
        if self.optimizer == 'adam':
            self.opt = torch.optim.Adam([params], lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.opt = torch.optim.RMSprop(
                [params], lr=self.learning_rate, alpha=0.98, eps=1e-08, 
                weight_decay=0.01, momentum=0.91, centered=False
            )
        else:
            raise ValueError("Unkown optimizer, valid choices are ['adam', 'rmsprop'].")

    def set_up_optimizer_placement(self, params):
        if self.optimizer == 'adam':
            self.opt = torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.opt = torch.optim.RMSprop(
                params, lr=self.learning_rate, alpha=0.98, eps=1e-08, 
                weight_decay=0.01, momentum=0.91, centered=False
            )
    
    def iterate(self):

        history = {
            'hpwl': [],
        }

        record_steps = [0, 250, 500, 750, 999]

        h = self.initialize()
        self.set_up_optimizer(h)
        step_max = len(self.betas)
        for step in range(step_max):
            # print(h[:, :, self.q // 2])
            p = torch.sigmoid(h) if self.binary else torch.softmax(h, dim=2)
            self.opt.zero_grad()
            if self.binary:
                entropy_grad = entropy_grad_binary
                entropy = entropy_binary
            else:
                entropy_grad = entropy_grad_q
                entropy = entropy_q
            if self.manual_grad:
                h.grad = self.problem.manual_grad(p) - \
                    entropy_grad(p) / self.betas[step]
            else:
                # cut_loss, balance_loss = self.problem.expectation(p)
                # free_energy = cut_loss + balance_loss - \
                #     entropy(p) / self.betas[step]

                # h_grad_cut = torch.autograd.grad(
                #     cut_loss.sum(), h, retain_graph=True, allow_unused=True
                # )[0]
                # h_grad_balance = torch.autograd.grad(
                #     balance_loss.sum(), h, retain_graph=True, allow_unused=True
                # )[0]

                
                # loss = self.problem.expectation(p)
                # free_energy = loss - \
                #     entropy(p) / self.betas[step]

                # free_energy.backward(gradient=torch.ones_like(free_energy)) # minimize free energy

                hpwl_loss, constrain_loss = self.problem.expectation(p)

                free_energy = hpwl_loss + constrain_loss - \
                    entropy(p) / self.betas[step]

                # free_energy = constrain_loss - \
                #     entropy(p) / self.betas[step]
                
                h_grad_hpwl = torch.autograd.grad(
                    hpwl_loss, h, grad_outputs=torch.ones_like(hpwl_loss), retain_graph=True, allow_unused=True
                )

                h_grad_constrain = torch.autograd.grad(
                    constrain_loss, h, grad_outputs=torch.ones_like(constrain_loss), retain_graph=True, allow_unused=True
                )
                
                h_grad_hpwl_norm = []
                h_grad_constrain_norm = []

                for i in range(len(h_grad_hpwl)):
                    h_grad_hpwl_norm.append(torch.norm(h_grad_hpwl[i]).item())
                    h_grad_constrain_norm.append(torch.norm(h_grad_constrain[i]).item())

                if step in record_steps:
                    print(f"INFO, step {step}, hpwl Loss = {[f'{x:.2f}' for x in hpwl_loss.tolist()]}, \n \
                        hpwl grad norm = {h_grad_hpwl_norm}, \n \
                        constrain loss = {[f'{x:.2f}' for x in constrain_loss.tolist()]}, \n \
                        constrain grad norm = {h_grad_constrain_norm} ")
                
                free_energy.backward(gradient=torch.ones_like(free_energy)) # minimize free energy
                # print(f"Step {step}:")
                # print(f"  cut_loss: {hpwl_loss.mean().item():.6f}")
                # if h.grad is not None:
                #     grad_norm = torch.norm(h.grad).item()
                #     grad_mean = h.grad.mean().item()
                #     grad_std = h.grad.std().item()
                #     print(f"  h.grad - norm: {grad_norm:.6f}, mean: {grad_mean:.6f}, std: {grad_std:.6f}")
                    
                #     # 检查梯度是否太小（梯度消失）
                #     if grad_norm < 1e-8:
                #         print("  ⚠️ 警告: 梯度范数太小，可能梯度消失")
                    
                #     # 检查梯度是否太大（梯度爆炸）
                #     if grad_norm > 1000:
                #         print("  ⚠️ 警告: 梯度范数太大，可能梯度爆炸")

                
                # 记录梯度信息
                # cut_grad_norm = torch.norm(h_grad_cut).item() if h_grad_cut is not None else 0.0
                # balance_grad_norm = torch.norm(h_grad_balance).item() if h_grad_balance is not None else 0.0
                
                # assignments = torch.argmax(p, dim=2)
                # group_assignment = assignments[0].cpu().numpy()  # [n_nodes]
                # kahypar_cut_value = evaluate_kahypar_cut_value_simple(group_assignment, self.problem.hyperedge)

                # print(f"step {step} cut loss: {cut_loss.mean().item():.8f} balance_loss: {balance_loss.mean().item():.8f}")
                # print(f"            cut_grad: {cut_grad_norm:.8f}          balance_grad: {balance_grad_norm:.8f}")
                # print(f"            kahypar_cut: {kahypar_cut_value:.8f}")

                # probabilities = torch.softmax(p, dim=2)
                # assignments = torch.argmax(probabilities, dim=2)
                # one_hot = torch.nn.functional.one_hot(assignments, num_classes=n_clusters)
                # S_k = one_hot.sum(dim=1).float()

                # if cut_grad_norm < 1e-8:
                #     print("⚠️  警告: cut_loss梯度接近0, 可能梯度消失")
                # if balance_grad_norm < 1e-8:
                #     print("⚠️  警告: balance_loss梯度接近0, 可能梯度消失")

                # history['free_energy'].append(free_energy[0])
                # history['step'].append(step)

            self.opt.step()
            # print(step)
            if step in record_steps:
                # print(" Recording placement at step ", step)
                sites_coords = get_hard_placements_from_index(p, self.problem.site_coords_matrix)
                self.drawer.add_placement(sites_coords[0], step)

        self.drawer.draw_multi_step_placement()

        return p

# **********************************   Start   *********************************** #
    def iterate_placement(self):

        """
        FEM optimization loop for FPGA placement with separate X/Y coordinates.
        Uses free energy minimization with HPWL and constraint losses.
        """

        record_steps = [0, 250, 500, 750, 999]

        h_x, h_y = self.initialize()
        self.set_up_optimizer_placement([h_x, h_y])
        step_max = len(self.betas)
        for step in range(step_max):
            p_x = torch.softmax(h_x, dim=2)
            p_y = torch.softmax(h_y, dim=2)
            
            self.opt.zero_grad()
            entropy = entropy_q
            hpwl_loss, constrain_loss = self.problem.expectation([p_x, p_y])

            free_energy = hpwl_loss + constrain_loss - \
                (entropy(p_x) + entropy(p_y)) / self.betas[step]
            
            h_grad_hpwl = torch.autograd.grad(
                hpwl_loss, h_x, grad_outputs=torch.ones_like(hpwl_loss), retain_graph=True, allow_unused=True
            )

            h_grad_constrain = torch.autograd.grad(
                constrain_loss, h_x, grad_outputs=torch.ones_like(constrain_loss), retain_graph=True, allow_unused=True
            )
            
            h_grad_hpwl_norm = []
            h_grad_constrain_norm = []

            for i in range(len(h_grad_hpwl)):
                h_grad_hpwl_norm.append(torch.norm(h_grad_hpwl[i]).item())
                h_grad_constrain_norm.append(torch.norm(h_grad_constrain[i]).item())

            if step in record_steps:
                print(f"INFO, step {step}, hpwl Loss = {[f'{x:.2f}' for x in hpwl_loss.tolist()]}, \n \
                    hpwl grad norm = {h_grad_hpwl_norm}, \n \
                    constrain loss = {[f'{x:.2f}' for x in constrain_loss.tolist()]}, \n \
                    constrain grad norm = {h_grad_constrain_norm} ")
            
            free_energy.backward(gradient=torch.ones_like(free_energy)) # minimize free energy
            self.opt.step()
            if step in record_steps:
                sites_coords = get_site_coordinates_from_px_py(p_x, p_y)
                self.drawer.add_placement(sites_coords[0], step)

        self.drawer.draw_multi_step_placement()

        return p_x, p_y
# **********************************   Start   *********************************** #

    def solve(self):

# **********************************   Start   *********************************** # 
        if self.problem.problem_type == 'fpga_placement':
            marginal_x, marginal_y = self.iterate_placement()
            configs, results = self.problem.inference_value([marginal_x, marginal_y])
            return configs, results
# **********************************    END    *********************************** # 

        marginal = self.iterate()
        configs, results = self.problem.inference_value(marginal)
        return configs, results