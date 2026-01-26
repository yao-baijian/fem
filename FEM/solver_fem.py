import torch
from .problem import *
from math import log
from .utils import *

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
                self.length = self.problem.bbox_length
                if self.problem.fpga_wrapper.with_io():
                    h_logic = self.h_factor * torch.randn(
                        [self.num_trials, self.problem.num_nodes, self.q], 
                        device=self.dev, dtype=self.dtype
                    )
                    
                    h_io = self.h_factor * torch.randn(
                        [self.num_trials, self.problem.fpga_wrapper.fixed_insts_num, self.problem.fpga_wrapper.fixed_insts_num], 
                        device=self.dev, dtype=self.dtype
                    )

                    h_logic.requires_grad=True
                    h_io.requires_grad=True

                    return h_logic, h_io
                    
                h = self.h_factor * torch.randn(
                    [self.num_trials, self.problem.num_nodes, self.q], 
                    device=self.dev, dtype=self.dtype
                )

                h.requires_grad=True

                return h
# **********************************    END    *********************************** #
            else:
                h = self.h_factor * torch.randn(
                    [self.num_trials, self.problem.num_nodes, self.q], 
                    device=self.dev, dtype=self.dtype
                )
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
        h = self.initialize()
        self.set_up_optimizer(h)
        step_max = len(self.betas)
        for step in range(step_max):
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
                free_energy = self.problem.expectation(p, step) \
                    - entropy(p) / self.betas[step]
                free_energy.backward(gradient=torch.ones_like(free_energy)) # minimize free energy
            self.opt.step()
        return p

# **********************************   Start   *********************************** #
    def iterate_placement(self):
        h_logic, h_io = self.initialize()
        self.set_up_optimizer_placement([h_logic, h_io])
        step_max = len(self.betas)
        for step in range(step_max):
            p_logic = torch.softmax(h_logic, dim=2)
            p_io = torch.softmax(h_io, dim=2)
            self.opt.zero_grad()
            entropy = entropy_q
            loss = self.problem.expectation([p_logic, p_io])
            free_energy = loss - \
                (entropy(p_logic) + entropy(p_io)) / self.betas[step]
            
            # h_grad_hpwl = torch.autograd.grad(
            #     hpwl_loss, h_x, grad_outputs=torch.ones_like(hpwl_loss), retain_graph=True, allow_unused=True
            # )

            # h_grad_constrain = torch.autograd.grad(
            #     constrain_loss, h_x, grad_outputs=torch.ones_like(constrain_loss), retain_graph=True, allow_unused=True
            # )
            free_energy.backward(gradient=torch.ones_like(free_energy)) # minimize free energy
            self.opt.step()
            # sites_coords = get_site_coordinates_from_px_py(p_x, p_y)
            # sites_coords = get_hard_placements_from_index(p, self.problem.site_coords_matrix)
            # self.drawer.add_placement(sites_coords[0], step)

        # self.drawer.draw_multi_step_placement()
        return [p_logic, p_io]
# **********************************   Start   *********************************** #

    def solve(self):

# **********************************   Start   *********************************** # 
        if self.problem.problem_type == 'fpga_placement':
            # marginal_x, marginal_y = self.iterate_placement()
            # configs, results = self.problem.inference_value([marginal_x, marginal_y])
            if self.problem.fpga_wrapper.with_io():
                marginal = self.iterate_placement()
                configs, results = self.problem.inference_value(marginal)
            else:
                marginal = self.iterate()
                configs, results = self.problem.inference_value(marginal)
            return configs, results
# **********************************    END    *********************************** # 

        marginal = self.iterate()
        configs, results = self.problem.inference_value(marginal)
        return configs, results