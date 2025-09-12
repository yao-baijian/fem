import torch
from .problem import OptimizationProblem
from .solver_fem import Solver
from .utils import parse_file


class FEM:
    """
    Interface class of the solver
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def from_file(cls, problem_type, filename, index_start=0, **args):
        num_nodes, num_interactions, couplings = parse_file(
            problem_type, filename, index_start
        )
        fem = cls()
        fem.set_up_problem(
            num_nodes, num_interactions, problem_type, couplings, **args
        )
        return fem
    
    @classmethod
    def from_couplings(cls, problem_type, num_nodes, num_interactions, couplings, **args):
        fem = cls()
        fem.set_up_problem(
            num_nodes, num_interactions, problem_type, couplings, **args
        )
        return fem

    def set_up_problem(
            self, num_nodes, num_interactions, problem_type, coupling_matrix, 
            imbalance_weight=5.0, discretization=False, 
            customize_expected_func=None, customize_grad_func=None,
            customize_infer_func=None
        ):
        supported_types = [
            'maxcut', 'bmincut', 'modularity', 'maxksat', 'vertexcover', 'customize'
        ]
        if problem_type not in supported_types:
            raise ValueError(
                f"Problem type '{problem_type}', current support types are {supported_types}"
            )
        self.problem = OptimizationProblem(
            num_nodes, num_interactions, coupling_matrix, 
            problem_type, imbalance_weight, discretization, 
            customize_expected_func, customize_grad_func, customize_infer_func
        )

    def set_up_solver(
            self, num_trials, num_steps, betamin=0.01, betamax=0.5, 
            anneal='inverse', optimizer='adam', learning_rate=0.1, dev='cuda', 
            dtype=torch.float32, seed=1, q=2, manual_grad=False, 
            h_factor=0.01, sparse=False
        ):
        assert 'problem' in self.__dict__.keys()
        self.solver = Solver(
            self.problem, num_trials, num_steps, betamin, betamax, anneal, 
            optimizer, learning_rate, dev, dtype, seed, q, manual_grad, 
            h_factor, sparse
        )

    def solve(self):
        return self.solver.solve()
    
