from typing import Callable, Iterable, Tuple, Union, Any, NamedTuple, List
import math

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn.parameter import Parameter

class ParamGroup(NamedTuple):
    params: List[Parameter]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    correct_bias: bool

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        """
        The keys are:
        params -> list
        lr -> float
        betas -> tuple
        eps -> float
        weight_decay -> float
        correct_bias -> bool
        """
        group: ParamGroup
        for group in map(lambda d: ParamGroup(**d), self.param_groups):
            p: Parameter
            for p in group.params:
                if p.grad is None:
                    continue
                # use .data to access the raw tensor, bypassing Autograd tracking.
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                # self.state is a defaultdict, where the keys are Parameters, and the values are dictionaries
                # The first time we call `step`, the state dictionary will be empty, and we initialize parameters
                # using the code below.
                state: dict[str, Union[int, Tensor]] = self.state[p]
                assert isinstance(state, dict)
                if not state:
                    state['step'] = 0
                    # Initialize first moment (mean) vector m_t
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Initialize second moment (variance) vector v_t
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Update step count
                state['step'] += 1

                # Get first and second moment variables
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Update biased first moment estimate
                exp_avg.mul_(group.betas[0]).add_(grad, alpha=(1 - group.betas[0]))

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(group.betas[1]).addcmul_(grad, grad, value=(1 - group.betas[1]))

                # Bias correction
                if group.correct_bias:
                    bias_correction1 = 1 - group.betas[0] ** state['step']
                    bias_correction2 = 1 - group.betas[1] ** state['step']
                    corrected_exp_avg = exp_avg / bias_correction1
                    corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                else:
                    corrected_exp_avg = exp_avg
                    corrected_exp_avg_sq = exp_avg_sq

                # Update parameter
                denom = corrected_exp_avg_sq.sqrt().add_(group.eps)
                p.data.addcdiv_(corrected_exp_avg, denom, value=-group.lr)

                # Apply weight decay
                if group.weight_decay > 0:
                    p.data.add_(p.data, alpha=-group.lr * group.weight_decay)

        return loss

