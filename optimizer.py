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
                param_state: dict[str, Union[int, Tensor]] = self.state[p]
                assert isinstance(param_state, dict)
                if not param_state:
                    param_state['step'] = 0
                    # Initialize first moment (mean) vector m_t
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    # Initialize second moment (variance) vector v_t
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                param_state['step'] += 1

                # Get first and second moment variables
                exp_avg: Tensor = param_state['exp_avg']
                exp_avg_sq: Tensor = param_state['exp_avg_sq']

                # exp_avg = exp_avg * beta1 + grad * (1 - beta1)
                exp_avg.mul_(group.betas[0]).add_(grad, alpha=(1 - group.betas[0]))

                # addcmul multiplies together grad * grad
                # exp_avg_sq = exp_avg_sq * beta2 + (grad^2) * (1 - beta2)
                exp_avg_sq.mul_(group.betas[1]).addcmul_(grad, grad, value=(1 - group.betas[1]))

                # Bias correction. Momentum / Variance are initially biased toward 0, so we divide by
                # (1 - beta^t). When t is small, this increases momentum/variance. When t is large,
                # we are dividing by 1 (no-op)
                bias_correction1 = 1
                bias_correction2 = 1
                if group.correct_bias:
                    bias_correction1 -= group.betas[0] ** param_state['step']
                    bias_correction2 -= group.betas[1] ** param_state['step']
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                denom = corrected_exp_avg_sq.sqrt().add_(group.eps)
                # add the first moment divided by the sqrt of second moment, scaled by the LR
                p.data.addcdiv_(corrected_exp_avg, denom, value=-group.lr)

                # Apply weight decay
                if group.weight_decay > 0:
                    p.data.add_(p.data, alpha=-group.lr * group.weight_decay)

        return loss

