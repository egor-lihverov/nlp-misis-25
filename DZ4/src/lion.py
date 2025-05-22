import torch
import torch.optim as optim

class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                momentum = state['momentum']
                update_direction = (1 - beta1) * grad + beta1 * momentum
                signed_update = torch.sign(update_direction)
                next_momentum = (1 - beta2) * grad + beta2 * momentum

                if weight_decay > 0:
                    update_val = signed_update + p.data * weight_decay
                else:
                    update_val = signed_update

                p.data.add_(update_val, alpha=-lr)
                state['momentum'].copy_(next_momentum)

        return loss