import math


class OneCycleScheduler:
    """
    One Cycle learning rate scheduler for super-convergence.
    """

    def __init__(
        self,
        optimizer,
        cycle_length,
        max_lr,
        base_lr=None,
        final_div_factor=1000.0,
        pct_start=0.5,
        anneal_strategy="cos",
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.base_lr = base_lr or max_lr / 10.0
        self.final_lr = self.base_lr / final_div_factor
        self.pct_start = 0.5
        self.cycle_length = float(cycle_length) + 1
        self.up_steps = cycle_length * pct_start
        self.down_steps = cycle_length - self.up_steps
        self.step_num = 1
        self.anneal_strategy = anneal_strategy
        if self.anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                f"Invalid anneal_strategy: {self.anneal_strategy}. "
                "Choose either 'cos' or 'linear'."
            )

        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr

    def get_lr(self):
        linear = self.anneal_strategy == "linear"
        if self.step_num <= self.up_steps:
            pct = self.step_num / float(self.up_steps)
            if linear:
                return self.base_lr + pct * (self.max_lr - self.base_lr)
            else:
                return (
                    self.base_lr
                    + (self.max_lr - self.base_lr) * (1 - math.cos(math.pi * pct)) / 2
                )

        pct_down = (self.step_num - self.up_steps) / float(self.down_steps)
        if linear:
            return self.max_lr - pct_down * (self.max_lr - self.final_lr)
        else:
            return (
                self.final_lr
                + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * pct_down)) / 2
            )

    def step(self):
        self.step_num = min(self.step_num + 1, self.cycle_length)
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def reset(self):
        self.step_num = 1
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr
