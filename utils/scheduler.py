import math


class OneCycleScheduler:
    """
    One Cycle learning rate scheduler for super-convergence.
    This scheduler increases the learning rate from a lower bound (base_lr) up to max_lr,
    then decreases it down to a very low final_lr over a specified number of steps (cycle_length).
    Supports either cosine or linear annealing strategy for both the upward and downward phases.
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
        # Store optimizer reference
        self.optimizer = optimizer
        # Maximum learning rate at the peak of the cycle
        self.max_lr = max_lr
        # Base learning rate for the start of the cycle (default 1/10th of max_lr if not provided)
        self.base_lr = base_lr or max_lr / 10.0
        # Final learning rate at the end of the cycle (base_lr divided by final_div_factor)
        self.final_lr = self.base_lr / final_div_factor
        # Fraction of cycle spent increasing LR (the remainder is spent decreasing)
        self.pct_start = pct_start
        # Total number of steps in the cycle (we add 1 for inclusive counting)
        self.cycle_length = float(cycle_length) + 1
        # Number of steps for the raising phase
        self.up_steps = cycle_length * pct_start
        # Number of steps for the lowering phase
        self.down_steps = cycle_length - self.up_steps
        # Counter to track current step (starts at 1)
        self.step_num = 1
        # Choose annealing strategy: 'cos' for cosine, 'linear' for linear
        self.anneal_strategy = anneal_strategy

        # Validate anneal_strategy value
        if self.anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                f"Invalid anneal_strategy: {self.anneal_strategy}. "
                "Choose either 'cos' or 'linear'."
            )

        # Initialize all parameter groups in optimizer with base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr

    def get_lr(self):
        """
        Compute the new learning rate for the current step in the cycle.
        Returns either a linearly or cosinely annealed LR depending on the phase.
        """
        # Check if we are in the upward phase
        linear = (self.anneal_strategy == "linear")
        if self.step_num <= self.up_steps:
            # Percentage of completion in the upward phase
            pct = self.step_num / float(self.up_steps)
            if linear:
                # Linear increase: base_lr -> max_lr
                return self.base_lr + pct * (self.max_lr - self.base_lr)
            else:
                # Cosine increase: smoother curve
                return (
                    self.base_lr
                    + (self.max_lr - self.base_lr) * (1 - math.cos(math.pi * pct)) / 2
                )

        # Falling phase: compute percentage past the peak
        pct_down = (self.step_num - self.up_steps) / float(self.down_steps)
        if linear:
            # Linear decrease: max_lr -> final_lr
            return self.max_lr - pct_down * (self.max_lr - self.final_lr)
        else:
            # Cosine decrease: smooth transition to final_lr
            return (
                self.final_lr
                + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * pct_down)) / 2
            )

    def step(self):
        """
        Advance one step in the cycle and update the optimizer's learning rate.
        Caps the step number at cycle_length so it does not exceed the total cycle.
        """
        # Increment step counter (but do not exceed cycle_length)
        self.step_num = min(self.step_num + 1, self.cycle_length)
        # Compute new learning rate
        lr = self.get_lr()
        # Update all parameter groups in optimizer
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def reset(self):
        """
        Reset the cycle back to step 1 and restore base_lr to optimizer.
        Useful for starting a new cycle without creating a new scheduler.
        """
        self.step_num = 1
        # Reset optimizer learning rate to base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr
