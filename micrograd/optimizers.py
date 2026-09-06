# Optimizers
import math


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        raise NotImplementedError("Each optimizer must implement its own step() method.")


# SGD
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum 
        self.velocities = {id(p): 0.0 for p in self.params} if momentum > 0.0 else {}


    def step(self):
        for p in self.params:
            if self.momentum > 0.0:
                # 1. Fetch previous speed from ledger
                v = self.velocities[id(p)]
                # 2. Accumulate: 90% old momentum + current gradient
                v = self.momentum * v + p.grad
                # 3. Store new speed back in ledger
                self.velocities[id(p)] = v
                # 4. Update parameter using velocity instead of raw grad
                p.data -= self.lr * v
            else:
                # Vanilla SGD (no momentum tracking needed)
                p.data -= self.lr * p.grad


# Adam (Adaptive Moment Estimation)
class Adam(Optimizer):
    """Adam optimizer for scalar ``Value`` parameters.

    Adam combines:
    - momentum: an exponential moving average of the gradients; and
    - adaptive scaling: an exponential moving average of squared gradients.

    The state is kept separately for every parameter because each parameter
    can have a different gradient history.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)

        if lr < 0.0:
            raise ValueError("Learning rate must be non-negative.")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Adam betas must be in the interval [0, 1).")
        if eps < 0.0:
            raise ValueError("Adam epsilon must be non-negative.")

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        # Adam needs two persistent scalar buffers per parameter:
        # m = first moment (smoothed gradient / direction)
        # v = second raw moment (smoothed squared gradient / scale)
        self.m = {id(p): 0.0 for p in self.params}
        self.v = {id(p): 0.0 for p in self.params}
        self.step_count = 0

    def step(self):
        self.step_count += 1
        t = self.step_count

        for p in self.params:
            grad = p.grad

            # Exponential moving averages.  The (1 - beta) terms make these
            # moving averages track the scale of the current gradients.
            m = self.beta1 * self.m[id(p)] + (1.0 - self.beta1) * grad
            v = self.beta2 * self.v[id(p)] + (1.0 - self.beta2) * (grad ** 2)
            self.m[id(p)] = m
            self.v[id(p)] = v

            # Starting from m_0 = v_0 = 0 makes early estimates too small.
            # Remove that cold-start bias before calculating the update.
            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)

            # Descend in the direction of the bias-corrected gradient, while
            # dividing by its recent RMS magnitude.
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


    
