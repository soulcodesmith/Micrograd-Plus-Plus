# Optimizers
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


    
