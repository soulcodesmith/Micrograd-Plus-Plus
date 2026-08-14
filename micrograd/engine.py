# Core Value autograd engine
import math

class Value:
    """
    Wraps a scalar float and tracks its computation history and gradient.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self), f'**{other}')

        def _backward():
            # Power rule from calculus: d/dx(x^n) = n * x^(n-1)
            self.grad  += (other * (self.data ** (other -1))) * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)

    # Activations Functions

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self), 'tanh')

        def _backward():
        # Derivative: d/dx(tanh(x)) = 1 - tanh(x)^2 = 1 - t^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            # Derivative: 1 if x > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
    

        def backward(self):
            """
            Orchestrates automatic differentiation by topologically sorting the graph
            and propagating gradients backwards via the chain rule.
            """
            topo = []
            visited = set()

        # Build the Topological Sort (Dependency Tree)
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

       # Set the gradient of the root node (usually Loss) to 1.0 (dL/dL = 1)
        self.grad = 1.0

        # Traverse in reverse topological order so every node receives its full
        # gradient from downstream before it passes gradients upstream.
        for node in reversed(topo):
            node._backward()
