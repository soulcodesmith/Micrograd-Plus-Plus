import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op

        # This will later contain the local backward rule.
        self._backward = lambda: None


    def __repr__(self):
        return (f"Value(data={self.data}, grad={self.grad})")


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*' )


        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out


    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out


    def backward(self):
        # topological sort
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0

        # run the backward rules
        for node in reversed(topo):
            node._backward()
    

    def __neg__(self):
        return self * -1


    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))

        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1) * out.grad) # chainrule

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)


# Activation functions
    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1 - s)) * out.grad  # chain rule

        out._backward = _backward
        return out

    def leakyrelu(self, alpha=0.01):
        out = Value(self.data if self.data > 0 else alpha * self.data, (self,), 'LeakyReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else alpha) * out.grad

        out._backward = _backward
        return out


