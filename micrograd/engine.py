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
