# Unit tests for Value engine vs PyTorch autograd
import torch
from micrograd.engine import Value

def main():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b
    d = c + a

    d.backward()
    assert d.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0

    print("Test passed!")


def test_add_backward():
    a = Value(2.0)
    b = Value(3.0)
    (a + b).backward()

    assert a.grad == 1.0
    assert b.grad == 1.0

    print("Add backward test passed!")


def test_chained_graph():
    a = Value(2.0)
    b = Value(3.0)
    d = a * b + a 
    d.backward()

    assert d.data == 8
    assert a.grad == 4
    assert b.grad == 2

    print("Chained graph test passed!")


def test_add_pytorch():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()

    # PyTorch equivalent
    a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    c_torch = a_torch + b_torch
    c_torch.backward()

    assert c.data == c_torch.item()
    assert a.grad == a_torch.grad.item()
    assert b.grad == b_torch.grad.item()

    print("Add PyTorch test passed!")

def test_sub_pytorch():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    c.backward()

    # PyTorch equivalent
    a_torch = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    c_torch = a_torch - b_torch
    c_torch.backward()

    assert a.grad == a_torch.grad.item()
    assert b.grad == b_torch.grad.item()
    assert c.data == c_torch.item()

    print("Sub PyTorch test passed!")

def test_mul_pytorch():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()

    # PyTorch equivalent
    a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    c_torch = a_torch * b_torch
    c_torch.backward()

    assert c.data == c_torch.item()
    assert a.grad == a_torch.grad.item()
    assert b.grad == b_torch.grad.item()

    print("Mul PyTorch test passed!")


def test_rsub():
    a = Value(2.0)
    c = 3.0 - a
    c.backward()

    a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    c_torch = 3.0 - a_torch
    c_torch.backward()

    assert a.grad == a_torch.grad.item()
    assert c.data == c_torch.item()


def test_div_pytorch():
    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    c.backward()

    # PyTorch equivalent
    a_torch = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    c_torch = a_torch / b_torch
    c_torch.backward()

    assert c.data == c_torch.item()
    assert a.grad == a_torch.grad.item()
    assert b.grad == b_torch.grad.item()

def test_pow_pytorch():
    a = Value(2.0)
    c = a ** 3
    c.backward()

    # PyTorch equivalent
    a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    c_torch = a_torch ** 3
    c_torch.backward()

    assert c.data == c_torch.item()
    assert a.grad == a_torch.grad.item()


def test_tanh_pytorch():
    a = Value(0.5)
    c = a.tanh()
    c.backward()

    # PyTorch equivalent
    a_torch = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    c_torch = torch.tanh(a_torch)
    c_torch.backward()

    assert abs(c.data - c_torch.item()) < 1e-6
    assert abs(a.grad - a_torch.grad.item()) < 1e-6


def test_gradient_accumulation():
    # Tests reusing the same variable multiple times (y = x + x -> grad = 2)
    x = Value(3.0)
    y = x + x
    y.backward()
    assert y.data == 6.0
    assert x.grad == 2.0
    # y = x * x -> dy/dx = 2*x = 6.0
    x2 = Value(3.0)
    y2 = x2 * x2
    y2.backward()
    assert y2.data == 9.0
    assert x2.grad == 6.0


def test_relu():
    x = Value(-2.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0

    x = Value(2.0)
    y = x.relu()
    y.backward()
    assert y.data == 2.0
    assert x.grad == 1.0


if __name__ == "__main__":
    main()
