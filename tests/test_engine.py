# Unit tests for Value engine vs PyTorch autograd
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

if __name__ == "__main__":
    main()