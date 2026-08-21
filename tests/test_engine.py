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



if __name__ == "__main__":
    main()