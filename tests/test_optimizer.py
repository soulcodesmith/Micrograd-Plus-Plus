import torch
from micrograd.engine import Value
from micrograd.optimizers import SGD

def test_sgd_zero_grad():
    w = Value(2.0)
    b = Value(1.0)
    w.grad = 4.0
    b.grad = -2.0

    optimizer = SGD([w, b], lr=0.1)
    optimizer.zero_grad()

    assert w.grad == 0.0
    assert b.grad == 0.0


def test_sgd_step_manual():
    w = Value(2.0)
    b = Value(1.0)
    w.grad = 4.0
    b.grad = 2.0

    optimizer = SGD([w, b], lr=0.1)
    optimizer.step()

    # w.data = 2.0 - (0.1 * 4.0) = 1.6
    # b.data = 1.0 - (0.1 * 2.0) = 0.8
    assert abs(w.data - 1.6) < 1e-6
    assert abs(b.data - 0.8) < 1e-6


def test_sgd_step_vs_pytorch():
    # 1. Micrograd implementation
    w = Value(2.0)
    b = Value(1.0)
    x = 3.0
    y_target = 10.0

    y_pred = w * x + b
    loss = (y_pred - y_target) ** 2

    optimizer = SGD([w, b], lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 2. PyTorch equivalent
    w_pt = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    x_pt = torch.tensor(3.0, dtype=torch.float64)
    y_target_pt = torch.tensor(10.0, dtype=torch.float64)

    y_pred_pt = w_pt * x_pt + b_pt
    loss_pt = (y_pred_pt - y_target_pt) ** 2

    opt_pt = torch.optim.SGD([w_pt, b_pt], lr=0.01)
    opt_pt.zero_grad()
    loss_pt.backward()
    opt_pt.step()

    # Verify both parameters match PyTorch after update
    assert abs(w.data - w_pt.item()) < 1e-6
    assert abs(b.data - b_pt.item()) < 1e-6


def test_sgd_multi_step_loop_vs_pytorch():
    # Micrograd multi-step loop
    w = Value(1.5)
    b = Value(0.5)
    optimizer = SGD([w, b], lr=0.05)

    # PyTorch multi-step loop
    w_pt = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    opt_pt = torch.optim.SGD([w_pt, b_pt], lr=0.05)

    xs = [1.0, 2.0, 3.0]
    ys = [2.0, 4.0, 6.0]

    for epoch in range(5):
        # Micrograd pass
        optimizer.zero_grad()
        loss = Value(0.0)
        for x_val, y_val in zip(xs, ys):
            pred = w * x_val + b
            loss = loss + (pred - y_val) ** 2
        loss.backward()
        optimizer.step()

        # PyTorch pass
        opt_pt.zero_grad()
        loss_pt = torch.tensor(0.0, dtype=torch.float64)
        for x_val, y_val in zip(xs, ys):
            pred_pt = w_pt * x_val + b_pt
            loss_pt = loss_pt + (pred_pt - y_val) ** 2
        loss_pt.backward()
        opt_pt.step()

        assert abs(w.data - w_pt.item()) < 1e-6
        assert abs(b.data - b_pt.item()) < 1e-6


def test_sgd_momentum_step_manual():
    w = Value(2.0)
    optimizer = SGD([w], lr=0.1, momentum=0.9)

    # Step 1: grad = 4.0
    w.grad = 4.0
    optimizer.step()
    # v1 = 0.9 * 0.0 + 4.0 = 4.0
    # w.data = 2.0 - (0.1 * 4.0) = 1.6
    assert abs(w.data - 1.6) < 1e-6

    # Step 2: grad = 2.0 (velocity carries over from step 1)
    w.grad = 2.0
    optimizer.step()
    # v2 = 0.9 * 4.0 + 2.0 = 5.6
    # w.data = 1.6 - (0.1 * 5.6) = 1.04
    assert abs(w.data - 1.04) < 1e-6


def test_sgd_momentum_multi_step_vs_pytorch():
    # Micrograd setup
    w = Value(1.5)
    b = Value(0.5)
    optimizer = SGD([w, b], lr=0.02, momentum=0.9)

    # PyTorch setup (float64 for precision parity)
    w_pt = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    opt_pt = torch.optim.SGD([w_pt, b_pt], lr=0.02, momentum=0.9)

    xs = [1.0, 2.0, 3.0]
    ys = [2.0, 4.0, 6.0]

    for epoch in range(5):
        # Micrograd pass
        optimizer.zero_grad()
        loss = Value(0.0)
        for x_val, y_val in zip(xs, ys):
            pred = w * x_val + b
            loss = loss + (pred - y_val) ** 2
        loss.backward()
        optimizer.step()

        # PyTorch pass
        opt_pt.zero_grad()
        loss_pt = torch.tensor(0.0, dtype=torch.float64)
        for x_val, y_val in zip(xs, ys):
            pred_pt = w_pt * x_val + b_pt
            loss_pt = loss_pt + (pred_pt - y_val) ** 2
        loss_pt.backward()
        opt_pt.step()

        # Verify exact match against PyTorch at each epoch
        assert abs(w.data - w_pt.item()) < 1e-6
        assert abs(b.data - b_pt.item()) < 1e-6
