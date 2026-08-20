# Micrograd++

An educational scalar automatic-differentiation engine, built from scratch while learning how neural networks and PyTorch autograd work.

![A computational graph showing forward values and reverse gradients](assets/computational-graph.png)

## Why I am building this

PyTorch makes training a neural network easy to use, but it also hides a lot of the mechanics. I wanted to understand what happens behind `loss.backward()` instead of treating it as magic.

This project is my attempt to rebuild those ideas in small pieces. The goal is not to compete with PyTorch. The goal is to be able to explain every part of the engine and eventually use it to train a small neural network.

## What works right now

The current engine works with scalar values and supports:

- computation-graph tracking;
- addition and multiplication;
- negation, subtraction, and reverse arithmetic;
- powers and division;
- `tanh` activation;
- automatic backpropagation through a topologically sorted graph;
- gradient checks for small hand-calculated examples.

For example, the engine can differentiate:

```python
from micrograd.engine import Value

a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + a
d.backward()

print(d.data)  # 8.0
print(a.grad)  # 4.0
print(b.grad)  # 2.0
```

The result for `a.grad` is `4.0` because `a` affects `d` through two paths:

```text
d = a*b + a
∂d/∂a = b + 1 = 3 + 1 = 4
```

## How the engine works

Each `Value` stores:

- its numerical data;
- its accumulated gradient;
- the values that created it;
- the operation that created it;
- a local backward function.

The forward pass builds a graph while calculating values. Calling `backward()` walks that graph in reverse and applies the chain rule one operation at a time.

## Running the current checks

From the project root:

```bash
python3 -m tests.test_engine
python3 -m py_compile micrograd/engine.py tests/test_engine.py
```

## Project structure

```text
micrograd-plus-plus/
├── micrograd/
│   ├── engine.py       # scalar Value class and autograd engine
│   ├── nn.py           # neural-network building blocks (planned)
│   ├── losses.py       # reusable loss functions (planned)
│   └── optimizers.py   # parameter-update algorithms (planned)
├── tests/
│   └── test_engine.py  # small correctness checks
├── Notebooks/
│   └── engine.ipynb    # learning notes and experiments
├── examples/           # future runnable examples
└── assets/
    └── computational-graph.png
```

## Next steps

1. Add more gradient tests and compare them with PyTorch.
2. Add ReLU and other activation functions.
3. Build `Neuron`, `Layer`, and `MLP` classes.
4. Train a small MLP on XOR.
5. Add reusable losses and an SGD optimizer.

## References

- Andrej Karpathy — [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- Andrej Karpathy — [micrograd](https://github.com/karpathy/micrograd)

This is a learning project. The implementation is intentionally small and readable before it becomes feature-complete.
