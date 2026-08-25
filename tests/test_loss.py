from micrograd.engine import Value
from micrograd.losses import MSELoss

#test
def test_mse_loss_value():
    predictions = [Value(3.0), Value(5.0)]
    targets = [2.0, 7.0]

    loss_fn = MSELoss()
    loss = loss_fn(predictions, targets)

    assert isinstance(loss,Value)
    assert loss.data == 2.5

def test_mse_loss_backwarda():
    predictions = [Value(3.0), Value(5.0)]
    targets = [2.0, 7.0]

    loss = MSELoss()(predictions, targets)
    loss.backward()

    assert predictions[0].grad == 1.0
    assert predictions[1].grad == -2.0

