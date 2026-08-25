# Loss functions
from micrograd.engine import Value

class MSELoss:
    def __call__(self, predictions, targets):
        if len(predictions) != len(targets):
            raise ValueError("Predictions and tragets must have the same length")

        total_loss = Value(0.0)

        for prediction, target in zip(predictions, targets):
            error = prediction - target
            squared_error = error ** 2
            total_loss = total_loss + squared_error

        mean_loss = total_loss/ len(predictions)

        return mean_loss

    