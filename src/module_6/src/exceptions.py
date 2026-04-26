class UserNotFoundException(Exception):
    """Raised when a user_id is not present in the feature store."""


class PredictionException(Exception):
    """Raised when the model fails to produce a prediction."""
