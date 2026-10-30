from .mlp import MLPClassifier
from .cnn import CNNClassifier

def get_model(name: str, **kwargs):
    """
    Factory function to retrieve model instances.

    Args:
        name: 'mlp' or 'cnn'
        **kwargs: Optional model constructor arguments.

    Returns:
        nn.Module instance
    """
    name_clean = name.strip().lower()
    if name_clean == "mlp":
        return MLPClassifier(**kwargs)
    elif name_clean == "cnn":
        return CNNClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model name '{name}'. Choose between 'mlp' and 'cnn'.")

__all__ = ["MLPClassifier", "CNNClassifier", "get_model"]
