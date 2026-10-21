def accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred")
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(1, len(y_true))
