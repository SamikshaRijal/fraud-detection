from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


class BalancedGradientBoostingClassifier(GradientBoostingClassifier):
    """Compute inverse-frequency weights inside each CV fit, never on validation data."""

    def fit(self, X, y, sample_weight=None, monitor=None):
        weights = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            weights = weights * sample_weight
        return super().fit(X, y, sample_weight=weights, monitor=monitor)

