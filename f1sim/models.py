"""All models, from the simplest baseline to the physics-informed network ensemble.

Every model is wrapped in `Tuned`: its hyperparameters are chosen on validation circuits taken
out of the training data (nested validation), so the test circuits are never used for tuning.
"""
import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from f1sim.common import SEED
from f1sim.features import COMPOUND_COLS, COND, FEATURE_SETS, LAP, wear_terms
from f1sim.nn import MLP, DegNet, Ensemble


class LinearDeg:
    """Linear wear-rate model: deg = (w . cond) * u1 + (v . compound) * u2 + t * traffic.
    It has the same shape as DegNet, with the network replaced by a linear function, so the
    two can be compared directly. With cond = compounds only, it is the baseline
    'every compound has one fixed wear rate'. The L2 penalty (alpha) only shrinks the
    condition effects; the per-compound wear rates and the traffic term are left free."""

    FREE = 1e3     # free columns are scaled up, which divides their penalty by FREE**2

    def __init__(self, cond=COND, alpha=1e-6):
        self.cond, self.alpha = list(cond), alpha

    def _design(self, d):
        u1, u2 = wear_terms(d)
        C = (d[self.cond].to_numpy(float) - self.mu) / self.sd
        C[:, self.is_comp] = d[self.comp_cols].to_numpy(float) * self.FREE     # keep one-hots as 0/1
        return np.column_stack([u1[:, None] * C, u2[:, None] * d[COMPOUND_COLS].to_numpy(float) * self.FREE,
                                d.in_traffic.to_numpy(float) * self.FREE])

    def fit(self, d, y):
        X = d[self.cond].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        self.comp_cols = [c for c in self.cond if c in COMPOUND_COLS]
        self.is_comp = np.array([c in COMPOUND_COLS for c in self.cond])
        self.m = Ridge(alpha=self.alpha, fit_intercept=False).fit(self._design(d), y)
        return self

    def predict(self, d):
        return self.m.predict(self._design(d))


class Columns:
    """Feeds a fixed list of columns to a scikit-learn model."""

    def __init__(self, model, features=LAP):
        self.model, self.features = model, list(features)

    def fit(self, d, y):
        self.model.fit(d[self.features], y)
        return self

    def predict(self, d):
        return self.model.predict(d[self.features])


class Tuned:
    """Pick hyperparameters on held-out validation circuits, then refit on all training data."""

    def __init__(self, make, grid, val_frac=0.25, final=None):
        self.make, self.grid, self.val_frac, self.final = make, grid, val_frac, final or {}

    def fit(self, d, y, groups):
        y, groups = np.asarray(y), np.asarray(groups)
        keys = list(self.grid)
        combos = [dict(zip(keys, v)) for v in itertools.product(*self.grid.values())]
        if len(combos) == 1:
            self.best = combos[0]
        else:
            rng = np.random.default_rng(SEED)
            g = np.unique(groups)
            val = np.isin(groups, rng.choice(g, size=max(1, round(len(g) * self.val_frac)), replace=False))
            self.scores = []
            for p in combos:
                m = self._fit(self.make(**p), d[~val], y[~val], groups[~val])
                self.scores.append((float(np.mean(np.abs(m.predict(d[val]) - y[val]))), p))
            self.best = min(self.scores, key=lambda s: s[0])[1]
        self.model = self._fit(self.make(**{**self.best, **self.final}), d, y, groups)
        return self

    @staticmethod
    def _fit(m, d, y, groups):
        return m.fit(d, y, groups) if getattr(m, "needs_groups", False) else m.fit(d, y)

    def predict(self, d):
        return self.model.predict(d)


def rf(n_estimators, min_samples_leaf, max_features):
    return Columns(RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                         max_features=max_features, n_jobs=-1, random_state=SEED))


def gbm(learning_rate, max_leaf_nodes, l2_regularization):
    return Columns(HistGradientBoostingRegressor(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                                 l2_regularization=l2_regularization, max_iter=500,
                                                 early_stopping=True, random_state=SEED))


def degnet_ensemble(features, hidden, wd, n=5):
    return Ensemble(lambda seed: DegNet(hidden=hidden, wd=wd, features=FEATURE_SETS[features], lr=3e-3,
                                        seed=seed), n=n)


# name -> (description, factory returning an unfitted Tuned model)
MODELS = {
    "baseline": ("Compound wear rate only", lambda: Tuned(lambda: LinearDeg(cond=COMPOUND_COLS), {})),
    "linear": ("Linear wear-rate model (OLS)", lambda: Tuned(lambda: LinearDeg(), {})),
    "ridge": ("Linear wear-rate model, L2 (Ridge)",
              lambda: Tuned(lambda alpha: LinearDeg(alpha=alpha), {"alpha": [10, 100, 1e3, 1e4, 1e5]})),
    "rf": ("Random forest", lambda: Tuned(rf, {"n_estimators": [200], "min_samples_leaf": [20, 100],
                                                "max_features": [0.5, 1.0]})),
    "gbm": ("Gradient boosting", lambda: Tuned(gbm, {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31],
                                                     "l2_regularization": [0.0, 1.0]})),
    "mlp": ("Neural network (numpy MLP)",
            lambda: Tuned(lambda hidden, wd, dropout: MLP(hidden=hidden, wd=wd, dropout=dropout, lr=3e-3),
                          {"hidden": [(16, 8), (32, 16)], "wd": [1e-2, 1e-1], "dropout": [0.2, 0.5]})),
    "degnet": ("Physics-informed NN ensemble (numpy)",
               lambda: Tuned(degnet_ensemble, {"features": ["phys", "full"], "hidden": [(16, 8)],
                                               "wd": [0.03, 0.1, 0.3], "n": [1]}, final={"n": 5})),
}
