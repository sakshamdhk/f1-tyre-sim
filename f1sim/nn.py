"""Neural networks written from scratch in numpy (no autograd).

MLP     plain regression network: ReLU hidden layers, linear output, MSE loss.
DegNet  physics-informed network: the MLP reads only the race/stint conditions and outputs
        the parameters of a tyre-wear curve,
            deg = b * u1 + q * u2 + tau * in_traffic,
        with wear rate b = softplus(z1) >= 0 and curvature q = softplus(z2) >= 0, where
        u1, u2 are linear and quadratic wear since the stint's reference lap (features.wear_terms).
        The positivity constraint means a tyre can never get faster as it ages, and the curves
        are smooth, which the pit-stop optimiser needs.
Ensemble   several networks with different seeds; the spread of their predictions is the
        model's uncertainty.

Both networks use He initialisation, Adam, L2 weight decay, optional dropout, mini-batches,
standardised inputs and target, and early stopping on validation races that are held out
from the training data (whole races, so a race's conditions never leak into validation).

    python -m f1sim.nn        # gradient check + overfit test for both networks
"""
import numpy as np

from f1sim.common import SEED
from f1sim.features import COMPOUND_COLS, COND, LAP, wear_terms


def softplus(z):
    return np.logaddexp(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class MLP:
    needs_groups = True
    n_out = 1

    def __init__(self, hidden=(64, 32), lr=1e-3, wd=1e-4, dropout=0.0, epochs=200, batch=256,
                 patience=15, val_frac=0.2, seed=SEED, features=LAP):
        self.hidden, self.lr, self.wd, self.dropout = hidden, lr, wd, dropout
        self.epochs, self.batch, self.patience, self.val_frac = epochs, batch, patience, val_frac
        self.seed, self.features = seed, list(features)

    # ---- parameters -------------------------------------------------------------------
    def _init(self, n_in):
        rng = np.random.default_rng(self.seed)
        sizes = [n_in, *self.hidden, self.n_out]
        self.W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.extra = self._init_extra()
        self.opt = [np.zeros_like(p) for p in self.params() for _ in (0, 1)]
        self.t = 0

    def _init_extra(self):
        return []

    def params(self):
        return self.W + self.b + self.extra

    # ---- network body -----------------------------------------------------------------
    def _body(self, X, train=False, rng=None):
        acts, masks = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            if i < len(self.W) - 1:
                z = np.maximum(z, 0)
                if train and self.dropout > 0:
                    m = (rng.random(z.shape) >= self.dropout) / (1 - self.dropout)
                    z = z * m
                    masks.append(m)
                else:
                    masks.append(None)
            acts.append(z)
        return acts, masks

    def _body_backward(self, acts, masks, delta):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta + self.wd * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.W[i].T
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]
                delta = delta * (acts[i] > 0)
        return gW, gb

    # ---- head: plain regression output --------------------------------------------------
    def _inputs(self, df):
        return df[self.features].to_numpy(float), {}

    def _head(self, out, aux):
        return out[:, 0], None

    def _head_backward(self, dpred, out, aux, cache):
        return dpred[:, None], []

    def _loss_grads(self, X, aux, y, rng=None):
        acts, masks = self._body(X, train=rng is not None, rng=rng)
        pred, cache = self._head(acts[-1], aux)
        dpred = 2 * (pred - y) / len(y)
        dout, gextra = self._head_backward(dpred, acts[-1], aux, cache)
        gW, gb = self._body_backward(acts, masks, dout)
        return float(np.mean((pred - y) ** 2)), gW + gb + gextra

    def _adam(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params(), grads)):
            m, v = self.opt[2 * i], self.opt[2 * i + 1]
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g ** 2
            p -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

    # ---- training ------------------------------------------------------------------------
    def _split(self, n, groups, rng):
        if groups is None or self.val_frac == 0:
            return np.zeros(n, dtype=bool) if self.val_frac == 0 else rng.random(n) < self.val_frac
        g = np.unique(groups)
        return np.isin(groups, rng.choice(g, size=max(1, round(len(g) * self.val_frac)), replace=False))

    def fit(self, df, y, groups=None):
        rng = np.random.default_rng(self.seed)
        X, aux = self._inputs(df)
        y = np.asarray(y, float)
        val = self._split(len(y), None if groups is None else np.asarray(groups), rng)
        tr = ~val
        self.mu, self.sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        self.ys = y[tr].std() + 1e-8
        Xs, ys = (X - self.mu) / self.sd, y / self.ys
        sub = lambda a, m: {k: v[m] for k, v in a.items()}
        Xt, yt, at = Xs[tr], ys[tr], sub(aux, tr)
        Xv, yv, av = Xs[val], ys[val], sub(aux, val)
        self._init(X.shape[1])
        self.history, best, wait = [], None, 0
        for _ in range(self.epochs):
            idx = rng.permutation(len(yt))
            for s in range(0, len(idx), self.batch):
                j = idx[s:s + self.batch]
                _, grads = self._loss_grads(Xt[j], sub(at, j), yt[j], rng)
                self._adam(grads)
            tl = self._mse(Xt, at, yt)
            vl = self._mse(Xv, av, yv) if len(yv) else tl
            self.history.append((tl, vl))
            if best is None or vl < best[0]:
                best, wait = (vl, [p.copy() for p in self.params()]), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        for p, q in zip(self.params(), best[1]):
            p[...] = q
        self.best_epoch = int(np.argmin([h[1] for h in self.history])) + 1
        return self

    def _mse(self, X, aux, y):
        pred, _ = self._head(self._body(X)[0][-1], aux)
        return float(np.mean((pred - y) ** 2))

    def predict(self, df):
        X, aux = self._inputs(df)
        pred, _ = self._head(self._body((X - self.mu) / self.sd)[0][-1], aux)
        return pred * self.ys


class DegNet(MLP):
    """Physics-informed network: conditions -> (wear rate, curvature) -> lap-time loss.

    Residual design: each compound has its own base wear rate and curvature (free parameters,
    no weight decay), and the network only learns corrections to them from the conditions:
        b = softplus(base_b[compound] + z1(cond)),   q = softplus(base_q[compound] + z2(cond)).
    Weight decay therefore shrinks the model toward 'every compound has one wear rate' (the
    baseline) rather than toward a constant, which matters with only ~20 training circuits."""
    n_out = 2

    def __init__(self, hidden=(32, 16), features=COND, **kw):
        super().__init__(hidden=hidden, features=features, **kw)

    def _init_extra(self):
        # traffic penalty (through softplus) and per-compound base (rate, curvature) logits
        return [np.array([0.0]), np.zeros((3, 2))]

    def _init(self, n_in):
        super()._init(n_in)
        self.W[-1] *= 0.1            # start close to the per-compound baseline

    def _inputs(self, df):
        u1, u2 = wear_terms(df)
        return df[self.features].to_numpy(float), {"u1": u1, "u2": u2,
                                                   "traffic": df.in_traffic.to_numpy(float),
                                                   "comp": df[COMPOUND_COLS].to_numpy(float)}

    def _logits(self, out, aux):
        return out + aux["comp"] @ self.extra[1]

    def _head(self, out, aux):
        z = self._logits(out, aux)
        b, q, tau = softplus(z[:, 0]), softplus(z[:, 1]), softplus(self.extra[0][0])
        return b * aux["u1"] + q * aux["u2"] + tau * aux["traffic"], z

    def _head_backward(self, dpred, out, aux, z):
        dz = np.stack([dpred * aux["u1"] * sigmoid(z[:, 0]),
                       dpred * aux["u2"] * sigmoid(z[:, 1])], axis=1)
        dtau = np.array([np.sum(dpred * aux["traffic"]) * sigmoid(self.extra[0][0])])
        return dz, [dtau, aux["comp"].T @ dz]

    def curve_params(self, df):
        """Wear rate (s per lap) and curvature for each row's conditions."""
        X, aux = df[self.features].to_numpy(float), {"comp": df[COMPOUND_COLS].to_numpy(float)}
        z = self._logits(self._body((X - self.mu) / self.sd)[0][-1], aux)
        return softplus(z[:, 0]) * self.ys / 10, softplus(z[:, 1]) * self.ys / 100


class Ensemble:
    """Average of several networks trained with different seeds; std is the uncertainty."""
    needs_groups = True

    def __init__(self, make, n=5):
        self.make, self.n = make, n

    def fit(self, df, y, groups=None):
        self.members = [self.make(seed=SEED + i).fit(df, y, groups) for i in range(self.n)]
        self.make = None            # the factory is often a lambda, which cannot be saved
        return self

    def predict_all(self, df):
        return np.stack([m.predict(df) for m in self.members])

    def predict(self, df):
        return self.predict_all(df).mean(0)


# ---- correctness tests -------------------------------------------------------------------
def gradient_check(net, df, y, n_checks=20, eps=1e-6):
    rng = np.random.default_rng(0)
    X, aux = net._inputs(df)
    net.mu, net.sd, net.ys = X.mean(0), X.std(0) + 1e-8, 1.0
    net._init(X.shape[1])
    Xs = (X - net.mu) / net.sd
    _, grads = net._loss_grads(Xs, aux, y)
    worst = 0.0
    for _ in range(n_checks):
        k = rng.integers(len(net.params()))
        p = net.params()[k]
        i = tuple(rng.integers(s) for s in p.shape)
        old = p[i]
        p[i] = old + eps
        lp = net._loss_grads(Xs, aux, y)[0]
        p[i] = old - eps
        lm = net._loss_grads(Xs, aux, y)[0]
        p[i] = old
        num, ana = (lp - lm) / (2 * eps), grads[k][i]
        worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-10))
    return worst


def _toy(n, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.normal(size=(n, len(LAP))), columns=LAP)
    d["tyre_age"] = rng.integers(1, 40, n)
    d["stint_start_age"] = rng.integers(1, 5, n)
    d["in_traffic"] = rng.integers(0, 2, n)
    u1, u2 = wear_terms(d)
    y = (0.5 + 0.3 * np.tanh(d.track_temp)) * u1 + 0.1 * u2 + 0.4 * d.in_traffic
    return d, y.to_numpy()


if __name__ == "__main__":
    d, y = _toy(64)
    for cls in (MLP, DegNet):
        net = cls(hidden=(8, 4), wd=0.0)
        err = gradient_check(net, d, y)
        print(f"{cls.__name__:<7} gradient check, worst relative error {err:.1e}  ({'OK' if err < 1e-5 else 'BAD'})")
        net = cls(hidden=(32, 16), lr=3e-3, wd=0.0, epochs=1500, batch=64, patience=10 ** 9, val_frac=0).fit(d, y)
        mse = net.history[-1][0]
        print(f"{cls.__name__:<7} overfit test on 64 rows, scaled train MSE {mse:.4f}  ({'OK' if mse < 0.05 else 'BAD'})")
