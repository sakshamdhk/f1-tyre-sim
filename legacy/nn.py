import numpy as np


class MLP:
    """Small numpy network for regression: ReLU hidden layers, linear output, MSE loss, Adam."""
    needs_groups = True

    def __init__(self, hidden=(16, 8), lr=3e-3, wd=1e-4, epochs=300, batch=64, patience=20, seed=42):
        self.hidden, self.lr, self.wd = hidden, lr, wd
        self.epochs, self.batch, self.patience, self.seed = epochs, batch, patience, seed

    def _init(self, n_in):
        rng = np.random.default_rng(self.seed)
        sizes = [n_in, *self.hidden, 1]
        self.W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else np.maximum(z, 0))
        return acts

    def _backward(self, acts, y):
        delta = 2 * (acts[-1] - y[:, None]) / len(y)
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, acts[i].T @ delta + self.wd * self.W[i])
            gb.insert(0, delta.sum(axis=0))
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    def _adam(self, gW, gb):
        self.t += 1
        for i in range(len(self.W)):
            for P, G, M, V in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                M[i] = 0.9 * M[i] + 0.1 * G[i]
                V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
                P[i] -= self.lr * (M[i] / (1 - 0.9 ** self.t)) / (np.sqrt(V[i] / (1 - 0.999 ** self.t)) + 1e-8)

    def fit(self, X, y, groups=None):
        self.cols = list(X.columns)
        X, y = X.to_numpy(float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        if groups is not None:
            races = np.unique(groups)
            val = np.isin(groups, rng.choice(races, size=max(1, len(races) // 8), replace=False))
        else:
            val = rng.random(len(y)) < 0.1
        self.mu, self.sd = X[~val].mean(0), X[~val].std(0) + 1e-8
        self.ym, self.ys = y[~val].mean(), y[~val].std() + 1e-8
        Xt, yt = (X[~val] - self.mu) / self.sd, (y[~val] - self.ym) / self.ys
        Xv, yv = (X[val] - self.mu) / self.sd, (y[val] - self.ym) / self.ys
        self._init(X.shape[1])
        self.history, best, wait = [], (np.inf, None, None), 0
        for _ in range(self.epochs):
            idx = rng.permutation(len(yt))
            for s in range(0, len(idx), self.batch):
                j = idx[s:s + self.batch]
                self._adam(*self._backward(self._forward(Xt[j]), yt[j]))
            tl = float(np.mean((self._forward(Xt)[-1][:, 0] - yt) ** 2))
            vl = float(np.mean((self._forward(Xv)[-1][:, 0] - yv) ** 2)) if len(yv) else tl
            self.history.append((tl, vl))
            if vl < best[0]:
                best, wait = (vl, [w.copy() for w in self.W], [b.copy() for b in self.b]), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.W, self.b = best[1], best[2]
        return self

    def predict(self, X):
        Z = (X[self.cols].to_numpy(float) - self.mu) / self.sd
        return self._forward(Z)[-1][:, 0] * self.ys + self.ym


if __name__ == "__main__":
    import pandas as pd
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(64, 4)), columns=list("abcd"))
    y = X.a * 2 - X.b + 0.5 * X.c * X.d
    m = MLP(hidden=(8, 4), wd=0.0)
    m._init(4)
    Xn, yn = X.to_numpy(), y.to_numpy()
    gW, _ = m._backward(m._forward(Xn), yn)
    w, eps, worst = m.W[0], 1e-6, 0.0
    for (i, j) in [(0, 0), (1, 2), (3, 5)]:
        w[i, j] += eps; lp = np.mean((m._forward(Xn)[-1][:, 0] - yn) ** 2)
        w[i, j] -= 2 * eps; lm = np.mean((m._forward(Xn)[-1][:, 0] - yn) ** 2)
        w[i, j] += eps
        num = (lp - lm) / (2 * eps)
        worst = max(worst, abs(num - gW[0][i, j]) / (abs(num) + abs(gW[0][i, j]) + 1e-12))
    print(f"gradient check, worst relative error: {worst:.2e}  ({'OK' if worst < 1e-5 else 'BAD'})")
    m = MLP(hidden=(32, 16), lr=0.01, wd=0.0, epochs=2000, batch=64, patience=10 ** 9).fit(X, y)
    print(f"overfit test, train MSE (scaled): {m.history[-1][0]:.4f}  ({'OK' if m.history[-1][0] < 0.05 else 'BAD'})")

