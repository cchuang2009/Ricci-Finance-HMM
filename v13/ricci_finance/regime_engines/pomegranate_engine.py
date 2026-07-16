from __future__ import annotations

import numpy as np

from .base import BaseRegimeEngine


class PomegranateDenseEngine(BaseRegimeEngine):
    """pomegranate v1 DenseHMM adapter.

    pomegranate v1 currently provides posterior probabilities but does not expose
    a stable Viterbi API compatible with this project. Hard states therefore use
    posterior argmax and are explicitly labeled as posterior decoding.
    """

    name = "pomegranate"
    supports_viterbi = False
    supports_probabilities = True

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 250,
        tol: float = 1e-4,
        device: str | None = None,
    ) -> None:
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = device
        self.model = None

    def fit(self, X: np.ndarray) -> "PomegranateDenseEngine":
        import torch
        from pomegranate.distributions import Normal
        from pomegranate.hmm import DenseHMM

        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.as_tensor(X[None, :, :], dtype=torch.float32, device=target_device)
        distributions = [Normal() for _ in range(self.n_components)]
        self.model = DenseHMM(
            distributions,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=False,
        ).to(target_device)
        self.model.fit(tensor)
        self._device = target_device
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Engine must be fitted before predict_proba().")
        import torch

        tensor = torch.as_tensor(X[None, :, :], dtype=torch.float32, device=self._device)
        return self.model.predict_proba(tensor)[0].detach().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X: np.ndarray) -> float | None:
        if self.model is None:
            return None
        try:
            import torch

            tensor = torch.as_tensor(X[None, :, :], dtype=torch.float32, device=self._device)
            value = self.model.log_probability(tensor)
            return float(value.detach().cpu().numpy().reshape(-1)[0])
        except Exception:
            return None
