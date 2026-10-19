"""Memory-bank anomaly detection in the style of PatchCore, with no deep-learning runtime.

Why: defense parts rarely have enough defect samples to train a classifier, so the
practical approach is *one-class* learning from good parts only. PatchCore does
this with a memory bank of patch embeddings and nearest-neighbour scoring. Here the
embedding is a hand-crafted multi-scale local descriptor (so the package stays
pip-installable offline with numpy/scipy only); swapping in a CNN backbone
(e.g. anomalib's PatchCore) only changes `patch_features`.

Model file (.npz): memory bank, feature normalisation, threshold, parameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .vision import VisionError, measure_geometry, to_gray8

PATCH = 8
STRIDE = 4


def _part_frame(img: np.ndarray, size: int = 192, has_pocket: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Crop the part to a canonical frame (centred, fixed scale) and return (image, face mask).

    Raises VisionError when the part cannot be located (callers decide how to report it).
    """
    img = to_gray8(img)
    g = measure_geometry(img, require_pocket=has_pocket)
    o, p = g.outer, g.pocket
    scale = (size / 2 - 4) / o.r
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    sx = (xx - size / 2) / scale + o.cx
    sy = (yy - size / 2) / scale + o.cy
    canon = ndi.map_coordinates(img.astype(float), [sy, sx], order=1, mode="nearest")
    r = np.hypot(xx - size / 2, yy - size / 2)
    face = r < o.r * scale - 4
    if p is not None:
        rp = np.hypot(xx - size / 2 - (p.cx - o.cx) * scale, yy - size / 2 - (p.cy - o.cy) * scale)
        face &= rp > p.r * scale + 4
    return canon, face


BACKENDS = ("handcrafted", "cnn")


def cnn_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return True


_CNN = {}


def _cnn_embedder():
    """ResNet-18 layer2 feature map (stride 8) as a PatchCore-style backbone. Pretrained weights
    are loaded from QINSPECT_CNN_WEIGHTS (offline) or torchvision's download; without either,
    random weights are used with a warning (still a usable but weaker embedding)."""
    if "model" in _CNN:
        return _CNN["model"]
    import os
    import warnings

    import torch
    import torchvision

    weights_path = os.environ.get("QINSPECT_CNN_WEIGHTS")
    net = torchvision.models.resnet18(weights=None)
    if weights_path:
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        try:
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:  # noqa: BLE001 - offline
            warnings.warn(f"cnn backend: pretrained weights unavailable ({exc}); using random init", stacklevel=2)
    net.eval()
    trunk = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
    _CNN["model"] = trunk
    return trunk


def _cnn_patch_features(canon: np.ndarray, face: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    import torch

    trunk = _cnn_embedder()
    x = torch.from_numpy((canon / 255.0).astype(np.float32))[None, None]
    x = (x.repeat(1, 3, 1, 1) - 0.45) / 0.225
    with torch.no_grad():
        fm = trunk(x)[0]  # [C, h, w], stride 8
        fm = torch.nn.functional.avg_pool2d(fm[None], 3, stride=1, padding=1)[0]  # local aggregation as in PatchCore
    C, h, w = fm.shape
    F = fm.permute(1, 2, 0).reshape(h * w, C).numpy()
    # a patch is valid when its receptive-field centre cell is fully on the face
    cell = size // h
    fwin = face[: h * cell, : w * cell].reshape(h, cell, w, cell).all(axis=(1, 3))
    return F, fwin.reshape(-1), (h, w)


def patch_features(img: np.ndarray, size: int = 192, has_pocket: bool = True,
                   backend: str = "handcrafted") -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return (features [N, D], valid-patch mask [N], grid shape) for one image.

    backend "handcrafted" (numpy/scipy only) or "cnn" (torchvision ResNet-18 layer2).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    canon, face = _part_frame(img, size, has_pocket)
    if backend == "cnn":
        if not cnn_available():
            raise RuntimeError("cnn backend needs `pip install torch torchvision` (qinspect[cnn])")
        return _cnn_patch_features(canon, face, size)
    # remove slow illumination so features describe texture, not lighting
    flat = canon - ndi.gaussian_filter(canon, 8)
    gx = ndi.sobel(flat, axis=1)
    gy = ndi.sobel(flat, axis=0)
    channels = [
        flat,
        ndi.gaussian_filter(flat, 1.0),
        ndi.gaussian_filter(flat, 2.0),
        np.hypot(gx, gy),
        np.abs(ndi.laplace(ndi.gaussian_filter(flat, 1.0))),
        flat - ndi.minimum_filter(flat, 5),  # dark-spot response
    ]
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    channels.append(rr * 20.0)  # radius as a weak positional cue (texture varies with radius)

    n = (size - PATCH) // STRIDE + 1
    feats, valid = [], []
    for ch in channels:
        win = np.lib.stride_tricks.sliding_window_view(ch, (PATCH, PATCH))[::STRIDE, ::STRIDE]
        win = win.reshape(n, n, -1)
        feats.append(win.mean(-1))
        feats.append(win.std(-1))
        feats.append(win.min(-1))
    fwin = np.lib.stride_tricks.sliding_window_view(face, (PATCH, PATCH))[::STRIDE, ::STRIDE].reshape(n, n, -1)
    valid = fwin.all(-1)
    F = np.stack(feats, -1).reshape(n * n, -1)
    return F, valid.reshape(-1), (n, n)


def _greedy_coreset(X: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """k-center greedy subsampling (PatchCore's coreset) on a random projection."""
    if len(X) <= m:
        return np.arange(len(X))
    proj = X @ rng.normal(size=(X.shape[1], min(32, X.shape[1])))
    idx = [int(rng.integers(len(X)))]
    d = np.linalg.norm(proj - proj[idx[0]], axis=1)
    for _ in range(m - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(proj - proj[j], axis=1))
    return np.array(idx)


@dataclass
class AnomalyModel:
    bank: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    threshold: float
    train_scores: np.ndarray
    k: int = 1
    size: int = 192
    has_pocket: bool = True
    backend: str = "handcrafted"

    def __post_init__(self):
        if self.k < 1 or self.k > len(self.bank):
            raise ValueError(f"k must be in 1..{len(self.bank)} (bank size)")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        self._tree = cKDTree(self.bank)

    def score_map(self, img: np.ndarray) -> tuple[float, np.ndarray]:
        """Image score and patch-level anomaly map. Raises VisionError if the part is not found."""
        F, valid, grid = patch_features(img, self.size, self.has_pocket, self.backend)
        Z = (F - self.mu) / self.sd
        if valid.sum() < 3:
            raise VisionError("degenerate_geometry", "fewer than 3 face patches")
        d, _ = self._tree.query(Z[valid], k=self.k)
        d = d if self.k == 1 else d.mean(1)
        amap = np.zeros(len(F))
        amap[valid] = d
        amap = amap.reshape(grid)
        return float(np.sort(d)[-3:].mean()), amap

    def score(self, img: np.ndarray) -> float:
        return self.score_map(img)[0]

    def predict(self, img: np.ndarray) -> tuple[bool, float]:
        s = self.score(img)
        return s > self.threshold, s

    def save(self, path: str | Path, meta: dict | None = None) -> str:
        """Write the model; returns its content hash. The path always gets a .npz suffix
        (the resolved location is available afterwards as `self.path`)."""
        path = model_file(path)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez_compressed(fh, bank=self.bank.astype(np.float32), mu=self.mu.astype(np.float64),
                                sd=self.sd.astype(np.float64), threshold=float(self.threshold),
                                train_scores=self.train_scores.astype(np.float64), k=int(self.k),
                                size=int(self.size), has_pocket=bool(self.has_pocket), backend=str(self.backend),
                                meta=json.dumps(meta or {}))
        return model_hash(path)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyModel":
        z = np.load(path, allow_pickle=False)
        has_pocket = bool(z["has_pocket"]) if "has_pocket" in z.files else True
        backend = str(z["backend"]) if "backend" in z.files else "handcrafted"
        return cls(z["bank"], z["mu"], z["sd"], float(z["threshold"]), z["train_scores"], int(z["k"]),
                   int(z["size"]), has_pocket, backend)


def model_file(path: str | Path) -> Path:
    """The file a model is (or will be) saved to: always with a .npz suffix."""
    p = Path(path)
    return p if p.suffix == ".npz" else p.with_suffix(p.suffix + ".npz")


def model_hash(path: str | Path) -> str:
    return hashlib.sha256(model_file(path).read_bytes()).hexdigest()[:16]


MIN_TRAIN_IMAGES = 4
RECOMMENDED_TRAIN_IMAGES = 10


def train(good_images: list[np.ndarray], *, coreset: int = 4000, k: int = 1, seed: int = 0,
          threshold_sigma: float = 5.0, has_pocket: bool = True, names: list[str] | None = None,
          backend: str = "handcrafted") -> AnomalyModel:
    """Fit the memory bank on good images.

    Threshold = median + `threshold_sigma` * spread of 2-fold held-out scores on the training
    set (so it reflects unseen good parts). The spread is floored at 10 % of the median so
    near-duplicate training sets cannot collapse the threshold; exact duplicate images are
    dropped with a warning. Raises ValueError naming the image if one cannot be segmented.
    """
    import warnings

    if names is not None and len(names) not in (0, len(good_images)):
        raise ValueError(f"names must have one entry per image ({len(names)} names, {len(good_images)} images)")
    names = names or [f"image[{i}]" for i in range(len(good_images))]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > coreset:
        raise ValueError(f"k ({k}) cannot exceed coreset ({coreset})")
    # drop exact duplicates (same bytes) - they would make held-out scores meaningless
    uniq, seen = [], set()
    for im, nm in zip(good_images, names):
        h = hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
        if h in seen:
            warnings.warn(f"train: dropping duplicate image {nm}", stacklevel=2)
            continue
        seen.add(h)
        uniq.append((im, nm))
    if len(uniq) < MIN_TRAIN_IMAGES:
        raise ValueError(f"need at least {MIN_TRAIN_IMAGES} distinct good images, got {len(uniq)}")
    if len(uniq) < RECOMMENDED_TRAIN_IMAGES:
        warnings.warn(f"train: only {len(uniq)} images; threshold estimate is rough (recommend >= {RECOMMENDED_TRAIN_IMAGES})",
                      stacklevel=2)
    rng = np.random.default_rng(seed)
    per_image = []
    for im, nm in uniq:
        try:
            F, valid, _ = patch_features(im, has_pocket=has_pocket, backend=backend)
        except VisionError as exc:
            raise ValueError(f"cannot use {nm} for training: {exc}") from exc
        if valid.sum() < 3:
            raise ValueError(f"cannot use {nm} for training: too few face patches")
        per_image.append(F[valid].astype(np.float32))
    allF = np.concatenate(per_image)
    mu, sd = allF.mean(0), allF.std(0) + 1e-6
    Z = [(f - mu) / sd for f in per_image]

    scores = np.zeros(len(Z))
    folds = np.arange(len(Z)) % 2
    for f in (0, 1):
        bank_f = np.concatenate([z for z, g in zip(Z, folds) if g != f])
        sel = _greedy_coreset(bank_f, coreset, rng)
        kk = min(k, len(sel))
        tree = cKDTree(bank_f[sel])
        for i in np.nonzero(folds == f)[0]:
            d, _ = tree.query(Z[i], k=kk)
            d = d if kk == 1 else d.mean(1)
            scores[i] = float(np.sort(d)[-3:].mean())
    med = float(np.median(scores))
    spread = max(1.4826 * float(np.median(np.abs(scores - med))), float(scores.std()), 0.1 * med, 1e-6)
    thr = float(med + threshold_sigma * spread)

    allZ = np.concatenate(Z)
    bank = allZ[_greedy_coreset(allZ, coreset, rng)]
    return AnomalyModel(bank=bank, mu=mu, sd=sd, threshold=thr, train_scores=scores, k=min(k, len(bank)),
                        has_pocket=has_pocket, backend=backend)


def auroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """Area under ROC via the Mann-Whitney statistic (no sklearn dependency)."""
    neg, pos = np.asarray(scores_neg), np.asarray(scores_pos)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    ranks = np.empty(len(allv))
    order = np.argsort(allv)
    sorted_v = allv[order]
    # average ranks for ties
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[len(neg) :].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def compare_backends(good: list[np.ndarray], test_good: list[np.ndarray], test_bad: dict[str, list[np.ndarray]],
                     *, coreset: int = 2000, seed: int = 0, backends: tuple[str, ...] = BACKENDS) -> dict:
    """Train each backend on the same images and report AUROC / FP rate / timings / model size."""
    import time

    out = {}
    for b in backends:
        if b == "cnn" and not cnn_available():
            out[b] = {"skipped": "torch/torchvision not installed"}
            continue
        t0 = time.time()
        m = train(good, coreset=coreset, seed=seed, backend=b)
        t_train = time.time() - t0
        t0 = time.time()
        g = np.array([m.score(im) for im in test_good])
        t_score = (time.time() - t0) / max(len(test_good), 1)
        per = {}
        for cls, ims in test_bad.items():
            sc = np.array([m.score(im) for im in ims])
            per[cls] = {"auroc": auroc(g, sc), "detection_rate": float((sc > m.threshold).mean()) if len(sc) else None}
        out[b] = {"n_train": len(good), "feature_dim": int(m.bank.shape[1]), "bank": int(m.bank.shape[0]),
                  "train_s": round(t_train, 2), "score_ms_per_image": round(1000 * t_score, 1),
                  "good_fp_rate": float((g > m.threshold).mean()) if len(g) else None, "per_class": per,
                  "model_bytes": int(m.bank.astype(np.float32).nbytes)}
    return out
