#!/usr/bin/env python3
"""Unit checks for the MuAt-comparator architecture-search harness guard.

Run from the repo root:  python scripts/test_muat_divergence_guard.py

These test the two pure decision rules the runner uses (``_loss_diverged`` and
``_ranked_finite_candidates``) plus the fingerprint invariant that an unguarded config must hash
identically before and after the guard code was added. They do not train anything; the full
smoke run (which needs the MC3 MAF) is
    python src/run_all_experiments.py --config config/experiment_settings.muat_arch_search_smoke.yaml ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from runners.run_muat_style_tcga_comparator import _loss_diverged, _ranked_finite_candidates  # noqa: E402

FAILURES = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global FAILURES
    tag = "ok  " if cond else "FAIL"
    if not cond:
        FAILURES += 1
    print(f"  [{tag}] {label}" + (f"   {detail}" if detail else ""))


def _first_trip(trajectory: list[float], *, margin: float = 0.5, warmup: int = 3) -> int | None:
    min_so_far = np.inf
    for epoch, loss in enumerate(trajectory, start=1):
        if _loss_diverged(loss, min_so_far, epoch, margin=margin, warmup_epochs=warmup):
            return epoch
        if np.isfinite(loss):
            min_so_far = min(min_so_far, loss)
    return None


def test_loss_diverged() -> None:
    print("_loss_diverged (additive margin, nats)")
    check("margin=0 disables", not _loss_diverged(float("nan"), 0.01, 10, margin=0.0, warmup_epochs=3))
    check("nan trips at epoch 1", _loss_diverged(float("nan"), np.inf, 1, margin=0.5, warmup_epochs=3))
    check("inf trips at epoch 1", _loss_diverged(float("inf"), np.inf, 1, margin=0.5, warmup_epochs=3))
    check("rise inside warm-up ignored", not _loss_diverged(5.0, 0.01, 3, margin=0.5, warmup_epochs=3))
    check("no baseline -> no trip", not _loss_diverged(5.0, np.inf, 10, margin=0.5, warmup_epochs=3))
    check("+0.49 does not trip", not _loss_diverged(0.59, 0.10, 10, margin=0.5, warmup_epochs=3))
    check("+0.51 trips", _loss_diverged(0.61, 0.10, 10, margin=0.5, warmup_epochs=3))

    # --- Real trajectories from run 20261002T202415Z (fold 3, v1 ratio guard), used as the
    # --- acceptance tests for v2. Values are the logged per-epoch training losses.

    # Candidate 5 (128/4/1): healthy. Loss fell to 0.0006 by epoch 110 and read 0.0019 at epoch
    # 118. The v1 ratio rule killed it there (0.0019 > 3 x 0.0006). v2 must NOT trip.
    cand5 = [2.3300, 2.0114, 1.8455, 1.7158, 1.5911, 1.4837, 1.3749, 1.2994, 1.2071, 1.1486,
             1.1457, 1.0586, 1.0309, 0.9781, 0.9424, 0.8291, 0.4172, 0.2902, 0.1383, 0.0517,
             0.0242, 0.0052, 0.0016, 0.0020, 0.0006, 0.0019]
    check("v1-false-positive candidate 5 does NOT trip", _first_trip(cand5) is None, f"trip={_first_trip(cand5)}")

    # Candidate 11 (256/4/1): genuine divergence. Loss bottomed at 1.5461 (epoch 7) then rose to
    # 2.6748 by epoch 11 with validation at chance (0.05). The v1 ratio rule never tripped
    # (3 x 1.55 = 4.6 > ln 20). v2 must trip, and must do so by epoch 11 at the latest.
    cand11 = [2.3403, 2.0597, 1.8806, 1.7622, 1.6766, 1.6567, 1.5461, 1.5998, 1.7458, 2.2797,
              2.6748, 2.6739, 2.5997, 2.6640, 2.7753]
    t11 = _first_trip(cand11)
    check("v1-false-negative candidate 11 DOES trip", t11 is not None and t11 <= 11, f"trip={t11} (2.2797 - 1.5461 = 0.73 > 0.5 at epoch 10)")

    # 20 Aug fold 3 shape: descended well below 1 then climbed back to 1.658. Must trip.
    aug = [2.95, 2.40, 1.80, 1.10, 0.60, 0.25, 0.08, 0.05, 0.30, 0.90, 1.66]
    check("20 Aug fold-3 shape trips", _first_trip(aug) == 10, f"trip={_first_trip(aug)} (0.90 - 0.05 = 0.85)")

    # Healthy run ending in the 0.003-0.020 band with ordinary noise: never trips.
    healthy = [2.95, 2.1, 1.4, 0.9, 0.5, 0.3, 0.2, 0.15, 0.17, 0.12, 0.13, 0.10, 0.11, 0.09, 0.02, 0.004, 0.012, 0.003]
    check("healthy trajectory never trips", _first_trip(healthy) is None)

    # Scale sanity: a regression pass on z-scored targets (MSE ~ 1 at init) with a 0.3 wobble
    # at the floor does not trip; a return to initial loss does.
    reg_ok = [1.05, 0.80, 0.60, 0.45, 0.40, 0.38, 0.41, 0.36, 0.39]
    reg_bad = [1.05, 0.80, 0.60, 0.45, 0.40, 0.38, 0.70, 1.00, 1.10]
    check("regression wobble does not trip", _first_trip(reg_ok) is None)
    check("regression return-to-init trips", _first_trip(reg_bad) is not None)


def test_ranked_candidates() -> None:
    print("_ranked_finite_candidates")
    results = [
        {"stage": "architecture_search", "candidate_id": 1, "selected_inner_score": 0.55},
        {"stage": "architecture_search", "candidate_id": 2, "selected_inner_score": 0.58},
        {"stage": "architecture_search", "candidate_id": 3, "selected_inner_score": float("-inf")},  # diverged
        {"stage": "architecture_search", "candidate_id": 4, "selected_inner_score": 0.58},           # tie with 2
        {"stage": "architecture_search", "candidate_id": 5, "selected_inner_score": float("nan")},
        {"stage": "final_epoch_selection", "candidate_id": 2, "selected_inner_score": 0.99},          # must be ignored
    ]
    ranked = _ranked_finite_candidates(results)
    ids = [cid for _s, cid in ranked]
    check("diverged (-inf) excluded", 3 not in ids, f"ids={ids}")
    check("nan excluded", 5 not in ids)
    check("final_epoch_selection rows ignored", all(s < 0.99 for s, _ in ranked))
    check("best first, tie -> lower id", ids[:2] == [2, 4], f"ids={ids}")
    check("full order", ids == [2, 4, 1], f"ids={ids}")
    check("empty when all diverged", _ranked_finite_candidates([{"stage": "architecture_search", "candidate_id": 1, "selected_inner_score": float("-inf")}]) == [])


def test_fingerprint_invariant() -> None:
    """The guard keys are only added to the fingerprint payload when the guard is on, so the
    20 Aug 2026 run's fingerprint must be unaffected by this code change. We cannot rebuild the
    full payload here (it digests the encoded event tensors), so instead assert the two
    properties the invariant depends on, read straight from the runner source."""
    print("fingerprint invariant")
    src = (REPO / "src/runners/run_muat_style_tcga_comparator.py").read_text(encoding="utf-8")
    body = src[src.index("fingerprint_payload = {") : src.index("fingerprint = stable_sha256(fingerprint_payload)")]
    check("guard keys added conditionally", "if _guard_margin > 0:" in body)
    literal_block = body[: body.index("}")]  # the dict literal itself
    check("no guard key inside the unconditional dict literal", "divergence_loss_margin" not in literal_block)
    # Also confirm the recorded 20 Aug fingerprint is the one the fold-3 config opts into.
    cfg = (REPO / "config/experiment_settings.muat_top20_fold3_rerun.yaml").read_text(encoding="utf-8")
    check("fold-3 config references the 20 Aug fingerprint", "876a04d478138629869b972af2fbce20aae789de6d48bd25260c827adf213f5e" in cfg)


def test_no_gradient_clipping() -> None:
    print("fidelity: no gradient clipping in the training step")
    src = (REPO / "src/runners/run_muat_style_tcga_comparator.py").read_text(encoding="utf-8")
    check("clip_grad_norm_ not called anywhere", "clip_grad_norm_(" not in src)


if __name__ == "__main__":
    test_loss_diverged()
    test_ranked_candidates()
    test_fingerprint_invariant()
    test_no_gradient_clipping()
    print()
    if FAILURES:
        print(f"{FAILURES} check(s) FAILED")
        sys.exit(1)
    print("all checks passed")
