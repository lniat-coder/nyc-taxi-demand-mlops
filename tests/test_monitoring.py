import numpy as np

from taxi.config import policy
from taxi.monitoring.drift import decide
from taxi.monitoring.psi import level, make_profile, psi, psi_from_shares
from taxi.pipelines.check_and_retrain import retrain_windows


def test_psi_by_hand_matches_the_course_example():
    # session 5, "PSI by hand": training 0.5/0.3/0.2, live 0.3/0.3/0.4 -> 0.241
    assert abs(psi_from_shares([0.5, 0.3, 0.2], [0.3, 0.3, 0.4]) - 0.241) < 1e-3


def test_psi_is_small_on_the_same_distribution_and_large_on_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.poisson(100, 20_000)
    profile = make_profile(ref)
    assert psi(profile, rng.poisson(100, 5_000)) < 0.02
    assert psi(profile, rng.poisson(10, 5_000)) > 0.25   # lockdown-like collapse


def test_level():
    assert [level(v, 0.10, 0.25) for v in (0.05, 0.2, 0.4)] == ["stable", "watch", "act"]


def test_decision_tree_order():
    pol = policy()
    assert decide([{"check": "silent_zone"}], 0.9, 3.0, 30, True, pol)[0] == "FIX_PIPELINE"
    assert decide([], 0.05, 1.0, 30, True, pol)[0] == "ROLLBACK"
    assert decide([], 0.05, 1.0, 30, False, pol)[0] == "OK"
    assert decide([], 0.20, 1.0, 30, False, pol)[0] == "WATCH"
    assert decide([], 0.90, 1.05, 30, False, pol)[0] == "WATCH"          # drift that does not hurt
    assert decide([], 0.90, 2.00, 5, False, pol)[0] == "WAIT_FOR_DATA"
    assert decide([], 0.90, 2.00, 30, False, pol)[0] == "RETRAIN"


def test_retrain_windows():
    assert retrain_windows("2020-04-30", train_days=35, holdout_days=7) == (
        "2020-03-20", "2020-04-23", "2020-04-24", "2020-04-30")
