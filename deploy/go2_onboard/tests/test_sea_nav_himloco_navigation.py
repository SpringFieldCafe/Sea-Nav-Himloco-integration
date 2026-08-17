import numpy as np
import pytest

from deploy.go2_onboard.sea_nav_himloco_navigation import (
    HIM_1460_SHA256,
    NAV_LOWER,
    NAV_UPPER,
    NavigationLimiter,
    NavigationMailbox,
    goal_is_reached,
    validate_navigation_model,
)


def test_navigation_model_contract_and_hash():
    loaded = validate_navigation_model(
        "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt",
        "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json",
    )
    assert loaded.sha256 == "d1242c74ff56189f20d7a12948d86078287651cd1f70215d9c308d04f4b561de"


def test_navigation_limits_allow_forward_arc_and_force_no_lateral_motion():
    limiter = NavigationLimiter(filter_alpha=1.0)
    raw, limited, safe = limiter.apply([0.4, 0.8, -0.4])
    np.testing.assert_allclose(raw, [0.4, 0.8, -0.4])
    np.testing.assert_allclose(limited, [0.15, 0.0, -0.15])
    np.testing.assert_allclose(safe, [0.15, 0.0, -0.15])
    np.testing.assert_allclose(NAV_LOWER, [0.0, 0.0, -0.15])
    np.testing.assert_allclose(NAV_UPPER, [0.15, 0.0, 0.15])


def test_navigation_limiter_rejects_nonfinite_output():
    with pytest.raises(FloatingPointError, match="finite 3-vector"):
        NavigationLimiter().apply([0.0, float("nan"), 0.0])


def test_navigation_stale_mailbox_fails_closed():
    mailbox = NavigationMailbox(max_age=0.1)
    command, reason, age = mailbox.current()
    assert command.as_array().tolist() == [0.0, 0.0, 0.0]
    assert reason == "navigation_command_stale"
    assert age == float("inf")

    mailbox.update({"command": [0.1, 0.0, -0.1], "runtime_state": "NAVIGATION"})
    command, reason, age = mailbox.current()
    np.testing.assert_allclose(command.as_array(), [0.1, 0.0, -0.1])
    assert reason == ""
    assert 0.0 <= age < 0.1


def test_goal_tolerance_is_explicit():
    assert goal_is_reached([0.2, 0.2], 0.3)
    assert not goal_is_reached([0.3, 0.1], 0.3)


def test_navigation_requires_the_1460_himloco_hash():
    assert HIM_1460_SHA256 == "cab2489dda7732a7d6f51595aa6362384445c738537d0c1c91569054c7b9f5d1"
