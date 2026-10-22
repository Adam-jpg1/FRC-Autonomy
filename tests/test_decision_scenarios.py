"""Runs every built-in decision scenario as part of the normal test suite.

``test_decisions.py`` is the interactive/CLI harness; this file makes its
scenarios permanent regressions so they run under plain ``pytest`` too.
Each scenario becomes one parametrized test, named after the scenario, so a
failure says which decision changed.

The harness drives the real pipeline, so these are integration tests over
the actual scheduler, ranging, localization, world model, navigation,
controller and NetworkTables payload -- not a parallel model.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_decisions as harness  # noqa: E402

SCENARIOS = harness.builtin_scenarios()
OFFLINE_SCENARIOS = [s for s in SCENARIOS if not s.requires_gpu]


def run(spec):
    """Execute one scenario and return (safety + declared) check results."""
    runner = harness.DecisionHarness(spec)
    traces = runner.run()
    return harness.safety_checks(traces) + harness.check_expectations(spec, traces)


@pytest.mark.parametrize("spec", OFFLINE_SCENARIOS, ids=lambda s: s.name)
def test_scenario(spec):
    failures = [r for r in run(spec) if not r.passed]
    assert not failures, "\n".join(f"{r.key}: {r.detail}" for r in failures)


def test_every_scenario_declares_expectations():
    """A scenario that asserts nothing is not a test."""
    silent = [s.name for s in SCENARIOS if not s.expect]
    assert not silent, f"scenarios with no expectations: {silent}"


def test_all_builtin_scenarios_run_offline():
    """None of the built-ins may require a GPU; that is what --gpu-check is for."""
    assert [s.name for s in SCENARIOS if s.requires_gpu] == []


def test_expectation_keys_are_all_documented():
    """Every key a scenario uses must appear in EXPECTATION_KEYS."""
    used = {key for spec in SCENARIOS for key in spec.expect}
    undocumented = sorted(used - set(harness.EXPECTATION_KEYS))
    assert not undocumented, f"undocumented expectation keys: {undocumented}"


def test_unknown_expectation_keys_fail_rather_than_pass_silently():
    """A typo in an expectation must FAIL, not be ignored."""
    spec = harness.ScenarioSpec(
        name="typo", cameras=["cam0"],
        cycles=[harness.CycleSpec()],
        expect={"pose_vaild": True},  # deliberate typo
    )
    results = harness.check_expectations(spec, [harness.CycleTrace(index=0, clock=0.0)])
    assert results and not results[0].passed
    assert "unknown expectation" in results[0].detail


# ---------------------------------------------------------------------------
# The harness's own safety guarantees
# ---------------------------------------------------------------------------


def test_harness_refuses_a_non_simulated_networktables_backend():
    """The guard that stops this ever reaching a real robot network."""
    spec = harness.ScenarioSpec(
        name="safety", cameras=["cam0"], cycles=[harness.CycleSpec()],
        expect={"controller_enabled": False},
    )
    runner = harness.DecisionHarness(spec)

    class PretendRealBackend:
        connected = True

        def connect(self):
            pass

        def close(self):
            pass

    runner.app.nt = PretendRealBackend()
    with pytest.raises(SystemExit, match="REFUSING TO RUN"):
        harness.assert_offline(runner.app)


def test_harness_refuses_an_enabled_virtual_controller():
    spec = harness.ScenarioSpec(
        name="safety", cameras=["cam0"], cycles=[harness.CycleSpec()],
        expect={"controller_enabled": False},
    )
    runner = harness.DecisionHarness(spec)
    runner.app.cfg.controller.enabled = True
    with pytest.raises(SystemExit, match="REFUSING TO RUN"):
        harness.assert_offline(runner.app)


def test_a_scenario_cannot_enable_the_controller_via_config_override():
    """SAFETY: the harness re-asserts the disable AFTER applying overrides."""
    spec = harness.ScenarioSpec(
        name="sneaky", cameras=["cam0"], cycles=[harness.CycleSpec()],
        config={"controller.enabled": True},
        expect={"controller_enabled": False},
    )
    runner = harness.DecisionHarness(spec)
    assert runner.cfg.controller.enabled is False


def test_simulated_networktables_stores_instead_of_transmitting():
    """Confirms published values go to a dict, not a socket."""
    from frc_perception.nt_io import SimulatedNT

    backend = SimulatedNT()
    backend.publish_perception({"robot_x": 1.0})
    assert backend.published_perception == {"robot_x": 1.0}
    # No connection, send, flush-to-network or socket attribute exists.
    for attribute in ("_instance", "_publishers", "_subscribers"):
        assert not hasattr(backend, attribute)


def test_safety_checks_catch_translation_without_a_valid_pose():
    """The invariant engine must actually fire, not just exist."""
    bad = harness.CycleTrace(index=0, clock=0.0)
    bad.pose_valid = False
    bad.nav_vx = 1.0
    failures = [r for r in harness.safety_checks([bad]) if not r.passed]
    assert any("no_translation_without_pose" in r.key for r in failures)


def test_safety_checks_catch_a_live_controller():
    bad = harness.CycleTrace(index=0, clock=0.0)
    bad.controller_enabled = True
    bad.controller_axes = (0.5, 0.0, 0.0)
    failures = [r for r in harness.safety_checks([bad]) if not r.passed]
    assert any("controller_disabled" in r.key for r in failures)
    assert any("axes_neutral" in r.key for r in failures)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_the_example_json_is_a_valid_runnable_scenario():
    specs = [
        harness.ScenarioSpec.from_dict(entry)
        for entry in harness.EXAMPLE_JSON["scenarios"]
    ]
    assert specs
    for spec in specs:
        failures = [r for r in run(spec) if not r.passed]
        assert not failures, "\n".join(f"{r.key}: {r.detail}" for r in failures)


def test_a_json_scenario_without_a_name_is_rejected():
    with pytest.raises(ValueError, match="name"):
        harness.ScenarioSpec.from_dict({"description": "no name"})


def test_an_unknown_config_override_is_rejected():
    from frc_perception.config import AppConfig

    with pytest.raises(ValueError, match="unknown config field"):
        harness._set_dotted(AppConfig(), "navigation.no_such_field", 1.0)


# ---------------------------------------------------------------------------
# Fixture math must agree with the production ranging path
# ---------------------------------------------------------------------------


def test_projection_fixture_agrees_with_production_ranging():
    """If this drifts, every scenario's object placement is wrong.

    Projects known robot-frame points to pixel boxes, then runs the real
    estimate_ground_object() on them and requires the position back.
    """
    from frc_perception.config import RangingConfig
    from frc_perception.geometry import estimate_ground_object

    cam = harness.make_camera("cam0")
    ranging = RangingConfig()
    for x, y in [(1.5, 0.0), (3.0, 0.5), (4.0, -0.8), (2.0, 1.0), (5.0, 0.0)]:
        bbox = harness.bbox_for_ground_point(cam, x, y)
        assert bbox is not None, f"({x}, {y}) should be visible to cam0"
        result = estimate_ground_object(bbox, cam, ranging)
        assert result.valid, f"({x}, {y}) -> {result.rejection}"
        assert result.x_robot_m == pytest.approx(x, abs=0.02)
        assert result.y_robot_m == pytest.approx(y, abs=0.02)


def test_points_outside_the_field_of_view_are_reported_not_silently_dropped():
    """A scenario that stages an invisible object must fail, loudly."""
    cam = harness.make_camera("cam0")
    assert harness.bbox_for_ground_point(cam, 1.0, 5.0) is None
    assert harness.project_ground_point(cam, -3.0, 0.0) is None
