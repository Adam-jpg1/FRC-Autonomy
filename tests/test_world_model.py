"""Tests for field-space tracking: dedupe, velocity, aging, alliance routing."""

import pytest

from frc_perception.config import WorldModelConfig
from frc_perception.field_map import FieldMap
from frc_perception.world_model import FieldDetection, WorldModel, positions


def make_field() -> FieldMap:
    # Nonexistent path -> fallback dimensions; that's all the tests need.
    return FieldMap("/nonexistent/layout.json", 16.0, 8.0)


def make_model(**overrides) -> WorldModel:
    cfg = WorldModelConfig(**overrides)
    return WorldModel(cfg, make_field())


def fuel(x: float, y: float) -> FieldDetection:
    return FieldDetection("fuel", "game_piece", x, y, 0.9)


class TestDedupe:
    def test_two_cameras_one_ball_one_track(self):
        model = make_model(min_hits_to_confirm=1)
        # The same ball seen by both cameras with slight disagreement.
        model.update([fuel(4.0, 3.0), fuel(4.2, 3.1)], now=1.0)
        assert len(model.tracks) == 1

    def test_distant_balls_stay_separate(self):
        model = make_model(min_hits_to_confirm=1)
        model.update([fuel(4.0, 3.0), fuel(7.0, 3.0)], now=1.0)
        assert len(model.tracks) == 2

    def test_different_classes_never_merge(self):
        model = make_model(min_hits_to_confirm=1)
        model.update(
            [fuel(4.0, 3.0), FieldDetection("bumper_red", "robot", 4.1, 3.0, 0.9)],
            now=1.0,
        )
        assert len(model.tracks) == 2


class TestVelocity:
    def test_constant_velocity_is_estimated_unbiased(self):
        model = make_model(min_hits_to_confirm=1)
        # Ball rolling +1.0 m/s in x, observed at 30 Hz.
        for i in range(60):
            t = i / 30.0
            model.update([fuel(2.0 + 1.0 * t, 3.0)], now=t)
        (track,) = model.game_pieces()
        assert track.vx == pytest.approx(1.0, abs=0.05)
        assert track.vy == pytest.approx(0.0, abs=0.05)

    def test_teleport_speed_is_rejected_for_game_pieces(self):
        model = make_model(min_hits_to_confirm=1, max_speed_game_piece_mps=6.0,
                           gate_game_piece_m=10.0)
        model.update([fuel(2.0, 3.0)], now=0.0)
        model.update([fuel(3.0, 3.0)], now=0.033)  # ~30 m/s: bogus association
        (track,) = model.game_pieces()
        assert track.vx == pytest.approx(0.0)


class TestAging:
    def test_stale_track_ages_out(self):
        model = make_model(min_hits_to_confirm=1, track_timeout_game_piece_s=1.0)
        model.update([fuel(4.0, 3.0)], now=0.0)
        model.prune(now=0.5)
        assert len(model.tracks) == 1
        model.prune(now=1.5)
        assert len(model.tracks) == 0

    def test_prune_only_never_creates_tracks(self):
        # This is what the app calls when the pose is invalid.
        model = make_model()
        model.prune(now=5.0)
        assert model.tracks == []


class TestFiltering:
    def test_detection_outside_field_rejected(self):
        model = make_model(min_hits_to_confirm=1, field_margin_m=0.25)
        model.update([fuel(-2.0, 3.0)], now=1.0)
        assert model.tracks == []

    def test_unconfirmed_track_not_reported(self):
        model = make_model(min_hits_to_confirm=2)
        model.update([fuel(4.0, 3.0)], now=1.0)
        assert model.game_pieces() == []
        model.update([fuel(4.0, 3.0)], now=1.05)
        assert len(model.game_pieces()) == 1


class TestAllianceRouting:
    def seed(self, model: WorldModel) -> None:
        model.update(
            [
                FieldDetection("bumper_red", "robot", 5.0, 3.0, 0.9),
                FieldDetection("bumper_blue", "robot", 9.0, 3.0, 0.9),
                fuel(4.0, 4.0),
            ],
            now=1.0,
        )

    def test_red_alliance_reports_blue_robots(self):
        model = make_model(min_hits_to_confirm=1)
        self.seed(model)
        opponents = model.opponents(is_red_alliance=True)
        assert [o.class_name for o in opponents] == ["bumper_blue"]

    def test_blue_alliance_reports_red_robots(self):
        model = make_model(min_hits_to_confirm=1)
        self.seed(model)
        opponents = model.opponents(is_red_alliance=False)
        assert [o.class_name for o in opponents] == ["bumper_red"]

    def test_unknown_alliance_reports_nobody(self):
        model = make_model(min_hits_to_confirm=1)
        self.seed(model)
        assert model.opponents(is_red_alliance=None) == []


def test_positions_helper():
    model = make_model(min_hits_to_confirm=1)
    model.update([fuel(1.0, 2.0), fuel(5.0, 6.0)], now=1.0)
    xs, ys = positions(model.game_pieces())
    assert sorted(zip(xs, ys)) == [(1.0, 2.0), (5.0, 6.0)]
