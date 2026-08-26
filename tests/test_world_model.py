"""World model: cross-camera de-duplication, association, velocity, aging."""

import math

import pytest

from frc_perception.config import CLASS_BUMPER_RED, CLASS_FUEL, WorldModelConfig
from frc_perception.field_map import FieldMap
from frc_perception.types import FieldObservation, ObjectRole, Track, TrackFreshness
from frc_perception.world_model import WorldModel, fit_velocity, merge_observations


def fuel_observation(x, y, timestamp=0.0, camera="cam0", confidence=0.9, quality=1.0):
    return FieldObservation(
        camera_id=camera,
        timestamp=timestamp,
        class_name=CLASS_FUEL,
        role=ObjectRole.GAME_PIECE,
        confidence=confidence,
        x_field_m=x,
        y_field_m=y,
        range_m=math.hypot(x, y),
        quality=quality,
    )


def robot_observation(x, y, timestamp=0.0, camera="cam0", role=ObjectRole.OPPONENT):
    return FieldObservation(
        camera_id=camera,
        timestamp=timestamp,
        class_name=CLASS_BUMPER_RED,
        role=role,
        confidence=0.85,
        x_field_m=x,
        y_field_m=y,
        range_m=math.hypot(x, y),
    )


def make_model(**overrides) -> WorldModel:
    defaults = dict(min_hits_to_confirm=1)
    defaults.update(overrides)
    return WorldModel(WorldModelConfig(**defaults))


# ---------------------------------------------------------------------------
# Cross-camera de-duplication
# ---------------------------------------------------------------------------


def test_two_cameras_seeing_one_ball_produce_one_observation():
    merged = merge_observations(
        [fuel_observation(4.0, 2.0, camera="cam0"), fuel_observation(4.15, 2.05, camera="cam1")],
        0.40, 0.90,
    )
    assert len(merged) == 1
    assert merged[0].merged_count == 2
    assert "cam0" in merged[0].camera_id and "cam1" in merged[0].camera_id
    assert 4.0 <= merged[0].x_field_m <= 4.15


def test_separated_balls_stay_separate():
    merged = merge_observations(
        [fuel_observation(4.0, 2.0), fuel_observation(6.0, 2.0)], 0.40, 0.90
    )
    assert len(merged) == 2


def test_merging_is_transitive_across_a_chain():
    """Three overlapping sightings are one object, not two."""
    merged = merge_observations(
        [fuel_observation(0.0, 0.0), fuel_observation(0.3, 0.0), fuel_observation(0.6, 0.0)],
        0.40, 0.90,
    )
    assert len(merged) == 1


def test_different_classes_never_merge():
    merged = merge_observations(
        [fuel_observation(4.0, 2.0), robot_observation(4.05, 2.0)], 0.40, 0.90
    )
    assert len(merged) == 2


def test_merged_position_is_pulled_toward_the_more_confident_sighting():
    merged = merge_observations(
        [
            fuel_observation(4.0, 0.0, confidence=0.95),
            fuel_observation(4.3, 0.0, confidence=0.20),
        ],
        0.40, 0.90,
    )
    assert len(merged) == 1
    assert merged[0].x_field_m < 4.15
    assert merged[0].confidence == pytest.approx(0.95)


def test_robots_use_the_larger_merge_radius():
    """A 0.6 m gap merges for robots but not for game pieces."""
    assert len(merge_observations([robot_observation(4.0, 2.0), robot_observation(4.6, 2.0)], 0.40, 0.90)) == 1
    assert len(merge_observations([fuel_observation(4.0, 2.0), fuel_observation(4.6, 2.0)], 0.40, 0.90)) == 2


# ---------------------------------------------------------------------------
# Track creation and association
# ---------------------------------------------------------------------------


def test_a_new_observation_creates_a_track():
    model = make_model()
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0)], now=1.0)
    tracks = model.game_pieces(1.0)
    assert len(tracks) == 1
    assert tracks[0].hit_count == 1
    assert tracks[0].class_name == CLASS_FUEL


def test_two_cameras_seeing_one_ball_create_one_track():
    model = make_model()
    model.update(
        [
            fuel_observation(4.0, 2.0, timestamp=1.0, camera="cam0"),
            fuel_observation(4.1, 2.02, timestamp=1.0, camera="cam1"),
        ],
        now=1.0,
    )
    assert len(model.game_pieces(1.0)) == 1


def test_a_repeated_sighting_updates_rather_than_duplicates():
    model = make_model()
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0)], now=1.0)
    model.update([fuel_observation(4.05, 2.0, timestamp=1.1)], now=1.1)
    tracks = model.game_pieces(1.1)
    assert len(tracks) == 1
    assert tracks[0].hit_count == 2


def test_an_observation_outside_the_gate_starts_a_new_track():
    model = make_model(gate_game_piece_m=0.3)
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0)], now=1.0)
    model.update([fuel_observation(6.0, 2.0, timestamp=1.1)], now=1.1)
    assert len(model.game_pieces(1.1)) == 2


def test_unconfirmed_tracks_are_not_reported():
    model = WorldModel(WorldModelConfig(min_hits_to_confirm=2))
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0)], now=1.0)
    assert model.game_pieces(1.0) == []
    assert len(model.all_tracks) == 1
    model.update([fuel_observation(4.0, 2.0, timestamp=1.05)], now=1.05)
    assert len(model.game_pieces(1.05)) == 1


def test_observations_outside_the_field_are_dropped():
    field_map = FieldMap(tags={}, length_m=16.0, width_m=8.0)
    model = WorldModel(WorldModelConfig(min_hits_to_confirm=1, field_margin_m=0.5), field_map)
    model.update([fuel_observation(40.0, 2.0, timestamp=1.0)], now=1.0)
    assert model.all_tracks == []


def test_role_is_updated_when_alliance_information_arrives():
    """A robot first seen with an unknown alliance must become an opponent
    once the FMS data shows up, not stay unclassified forever."""
    model = make_model()
    model.update([robot_observation(4.0, 2.0, timestamp=1.0, role=ObjectRole.UNKNOWN)], 1.0)
    model.update([robot_observation(4.0, 2.0, timestamp=1.1, role=ObjectRole.OPPONENT)], 1.1)
    assert len(model.opponents(1.1)) == 1


def test_obstacle_list_includes_unidentifiable_robots_but_not_friendlies():
    model = make_model()
    model.update(
        [
            robot_observation(4.0, 2.0, timestamp=1.0, role=ObjectRole.UNKNOWN),
            robot_observation(8.0, 2.0, timestamp=1.0, role=ObjectRole.FRIENDLY),
            robot_observation(12.0, 2.0, timestamp=1.0, role=ObjectRole.OPPONENT),
        ],
        now=1.0,
    )
    obstacles = model.obstacle_robots(1.0)
    roles = {t.role for t in obstacles}
    assert ObjectRole.FRIENDLY not in roles
    assert roles == {ObjectRole.UNKNOWN, ObjectRole.OPPONENT}


def test_unknown_robots_can_be_excluded_by_configuration():
    model = make_model(treat_unknown_robots_as_obstacles=False)
    model.update([robot_observation(4.0, 2.0, timestamp=1.0, role=ObjectRole.UNKNOWN)], 1.0)
    assert model.obstacle_robots(1.0) == []


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------


def test_fit_velocity_recovers_a_known_constant_velocity():
    samples = [(index * 0.1, 1.0 + 2.0 * index * 0.1, 5.0 - 0.5 * index * 0.1) for index in range(6)]
    vx, vy, valid = fit_velocity(samples, min_samples=3, min_span_s=0.15, max_speed_mps=6.0)
    assert valid
    assert pytest.approx(2.0, abs=1e-9) == vx
    assert pytest.approx(-0.5, abs=1e-9) == vy


def test_fit_velocity_needs_enough_samples_and_time_span():
    assert fit_velocity([(0.0, 0.0, 0.0), (0.1, 1.0, 0.0)], 3, 0.15, 6.0)[2] is False
    close = [(0.0, 0.0, 0.0), (0.01, 0.01, 0.0), (0.02, 0.02, 0.0)]
    assert fit_velocity(close, 3, 0.15, 6.0)[2] is False


def test_absurd_velocity_is_refused_rather_than_reported():
    samples = [(index * 0.1, index * 10.0, 0.0) for index in range(4)]
    vx, vy, valid = fit_velocity(samples, 3, 0.15, max_speed_mps=6.0)
    assert not valid and vx == 0.0 and vy == 0.0


def test_track_velocity_is_unbiased_by_position_smoothing():
    """A ball moving at a known speed must report that speed.

    Position is smoothed with an EMA; differencing THAT would report a
    speed biased low. Fitting the raw measurements does not.
    """
    model = make_model(position_alpha=0.5, velocity_min_samples=3, velocity_min_span_s=0.1)
    speed = 1.5
    for index in range(8):
        timestamp = 1.0 + index * 0.1
        model.update([fuel_observation(2.0 + speed * index * 0.1, 3.0, timestamp)], timestamp)
    track = model.game_pieces(1.7)[0]
    assert track.velocity_valid
    assert pytest.approx(speed, rel=0.02) == track.vx
    assert pytest.approx(0.0, abs=1e-6) == track.vy


def test_velocity_sign_follows_the_direction_of_travel():
    model = make_model(velocity_min_samples=3, velocity_min_span_s=0.1)
    for index in range(6):
        timestamp = 1.0 + index * 0.1
        model.update([fuel_observation(8.0 - 1.0 * index * 0.1, 3.0, timestamp)], timestamp)
    track = model.game_pieces(1.5)[0]
    assert track.vx < 0.0


def test_a_stationary_object_reports_no_velocity():
    model = make_model(velocity_min_samples=3, velocity_min_span_s=0.1)
    for index in range(6):
        timestamp = 1.0 + index * 0.1
        model.update([fuel_observation(4.0, 2.0, timestamp)], timestamp)
    track = model.game_pieces(1.5)[0]
    assert abs(track.vx) < 1e-9 and abs(track.vy) < 1e-9


def test_track_prediction_uses_constant_velocity():
    track = Track(1, CLASS_FUEL, ObjectRole.GAME_PIECE, x=1.0, y=2.0, vx=2.0, vy=0.0,
                  last_seen=10.0, velocity_valid=True)
    assert track.predict(10.5, 1.0) == (2.0, 2.0)
    # Prediction is clamped so a long-lost track is not flung across the field.
    assert track.predict(100.0, 1.0) == (3.0, 2.0)


def test_prediction_is_the_last_position_without_a_valid_velocity():
    track = Track(1, CLASS_FUEL, ObjectRole.GAME_PIECE, x=1.0, y=2.0, vx=9.0, vy=9.0,
                  last_seen=10.0, velocity_valid=False)
    assert track.predict(11.0, 1.0) == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Aging
# ---------------------------------------------------------------------------


def test_a_track_ages_out_after_the_expiry_timeout():
    model = make_model(expire_timeout_game_piece_s=1.0)
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0)], now=1.0)
    assert len(model.game_pieces(1.0)) == 1
    model.update([], now=1.5)
    assert len(model.all_tracks) == 1, "still aging, not yet expired"
    model.update([], now=2.5)
    assert model.all_tracks == []


def test_freshness_classifies_recent_aging_and_expired():
    model = make_model(fresh_timeout_game_piece_s=0.5, expire_timeout_game_piece_s=2.0)
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0)], now=1.0)
    track = model.all_tracks[0]
    assert model.freshness(track, 1.2) is TrackFreshness.RECENT
    assert model.freshness(track, 1.9) is TrackFreshness.AGING
    assert model.freshness(track, 4.0) is TrackFreshness.EXPIRED


def test_confidence_decays_as_a_track_gets_stale():
    model = make_model(expire_timeout_game_piece_s=2.0, min_report_confidence=0.0)
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0, confidence=1.0)], now=1.0)
    model.update([], now=2.0)
    track = model.all_tracks[0]
    assert 0.4 < track.confidence < 0.6


def test_a_faded_track_stops_being_reported_before_it_expires():
    model = make_model(expire_timeout_game_piece_s=2.0, min_report_confidence=0.6)
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0, confidence=1.0)], now=1.0)
    model.update([], now=1.9)
    assert model.game_pieces(1.9) == []
    assert len(model.all_tracks) == 1


def test_clear_drops_everything():
    model = make_model()
    model.update([fuel_observation(4.0, 2.0, timestamp=1.0)], now=1.0)
    model.clear()
    assert model.all_tracks == []
