"""Alliance logic: which bumper color is an opponent.

Getting this backwards makes the robot treat teammates as obstacles and
drive through opponents, so every branch is pinned down explicitly.
"""

import pytest

from frc_perception.alliance import alliance_from_is_red, is_obstacle, role_for_class
from frc_perception.config import CLASS_BUMPER_BLUE, CLASS_BUMPER_RED, CLASS_FUEL
from frc_perception.types import Alliance, ObjectRole


# ---------------------------------------------------------------------------
# Reading FMSInfo/IsRedAlliance
# ---------------------------------------------------------------------------


def test_true_means_we_are_red():
    assert alliance_from_is_red(True) is Alliance.RED


def test_false_means_we_are_blue():
    assert alliance_from_is_red(False) is Alliance.BLUE


def test_missing_value_is_unknown_and_is_not_guessed():
    assert alliance_from_is_red(None) is Alliance.UNKNOWN


# ---------------------------------------------------------------------------
# Class -> role
# ---------------------------------------------------------------------------


def test_red_alliance_blue_bumpers_are_opponents():
    assert role_for_class(CLASS_BUMPER_BLUE, Alliance.RED) is ObjectRole.OPPONENT


def test_red_alliance_red_bumpers_are_friendly():
    assert role_for_class(CLASS_BUMPER_RED, Alliance.RED) is ObjectRole.FRIENDLY


def test_blue_alliance_red_bumpers_are_opponents():
    assert role_for_class(CLASS_BUMPER_RED, Alliance.BLUE) is ObjectRole.OPPONENT


def test_blue_alliance_blue_bumpers_are_friendly():
    assert role_for_class(CLASS_BUMPER_BLUE, Alliance.BLUE) is ObjectRole.FRIENDLY


@pytest.mark.parametrize("alliance", list(Alliance))
def test_fuel_is_a_game_piece_regardless_of_alliance(alliance):
    assert role_for_class(CLASS_FUEL, alliance) is ObjectRole.GAME_PIECE


@pytest.mark.parametrize("class_name", [CLASS_BUMPER_RED, CLASS_BUMPER_BLUE])
def test_unknown_alliance_leaves_bumpers_unclassified(class_name):
    """We must not guess. An unclassified robot is handled conservatively
    downstream, which is not the same thing as calling it an opponent."""
    assert role_for_class(class_name, Alliance.UNKNOWN) is ObjectRole.UNKNOWN


def test_an_unexpected_class_name_is_unknown_not_an_exception():
    assert role_for_class("traffic_cone", Alliance.RED) is ObjectRole.UNKNOWN


def test_routing_is_by_name_so_the_two_bumper_classes_never_collide():
    """Both bumper classes must resolve to opposite roles under one
    alliance -- a routing bug that mapped them to the same role would
    make this fail."""
    for alliance in (Alliance.RED, Alliance.BLUE):
        red = role_for_class(CLASS_BUMPER_RED, alliance)
        blue = role_for_class(CLASS_BUMPER_BLUE, alliance)
        assert {red, blue} == {ObjectRole.OPPONENT, ObjectRole.FRIENDLY}


# ---------------------------------------------------------------------------
# Obstacle policy
# ---------------------------------------------------------------------------


def test_opponents_are_obstacles():
    assert is_obstacle(ObjectRole.OPPONENT)


def test_friendlies_are_never_obstacles_under_any_policy():
    assert not is_obstacle(ObjectRole.FRIENDLY)
    assert not is_obstacle(ObjectRole.FRIENDLY, treat_unknown_as_obstacle=True)


def test_unknown_robots_are_obstacles_by_default():
    """Conservative default: refusing to drive through a robot we cannot
    identify is always safe; the opposite is not."""
    assert is_obstacle(ObjectRole.UNKNOWN)


def test_unknown_robots_can_be_ignored_only_by_explicit_configuration():
    assert not is_obstacle(ObjectRole.UNKNOWN, treat_unknown_as_obstacle=False)


def test_game_pieces_are_not_obstacles():
    assert not is_obstacle(ObjectRole.GAME_PIECE)


# ---------------------------------------------------------------------------
# End-to-end: the rule stated in plain language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "is_red,opponent_class,friendly_class",
    [
        (True, CLASS_BUMPER_BLUE, CLASS_BUMPER_RED),
        (False, CLASS_BUMPER_RED, CLASS_BUMPER_BLUE),
    ],
)
def test_full_chain_from_the_fms_boolean(is_red, opponent_class, friendly_class):
    alliance = alliance_from_is_red(is_red)
    assert is_obstacle(role_for_class(opponent_class, alliance))
    assert not is_obstacle(role_for_class(friendly_class, alliance))
