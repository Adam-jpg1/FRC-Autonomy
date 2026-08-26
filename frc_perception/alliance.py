"""Alliance resolution: which bumper color is an opponent.

Kept as a tiny pure module (no NetworkTables, no threading) because it is
a safety decision -- driving into a friendly robot because a boolean was
inverted is exactly the failure this file exists to prevent.

Rule (FRC): the FMS publishes ``FMSInfo/IsRedAlliance``.  If WE are red,
BLUE bumpers are opponents.  If WE are blue, RED bumpers are opponents.
If we do not know, we know nothing: every bumper resolves to UNKNOWN and
the caller must treat unknown robots conservatively.
"""

from __future__ import annotations

from typing import Optional

from .config import CLASS_BUMPER_BLUE, CLASS_BUMPER_RED, CLASS_FUEL
from .types import Alliance, ObjectRole


def alliance_from_is_red(is_red: Optional[bool]) -> Alliance:
    """Map the raw ``FMSInfo/IsRedAlliance`` value onto an :class:`Alliance`.

    ``None`` (key absent, never published, or stale) is UNKNOWN.  We do
    not default to an alliance: guessing wrong swaps friend and foe.
    """
    if is_red is None:
        return Alliance.UNKNOWN
    return Alliance.RED if bool(is_red) else Alliance.BLUE


def role_for_class(class_name: str, alliance: Alliance) -> ObjectRole:
    """Resolve a YOLO class name to its role given our alliance.

    Routing is by class NAME, never by the model's class index, so
    re-exporting the model with a different class order cannot silently
    turn opponents into game pieces.
    """
    if class_name == CLASS_FUEL:
        return ObjectRole.GAME_PIECE

    if class_name == CLASS_BUMPER_RED:
        if alliance is Alliance.RED:
            return ObjectRole.FRIENDLY
        if alliance is Alliance.BLUE:
            return ObjectRole.OPPONENT
        return ObjectRole.UNKNOWN

    if class_name == CLASS_BUMPER_BLUE:
        if alliance is Alliance.BLUE:
            return ObjectRole.FRIENDLY
        if alliance is Alliance.RED:
            return ObjectRole.OPPONENT
        return ObjectRole.UNKNOWN

    return ObjectRole.UNKNOWN


def is_obstacle(role: ObjectRole, treat_unknown_as_obstacle: bool = True) -> bool:
    """Should a robot with this role be avoided?

    Conservative default: an UNKNOWN robot (alliance data missing or
    stale) is avoided.  Refusing to drive through a robot we cannot
    identify is always safe; the opposite is not.

    A FRIENDLY robot is never added to the opponent list -- avoidance of
    teammates is the human driver's job and adding them here would make
    the opponent count wrong.
    """
    if role is ObjectRole.OPPONENT:
        return True
    if role is ObjectRole.UNKNOWN:
        return treat_unknown_as_obstacle
    return False


__all__ = ["alliance_from_is_red", "is_obstacle", "role_for_class"]
