"""Dice Roll Tool — RPG dice mechanics and skill checks."""

from __future__ import annotations

import random

from adk.tools import tool


@tool(
    name="roll_dice",
    description="Roll RPG dice (d4, d6, d8, d10, d12, d20, d100) with optional modifier and skill check.",
)
def roll_dice(
    dice_type: str = "d20",
    modifier: int = 0,
    check_type: str = "",
    difficulty: int = 0,
) -> dict:
    """Roll dice with RPG mechanics.

    Args:
        dice_type: Dice type: 'd4', 'd6', 'd8', 'd10', 'd12', 'd20', 'd100'
        modifier: Numeric modifier to add to the roll
        check_type: Type of check (e.g. 'strength', 'perception', 'persuasion')
        difficulty: Difficulty class (DC) for the check — 0 means no check
    """
    valid_dice = {"d4": 4, "d6": 6, "d8": 8, "d10": 10, "d12": 12, "d20": 20, "d100": 100}
    sides = valid_dice.get(dice_type.lower(), 20)

    roll = random.randint(1, sides)
    total = roll + modifier

    result = {
        "dice": dice_type,
        "roll": roll,
        "modifier": modifier,
        "total": total,
        "critical_success": dice_type == "d20" and roll == 20,
        "critical_failure": dice_type == "d20" and roll == 1,
    }

    if check_type:
        result["check_type"] = check_type

    if difficulty > 0:
        result["difficulty"] = difficulty
        result["success"] = total >= difficulty
        result["margin"] = total - difficulty

    # Narrative flavor
    if result.get("critical_success"):
        result["flavor"] = "CRITICAL SUCCESS! The fates smile upon you!"
    elif result.get("critical_failure"):
        result["flavor"] = "CRITICAL FAILURE! Disaster strikes..."
    elif result.get("success"):
        result["flavor"] = f"Success! ({total} vs DC {difficulty})"
    elif difficulty > 0:
        result["flavor"] = f"Failed. ({total} vs DC {difficulty})"
    else:
        result["flavor"] = f"Rolled {total} on {dice_type}"

    return result


@tool(
    name="roll_multiple",
    description="Roll multiple dice at once (e.g., 3d6 for damage).",
)
def roll_multiple(
    count: int = 1,
    dice_type: str = "d6",
    modifier: int = 0,
) -> dict:
    """Roll multiple dice.

    Args:
        count: Number of dice to roll
        dice_type: Dice type
        modifier: Modifier added to the total
    """
    valid_dice = {"d4": 4, "d6": 6, "d8": 8, "d10": 10, "d12": 12, "d20": 20, "d100": 100}
    sides = valid_dice.get(dice_type.lower(), 6)

    count = max(1, min(count, 20))  # Clamp 1-20
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls) + modifier

    return {
        "dice": f"{count}{dice_type}",
        "rolls": rolls,
        "modifier": modifier,
        "total": total,
        "average": round(sum(rolls) / len(rolls), 1),
    }
