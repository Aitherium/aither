import json
import logging
import os
import random
from typing import Any, Dict

logger = logging.getLogger(__name__)

class GameEngine:
    """
    Manages the state of the Narrative RPG.
    Handles Stats, Inventory, Quests, and Dice Rolls.
    """
    def __init__(self, state_file: str = "game_state.json"):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as exc:
                logger.debug(f"Game state load failed: {exc}")
        return self._initialize_new_state()

    def _initialize_new_state(self) -> Dict[str, Any]:
        return {
            "player": {
                "name": "Traveler",
                "level": 1,
                "xp": 0,
                "hp": 100,
                "max_hp": 100,
                "stats": {
                    "strength": 10,
                    "dexterity": 10,
                    "intelligence": 10,
                    "charisma": 10
                },
                "inventory": [],
                "gold": 0
            },
            "quests": [],
            "location": "Unknown",
            "history": []
        }

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_status(self) -> str:
        p = self.state["player"]
        stats = p["stats"]
        return (
            f" **{p['name']}** (Lvl {p['level']})\n"
            f" HP: {p['hp']}/{p['max_hp']} | * XP: {p['xp']}\n"
            f"[OK] STR: {stats['strength']} |  DEX: {stats['dexterity']} | [BRAIN] INT: {stats['intelligence']} |  CHA: {stats['charisma']}\n"
            f" Gold: {p['gold']} |  Items: {len(p['inventory'])}"
        )

    def update_stats(self, stat: str, value: int, relative: bool = True) -> str:
        """Updates a player stat."""
        p = self.state["player"]

        if stat.lower() in ["hp", "health"]:
            target = "hp"
        elif stat.lower() in ["xp", "experience"]:
            target = "xp"
        elif stat.lower() in ["gold", "money"]:
            target = "gold"
        elif stat.lower() in p["stats"]:
            # Attribute update
            current = p["stats"][stat.lower()]
            new_val = (current + value) if relative else value
            p["stats"][stat.lower()] = new_val
            self.save_state()
            return f"Stat updated: {stat.title()} is now {new_val}."
        else:
            return f"Unknown stat: {stat}"

        # Direct property update
        current = p.get(target, 0)
        new_val = (current + value) if relative else value

        # Cap HP
        if target == "hp":
            new_val = min(new_val, p["max_hp"])
            new_val = max(0, new_val)

        p[target] = new_val

        # Level up check
        if target == "xp":
            # Simple level curve: Level * 1000
            req = p["level"] * 1000
            if p["xp"] >= req:
                p["level"] += 1
                p["xp"] -= req
                p["max_hp"] += 10
                p["hp"] = p["max_hp"]
                self.save_state()
                return f"[!] LEVEL UP! You are now Level {p['level']}! Max HP increased."

        self.save_state()
        return f"{target.upper()} updated. Now: {new_val}"

    def manage_inventory(self, action: str, item: str, quantity: int = 1) -> str:
        p = self.state["player"]
        inv = p["inventory"]

        if action == "add":
            # Check if exists
            found = False
            for i in inv:
                if i["name"].lower() == item.lower():
                    i["quantity"] += quantity
                    found = True
                    break
            if not found:
                inv.append({"name": item, "quantity": quantity})
            self.save_state()
            return f"Added {quantity}x {item} to inventory."

        elif action == "remove":
            for i in inv:
                if i["name"].lower() == item.lower():
                    i["quantity"] -= quantity
                    if i["quantity"] <= 0:
                        inv.remove(i)
                    self.save_state()
                    return f"Removed {quantity}x {item} from inventory."
            return f"Item not found: {item}"

        elif action == "list":
            if not inv:
                return "Inventory is empty."
            return " **Inventory:**\n" + "\n".join([f"- {i['name']} ({i['quantity']})" for i in inv])

        return "Invalid inventory action."

    def manage_quest(self, action: str, title: str, description: str = "") -> str:
        quests = self.state["quests"]

        if action == "add":
            quests.append({
                "title": title,
                "description": description,
                "status": "active"
            })
            self.save_state()
            return f"[DOC] New Quest: **{title}**"

        elif action == "complete":
            for q in quests:
                if q["title"].lower() == title.lower():
                    q["status"] = "completed"
                    self.save_state()
                    return f"[DONE] Quest Completed: **{title}**"
            return f"Quest not found: {title}"

        elif action == "list":
            active = [q for q in quests if q["status"] == "active"]
            if not active:
                return "No active quests."
            return "[DOC] **Active Quests:**\n" + "\n".join([f"- {q['title']}: {q['description']}" for q in active])

        return "Invalid quest action."

    def roll_dice(self, expression: str = "1d20") -> str:
        """Parses 1d20+5, 2d6, etc."""
        try:
            parts = expression.lower().split('+')
            modifier = int(parts[1]) if len(parts) > 1 else 0
            dice_part = parts[0]

            if 'd' not in dice_part:
                return "Invalid dice format. Use XdY (e.g. 1d20)"

            num, sides = map(int, dice_part.split('d'))

            rolls = [random.randint(1, sides) for _ in range(num)]
            total = sum(rolls) + modifier

            roll_str = f"[{', '.join(map(str, rolls))}]"
            if modifier:
                return f" Rolled {expression}: {roll_str} + {modifier} = **{total}**"
            return f" Rolled {expression}: {roll_str} = **{total}**"
        except Exception as e:
            return f"Dice error: {e}"
