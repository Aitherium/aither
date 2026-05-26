import random
from aither_adk.ui.console import safe_print

class GroupChatManager:
    def __init__(self, groups):
        self.state = {
            "active": False,
            "active_group_name": None,
            "groups": groups, # {name: [members]}
            "members": [], # Current active members
            "moderator": None,
            "free_chat": False, # If true, agents respond without @mentions
            "agent_states": {}, # {name: {"sleeping": 0, "exited": False, "attention": 0}}
            "scene_active": False, # True when in a specific scene (spicy, combat, story beat)
            "scene_turn_count": 0, # Current turn in the scene
            "scene_max_turns": 0, # Random max turns for this scene (2-10)
            "scene_type": None, # "spicy", "combat", "story_beat", etc.
            "manual_turn_limit": False
        }

    def start_group(self, group_name):
        if group_name not in self.state["groups"]:
            return False, f"Group '{group_name}' not found."

        self.state["active"] = True
        self.state["active_group_name"] = group_name
        self.state["members"] = self.state["groups"][group_name]
        self.state["agent_states"] = {}
        # Reset scene state
        self.state["scene_active"] = False
        self.state["scene_turn_count"] = 0

        return True, f"Group '{group_name}' started with members: {', '.join(self.state['members'])}"

    def stop_group(self):
        self.state["active"] = False
        self.state["active_group_name"] = None
        self.state["members"] = []
        self.state["agent_states"] = {}
        self.state["scene_active"] = False
        return True, "Group chat stopped."

    def get_active_members(self):
        return self.state["members"] if self.state["active"] else []

    def is_active(self):
        return self.state["active"]

    def set_free_chat(self, enabled: bool):
        self.state["free_chat"] = enabled

    def update_scene_turn(self):
        if self.state["scene_active"]:
            self.state["scene_turn_count"] += 1
            return self.state["scene_turn_count"], self.state["scene_max_turns"]
        return 0, 0

    def start_scene(self, scene_type, max_turns=None):
        self.state["scene_active"] = True
        self.state["scene_type"] = scene_type
        self.state["scene_turn_count"] = 0
        if max_turns:
            self.state["scene_max_turns"] = max_turns
            self.state["manual_turn_limit"] = True
        elif not self.state["manual_turn_limit"]:
             self.state["scene_max_turns"] = random.randint(3, 8)

        return self.state["scene_max_turns"]

    def end_scene(self):
        self.state["scene_active"] = False
        self.state["scene_turn_count"] = 0
        self.state["scene_max_turns"] = 0
        self.state["scene_type"] = None
        self.state["manual_turn_limit"] = False

    def get_member_state(self, member_name):
        if "agent_states" not in self.state:
            self.state["agent_states"] = {}
        if member_name not in self.state["agent_states"]:
            self.state["agent_states"][member_name] = {"sleeping": 0, "exited": False, "attention": 0}
        return self.state["agent_states"][member_name]

    def wake_member(self, member_name):
        state = self.get_member_state(member_name)
        state["exited"] = False
        state["sleeping"] = 0
        state["attention"] = 5 # Boost attention

async def decide_next_speaker(members, history_context=""):
    """Decides who should speak next in a group chat."""
    # Simplified approach:
    if members:
        return members[0]
    return "User"
