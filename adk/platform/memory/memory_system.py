"""
AitherOS Unified Memory System

Three-tier memory architecture:
1. WORLD MEMORY - Shared facts about the world/setting (persistent)
2. SYSTEM MEMORY - How the system works, configurations (persistent)
3. PERSONA MEMORY - Character-specific memories, relationships, events (per-persona)

All memories integrate to create progressively extensive expanded context.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque


# Paths - use writable data directory (avoids read-only filesystem in Docker)
try:
    from aither_adk.paths import get_saga_subdir, get_saga_data_dir
    NARRATIVE_AGENT_DIR = get_saga_data_dir()
    MEMORY_DIR = get_saga_subdir("memory", create=True)
except ImportError:
    AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    NARRATIVE_AGENT_DIR = os.path.join(AGENT_DIR, "Saga")
    MEMORY_DIR = os.path.join(NARRATIVE_AGENT_DIR, "memory")
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
    except OSError:
        pass


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    category: str = "general"  # fact, event, relationship, preference, etc.
    importance: float = 0.5  # 0-1, higher = more important
    timestamp: str = ""
    source: str = ""  # who/what created this memory
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class WorldMemory:
    """
    Shared facts about the world/setting.
    
    Examples:
    - "The year is 2045"
    - "AitherOS runs on a high-tech server in Alex's home"
    - "Aither was created by Alex"
    - "The office has neon cyan and magenta lighting"
    """
    
    def __init__(self):
        self.path = os.path.join(MEMORY_DIR, "world_memory.json")
        self.facts: List[MemoryEntry] = []
        self.load()
    
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                self.facts = [MemoryEntry(**entry) for entry in data.get("facts", [])]
            except Exception as e:
                print(f"[WorldMemory] Load error: {e}")
    
    def save(self):
        try:
            data = {"facts": [asdict(f) for f in self.facts]}
            with open(self.path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WorldMemory] Save error: {e}")
    
    def add_fact(self, content: str, category: str = "fact", importance: float = 0.5, tags: List[str] = None):
        """Add a world fact."""
        entry = MemoryEntry(
            content=content,
            category=category,
            importance=importance,
            source="world",
            tags=tags or []
        )
        self.facts.append(entry)
        self.save()
    
    def get_context(self, max_entries: int = 10, categories: List[str] = None) -> str:
        """Get world context for prompt injection."""
        filtered = self.facts
        if categories:
            filtered = [f for f in filtered if f.category in categories]
        
        # Sort by importance
        sorted_facts = sorted(filtered, key=lambda x: x.importance, reverse=True)[:max_entries]
        
        if not sorted_facts:
            return ""
        
        lines = ["[WORLD CONTEXT]"]
        for fact in sorted_facts:
            lines.append(f"- {fact.content}")
        
        return "\n".join(lines)
    
    def search(self, query: str) -> List[MemoryEntry]:
        """Search facts by keyword."""
        query_lower = query.lower()
        return [f for f in self.facts if query_lower in f.content.lower()]


class SystemMemory:
    """
    System configuration and capabilities memory.
    
    Examples:
    - "ComfyUI is available for local image generation"
    - "Ollama runs mistral-nemo for uncensored chat"
    - "Safety level is set to UNSAFE (no content filtering)"
    - "The user's name is Alex"
    """
    
    def __init__(self):
        self.path = os.path.join(MEMORY_DIR, "system_memory.json")
        self.entries: List[MemoryEntry] = []
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                self.entries = [MemoryEntry(**entry) for entry in data.get("entries", [])]
                self.config = data.get("config", {})
            except Exception as e:
                print(f"[SystemMemory] Load error: {e}")
    
    def save(self):
        try:
            data = {
                "entries": [asdict(e) for e in self.entries],
                "config": self.config
            }
            with open(self.path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SystemMemory] Save error: {e}")
    
    def add_entry(self, content: str, category: str = "capability", importance: float = 0.5):
        """Add a system memory entry."""
        entry = MemoryEntry(
            content=content,
            category=category,
            importance=importance,
            source="system"
        )
        self.entries.append(entry)
        self.save()
    
    def set_config(self, key: str, value: Any):
        """Set a system configuration value."""
        self.config[key] = value
        self.save()
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a system configuration value."""
        return self.config.get(key, default)
    
    def get_context(self, max_entries: int = 10) -> str:
        """Get system context for prompt injection."""
        if not self.entries:
            return ""
        
        sorted_entries = sorted(self.entries, key=lambda x: x.importance, reverse=True)[:max_entries]
        
        lines = ["[SYSTEM CONTEXT]"]
        for entry in sorted_entries:
            lines.append(f"- {entry.content}")
        
        # Add key configs
        if self.config:
            lines.append("\n[CONFIGURATION]")
            for key, value in list(self.config.items())[:5]:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)


class PersonaMemory:
    """
    Character-specific memories.
    
    Each persona has:
    - Core traits (personality, speech patterns)
    - Relationships (with user, other personas)
    - Events (significant interactions)
    - Preferences (likes, dislikes)
    - Conversation history (recent exchanges)
    """
    
    def __init__(self, persona_name: str):
        self.persona_name = persona_name.lower()
        self.path = os.path.join(MEMORY_DIR, f"persona_{self.persona_name}.json")
        
        self.traits: List[MemoryEntry] = []
        self.relationships: Dict[str, str] = {}  # name -> relationship description
        self.events: List[MemoryEntry] = []
        self.preferences: Dict[str, List[str]] = {"likes": [], "dislikes": []}
        self.conversation_history: deque = deque(maxlen=20)  # Last 20 exchanges
        self.visual_description: str = ""
        
        self.load()
    
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                
                self.traits = [MemoryEntry(**e) for e in data.get("traits", [])]
                self.relationships = data.get("relationships", {})
                self.events = [MemoryEntry(**e) for e in data.get("events", [])]
                self.preferences = data.get("preferences", {"likes": [], "dislikes": []})
                self.conversation_history = deque(data.get("conversation_history", []), maxlen=20)
                self.visual_description = data.get("visual_description", "")
            except Exception as e:
                print(f"[PersonaMemory:{self.persona_name}] Load error: {e}")
    
    def save(self):
        try:
            data = {
                "traits": [asdict(t) for t in self.traits],
                "relationships": self.relationships,
                "events": [asdict(e) for e in self.events],
                "preferences": self.preferences,
                "conversation_history": list(self.conversation_history),
                "visual_description": self.visual_description
            }
            with open(self.path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[PersonaMemory:{self.persona_name}] Save error: {e}")
    
    def add_trait(self, content: str, importance: float = 0.7):
        """Add a personality trait."""
        entry = MemoryEntry(
            content=content,
            category="trait",
            importance=importance,
            source=self.persona_name
        )
        self.traits.append(entry)
        self.save()
    
    def set_relationship(self, name: str, description: str):
        """Set relationship with another entity."""
        self.relationships[name.lower()] = description
        self.save()
    
    def add_event(self, content: str, importance: float = 0.5):
        """Add a significant event."""
        entry = MemoryEntry(
            content=content,
            category="event",
            importance=importance,
            source=self.persona_name
        )
        self.events.append(entry)
        # Keep only last 50 events
        if len(self.events) > 50:
            self.events = self.events[-50:]
        self.save()
    
    def add_like(self, item: str):
        """Add something the persona likes."""
        if item not in self.preferences["likes"]:
            self.preferences["likes"].append(item)
            self.save()
    
    def add_dislike(self, item: str):
        """Add something the persona dislikes."""
        if item not in self.preferences["dislikes"]:
            self.preferences["dislikes"].append(item)
            self.save()
    
    def add_conversation_turn(self, role: str, content: str):
        """Add a conversation turn to history."""
        self.conversation_history.append({
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def get_conversation_context(self, max_turns: int = 10) -> str:
        """Get recent conversation history."""
        if not self.conversation_history:
            return ""
        
        recent = list(self.conversation_history)[-max_turns:]
        
        lines = ["[RECENT CONVERSATION]"]
        for turn in recent:
            role = "Admin" if turn["role"] == "user" else self.persona_name.title()
            content = turn["content"][:200]
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def get_context(self, include_conversation: bool = True) -> str:
        """Get full persona context for prompt injection."""
        lines = [f"[PERSONA MEMORY: {self.persona_name.upper()}]"]
        
        # Traits
        if self.traits:
            lines.append("\n[PERSONALITY]")
            for trait in sorted(self.traits, key=lambda x: x.importance, reverse=True)[:5]:
                lines.append(f"- {trait.content}")
        
        # Relationships
        if self.relationships:
            lines.append("\n[RELATIONSHIPS]")
            for name, rel in self.relationships.items():
                lines.append(f"- {name.title()}: {rel}")
        
        # Preferences
        if self.preferences["likes"]:
            lines.append(f"\n[LIKES] {', '.join(self.preferences['likes'][:5])}")
        if self.preferences["dislikes"]:
            lines.append(f"[DISLIKES] {', '.join(self.preferences['dislikes'][:5])}")
        
        # Recent events
        if self.events:
            lines.append("\n[RECENT EVENTS]")
            for event in self.events[-3:]:
                lines.append(f"- {event.content}")
        
        # Conversation history
        if include_conversation and self.conversation_history:
            lines.append("\n" + self.get_conversation_context())
        
        return "\n".join(lines)


class UnifiedMemorySystem:
    """
    Main memory manager that integrates all three memory types.
    """
    
    def __init__(self):
        self.world = WorldMemory()
        self.system = SystemMemory()
        self._persona_cache: Dict[str, PersonaMemory] = {}
        
        # Initialize with defaults if empty
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Set up default memories if none exist."""
        if not self.world.facts:
            self.world.add_fact("AitherOS is an advanced AI assistant platform", "fact", 0.9)
            self.world.add_fact("The setting is a high-tech futuristic office with neon lighting", "setting", 0.8)
            self.world.add_fact("The user's handle is 'Admin'", "fact", 0.9)
        
        if not self.system.entries:
            self.system.add_entry("ComfyUI is available for local image generation", "capability", 0.8)
            self.system.add_entry("Ollama provides local LLM inference", "capability", 0.8)
            self.system.add_entry("Safety level is HIGH - strict content filtering", "config", 0.9)
    
    def get_persona(self, name: str) -> PersonaMemory:
        """Get or create persona memory."""
        name_lower = name.lower()
        if name_lower not in self._persona_cache:
            self._persona_cache[name_lower] = PersonaMemory(name_lower)
        return self._persona_cache[name_lower]
    
    def get_full_context(
        self,
        persona_name: str = None,
        include_world: bool = True,
        include_system: bool = True,
        include_conversation: bool = True,
        max_tokens: int = 2000
    ) -> str:
        """
        Build complete context from all memory sources.
        
        Returns a formatted string suitable for prompt injection.
        """
        parts = []
        
        # World context
        if include_world:
            world_ctx = self.world.get_context(max_entries=5)
            if world_ctx:
                parts.append(world_ctx)
        
        # System context
        if include_system:
            system_ctx = self.system.get_context(max_entries=5)
            if system_ctx:
                parts.append(system_ctx)
        
        # Persona context
        if persona_name:
            persona = self.get_persona(persona_name)
            persona_ctx = persona.get_context(include_conversation=include_conversation)
            if persona_ctx:
                parts.append(persona_ctx)
        
        full_context = "\n\n".join(parts)
        
        # Truncate if too long (rough estimate)
        if len(full_context) > max_tokens * 4:  # ~4 chars per token
            full_context = full_context[:max_tokens * 4] + "\n[...context truncated...]"
        
        return full_context
    
    def record_conversation(self, persona_name: str, user_message: str, assistant_response: str):
        """Record a conversation exchange for a persona."""
        persona = self.get_persona(persona_name)
        persona.add_conversation_turn("user", user_message)
        persona.add_conversation_turn("assistant", assistant_response)
    
    def learn_from_conversation(self, persona_name: str, user_message: str, assistant_response: str):
        """
        Analyze conversation and extract learnable facts/events.
        This could be enhanced with LLM analysis later.
        """
        # Simple keyword-based learning
        msg_lower = user_message.lower()
        resp_lower = assistant_response.lower()
        
        # Learn preferences
        persona = self.get_persona(persona_name)
        
        if "i like" in msg_lower or "i love" in msg_lower:
            # Extract what user likes
            for phrase in ["i like", "i love"]:
                if phrase in msg_lower:
                    parts = msg_lower.split(phrase)
                    if len(parts) > 1:
                        item = parts[1].split(".")[0].split(",")[0].strip()[:50]
                        if item:
                            persona.add_like(item)
        
        if "i hate" in msg_lower or "i don't like" in msg_lower:
            for phrase in ["i hate", "i don't like"]:
                if phrase in msg_lower:
                    parts = msg_lower.split(phrase)
                    if len(parts) > 1:
                        item = parts[1].split(".")[0].split(",")[0].strip()[:50]
                        if item:
                            persona.add_dislike(item)
        
        # Record conversation
        self.record_conversation(persona_name, user_message, assistant_response)


# Singleton
_memory_system: Optional[UnifiedMemorySystem] = None


def get_memory_system() -> UnifiedMemorySystem:
    """Get the singleton memory system."""
    global _memory_system
    if _memory_system is None:
        _memory_system = UnifiedMemorySystem()
    return _memory_system


# Convenience functions

def get_context_for_persona(persona_name: str, include_conversation: bool = True) -> str:
    """Quick function to get full context for a persona."""
    return get_memory_system().get_full_context(
        persona_name=persona_name,
        include_conversation=include_conversation
    )


def record_chat(persona_name: str, user_msg: str, assistant_msg: str):
    """Quick function to record a conversation."""
    system = get_memory_system()
    system.learn_from_conversation(persona_name, user_msg, assistant_msg)


def add_world_fact(content: str, importance: float = 0.5):
    """Quick function to add a world fact."""
    get_memory_system().world.add_fact(content, importance=importance)


def add_persona_trait(persona_name: str, trait: str):
    """Quick function to add a persona trait."""
    get_memory_system().get_persona(persona_name).add_trait(trait)


def add_persona_event(persona_name: str, event: str):
    """Quick function to add a persona event."""
    get_memory_system().get_persona(persona_name).add_event(event)


def set_relationship(persona_name: str, other_name: str, description: str):
    """Quick function to set a relationship."""
    get_memory_system().get_persona(persona_name).set_relationship(other_name, description)

