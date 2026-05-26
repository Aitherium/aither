"""Simulation Client — Dual-backend bridge to Prometheus or local fallback.

Saga tools call SimulationClient, which auto-detects whether Prometheus
is running. If yes, routes to PrometheusBackend (httpx). If no, falls
back to LocalBackend (deterministic rules + Markov weather).

No AitherOS imports — standalone-safe.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger("saga.simulation")


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SimulationEvent:
    event_type: str
    description: str
    severity: str = "normal"  # trivial, normal, notable, important, critical
    actors: List[str] = field(default_factory=list)
    location: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationState:
    world_id: str = ""
    game_time: str = ""
    weather: str = "clear"
    season: str = "spring"
    events: List[SimulationEvent] = field(default_factory=list)
    npc_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    faction_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    economy: Dict[str, Any] = field(default_factory=dict)
    awi_metrics: Dict[str, float] = field(default_factory=dict)
    connected: bool = False


@dataclass
class TickResult:
    events: List[SimulationEvent] = field(default_factory=list)
    game_time: str = ""
    weather: str = "clear"
    npc_movements: List[Dict[str, str]] = field(default_factory=list)
    faction_actions: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# BACKEND PROTOCOL
# ============================================================================

class SimulationBackend(Protocol):
    async def start(self, project_data: dict) -> SimulationState: ...
    async def tick(self, world_id: str, minutes: int) -> TickResult: ...
    async def get_state(self, world_id: str) -> SimulationState: ...
    async def get_recent_events(self, world_id: str, count: int) -> List[SimulationEvent]: ...
    async def get_npc_decision(self, world_id: str, npc_id: str, situation: str, player_action: str) -> dict: ...
    async def get_dialogue_context(self, world_id: str, npc_id: str) -> dict: ...
    async def get_narrative_context(self, world_id: str, player_location: str) -> dict: ...
    async def mcts_search(self, world_id: str, faction_id: str) -> dict: ...
    async def register_system(self, world_id: str, system_type: str, config: dict) -> bool: ...
    async def get_awi_metrics(self, world_id: str) -> dict: ...
    async def stop(self, world_id: str) -> bool: ...
    async def health_check(self) -> bool: ...


# ============================================================================
# PROMETHEUS BACKEND — HTTP calls to AitherPrometheus:8178
# ============================================================================

class PrometheusBackend:
    """Routes all simulation calls to Prometheus via httpx."""

    def __init__(self, url: str | None = None, timeout: float = 30.0):
        self.url = url or os.environ.get("SAGA_PROMETHEUS_URL", "http://localhost:8178")
        self.timeout = timeout

    async def _client(self):
        import httpx
        return httpx.AsyncClient(timeout=self.timeout)

    async def start(self, project_data: dict) -> SimulationState:
        import httpx
        async with await self._client() as client:
            resp = await client.post(f"{self.url}/simulate", json=project_data)
            resp.raise_for_status()
            d = resp.json()
            return SimulationState(
                world_id=d.get("world_id", ""),
                game_time=d.get("game_time", ""),
                weather=d.get("weather", "clear"),
                season=d.get("season", "spring"),
                connected=True,
            )

    async def tick(self, world_id: str, minutes: int) -> TickResult:
        import httpx
        async with await self._client() as client:
            resp = await client.post(
                f"{self.url}/tick",
                json={"world_id": world_id, "minutes": minutes},
            )
            resp.raise_for_status()
            d = resp.json()
            events = [
                SimulationEvent(
                    event_type=e.get("type", e.get("event_type", "world_event")),
                    description=e.get("description", ""),
                    severity=e.get("severity", "normal"),
                    actors=e.get("actors", []),
                    location=e.get("location", ""),
                    data=e.get("data", {}),
                )
                for e in d.get("events", [])
            ]
            return TickResult(
                events=events,
                game_time=d.get("game_time", ""),
                weather=d.get("weather", "clear"),
                npc_movements=d.get("npc_movements", []),
                faction_actions=d.get("faction_actions", []),
            )

    async def get_state(self, world_id: str) -> SimulationState:
        import httpx
        async with await self._client() as client:
            resp = await client.get(f"{self.url}/state/{world_id}")
            resp.raise_for_status()
            d = resp.json()
            return SimulationState(
                world_id=world_id,
                game_time=d.get("game_time", ""),
                weather=d.get("weather", "clear"),
                season=d.get("season", "spring"),
                npc_states=d.get("npc_states", {}),
                faction_states=d.get("faction_states", {}),
                economy=d.get("economy", {}),
                awi_metrics=d.get("awi_metrics", {}),
                connected=True,
            )

    async def get_recent_events(self, world_id: str, count: int) -> List[SimulationEvent]:
        import httpx
        async with await self._client() as client:
            resp = await client.get(
                f"{self.url}/world/{world_id}/story/recent",
                params={"count": count},
            )
            resp.raise_for_status()
            return [
                SimulationEvent(
                    event_type=e.get("type", e.get("event_type", "world_event")),
                    description=e.get("description", ""),
                    severity=e.get("severity", "normal"),
                    actors=e.get("actors", []),
                    location=e.get("location", ""),
                    data=e.get("data", {}),
                )
                for e in resp.json()
            ]

    async def get_npc_decision(self, world_id: str, npc_id: str,
                               situation: str, player_action: str) -> dict:
        import httpx
        async with await self._client() as client:
            resp = await client.post(
                f"{self.url}/oracle/npc-decision/{world_id}/{npc_id}",
                json={"situation": situation, "player_action": player_action},
            )
            resp.raise_for_status()
            return resp.json()

    async def get_dialogue_context(self, world_id: str, npc_id: str) -> dict:
        import httpx
        async with await self._client() as client:
            resp = await client.get(
                f"{self.url}/oracle/dialogue-context/{world_id}/{npc_id}"
            )
            resp.raise_for_status()
            return resp.json()

    async def get_narrative_context(self, world_id: str, player_location: str) -> dict:
        import httpx
        params = {"player_location": player_location} if player_location else {}
        async with await self._client() as client:
            resp = await client.get(
                f"{self.url}/oracle/narrative-context/{world_id}",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    async def mcts_search(self, world_id: str, faction_id: str) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.url}/mcts/search/{world_id}/{faction_id}"
            )
            resp.raise_for_status()
            return resp.json()

    async def register_system(self, world_id: str, system_type: str, config: dict) -> bool:
        import httpx
        async with await self._client() as client:
            resp = await client.post(
                f"{self.url}/world/{world_id}/config/system",
                json={"system_type": system_type, **config},
            )
            return resp.status_code == 200

    async def get_awi_metrics(self, world_id: str) -> dict:
        import httpx
        async with await self._client() as client:
            resp = await client.get(f"{self.url}/world/{world_id}/awi")
            resp.raise_for_status()
            return resp.json()

    async def stop(self, world_id: str) -> bool:
        import httpx
        async with await self._client() as client:
            resp = await client.delete(f"{self.url}/world/{world_id}")
            return resp.status_code == 200

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.url}/health")
                return resp.status_code == 200
        except Exception:
            return False


# ============================================================================
# LOCAL BACKEND — Deterministic fallback for standalone use
# ============================================================================

# Markov weather transitions ported from AitherOS weather_manager.py
WEATHER_TRANSITIONS: Dict[str, Dict[str, float]] = {
    "clear":        {"clear": 0.6, "cloudy": 0.3, "windy": 0.1},
    "cloudy":       {"cloudy": 0.4, "clear": 0.2, "overcast": 0.2, "light_rain": 0.15, "fog": 0.05},
    "overcast":     {"overcast": 0.3, "cloudy": 0.2, "light_rain": 0.3, "rain": 0.15, "fog": 0.05},
    "light_rain":   {"light_rain": 0.3, "rain": 0.3, "cloudy": 0.3, "clear": 0.1},
    "rain":         {"rain": 0.4, "light_rain": 0.2, "heavy_rain": 0.2, "thunderstorm": 0.1, "cloudy": 0.1},
    "heavy_rain":   {"heavy_rain": 0.3, "rain": 0.3, "thunderstorm": 0.2, "light_rain": 0.2},
    "thunderstorm": {"thunderstorm": 0.2, "heavy_rain": 0.3, "rain": 0.3, "cloudy": 0.2},
    "snow":         {"snow": 0.5, "blizzard": 0.1, "cloudy": 0.3, "clear": 0.1},
    "blizzard":     {"blizzard": 0.3, "snow": 0.5, "cloudy": 0.2},
    "fog":          {"fog": 0.3, "cloudy": 0.4, "clear": 0.3},
    "windy":        {"windy": 0.3, "clear": 0.4, "cloudy": 0.3},
}

FACTION_ACTIONS = ["expand", "recruit", "trade", "spy", "fortify", "diplomacy", "patrol"]

TIME_PROGRESSION = ["dawn", "day", "dusk", "night"]


class LocalBackend:
    """Self-contained lightweight simulation for standalone use."""

    def __init__(self):
        self._worlds: Dict[str, Dict[str, Any]] = {}

    def _world(self, world_id: str) -> Dict[str, Any]:
        if world_id not in self._worlds:
            self._worlds[world_id] = {
                "game_time": "Year 1, Month 1, Day 1",
                "weather": "clear",
                "season": "spring",
                "hour": 8,
                "day": 1,
                "month": 1,
                "year": 1,
                "events": [],
                "npc_states": {},
                "faction_states": {},
                "systems": {},
            }
        return self._worlds[world_id]

    def _advance_weather(self, current: str) -> str:
        transitions = WEATHER_TRANSITIONS.get(current, {"clear": 1.0})
        roll = random.random()
        cumulative = 0.0
        for weather, prob in transitions.items():
            cumulative += prob
            if roll <= cumulative:
                return weather
        return current

    def _advance_time(self, w: dict, minutes: int):
        w["hour"] = (w["hour"] + minutes // 60) % 24
        days_passed = minutes // (24 * 60)
        if days_passed > 0 or (w["hour"] + minutes // 60) >= 24:
            days_passed = max(1, days_passed)
        w["day"] += days_passed
        while w["day"] > 30:
            w["day"] -= 30
            w["month"] += 1
        while w["month"] > 12:
            w["month"] -= 12
            w["year"] += 1
        seasons = ["winter", "winter", "spring", "spring", "spring",
                    "summer", "summer", "summer", "fall", "fall", "fall", "winter"]
        w["season"] = seasons[w["month"] - 1]
        w["game_time"] = f"Year {w['year']}, Month {w['month']}, Day {w['day']}"

    async def start(self, project_data: dict) -> SimulationState:
        world_id = hashlib.sha256(
            f"{time.time()}-{id(project_data)}".encode()
        ).hexdigest()[:12]
        w = self._world(world_id)
        return SimulationState(
            world_id=world_id,
            game_time=w["game_time"],
            weather=w["weather"],
            season=w["season"],
            connected=True,
        )

    async def tick(self, world_id: str, minutes: int) -> TickResult:
        w = self._world(world_id)
        old_weather = w["weather"]
        w["weather"] = self._advance_weather(old_weather)
        self._advance_time(w, minutes)

        events: List[SimulationEvent] = []
        npc_movements: List[Dict[str, str]] = []
        faction_actions: List[Dict[str, Any]] = []

        # Weather change event
        if w["weather"] != old_weather:
            events.append(SimulationEvent(
                event_type="weather_changed",
                description=f"Weather shifted from {old_weather} to {w['weather']}",
                severity="normal",
            ))

        # NPC movements — probabilistic for each tracked NPC
        for npc_id, npc in w.get("npc_states", {}).items():
            if random.random() < 0.3:
                locs = npc.get("known_locations", ["town square", "market", "tavern", "gate"])
                old_loc = npc.get("location", locs[0] if locs else "unknown")
                new_loc = random.choice([l for l in locs if l != old_loc] or locs)
                npc["location"] = new_loc
                npc_movements.append({
                    "npc_id": npc_id,
                    "name": npc.get("name", npc_id),
                    "from": old_loc,
                    "to": new_loc,
                })

        # Faction round-robin actions
        for faction_id, faction in w.get("faction_states", {}).items():
            action = random.choice(FACTION_ACTIONS)
            props = faction.get("properties", {})
            # Weight by faction goals
            if "military" in props.get("goals", "").lower():
                action = random.choice(["expand", "fortify", "patrol", "recruit"])
            elif "trade" in props.get("goals", "").lower():
                action = random.choice(["trade", "diplomacy", "recruit", "expand"])
            faction_actions.append({
                "faction_id": faction_id,
                "name": faction.get("name", faction_id),
                "action": action,
                "description": f"{faction.get('name', faction_id)} chose to {action}",
            })

        # Random notable event (~15% per day equivalent)
        if random.random() < 0.15 * max(1, minutes / (24 * 60)):
            notable_events = [
                "A traveling merchant arrives with exotic goods",
                "Strange lights seen in the sky at dusk",
                "A messenger arrives bearing urgent news",
                "An old ruin is discovered nearby",
                "Bandits spotted on the trade roads",
                "A local festival draws crowds",
                "Unusual animal behavior reported",
                "A mysterious stranger appears at the gate",
            ]
            events.append(SimulationEvent(
                event_type="world_event",
                description=random.choice(notable_events),
                severity="notable",
            ))

        w["events"].extend(events)
        # Keep last 100 events
        w["events"] = w["events"][-100:]

        return TickResult(
            events=events,
            game_time=w["game_time"],
            weather=w["weather"],
            npc_movements=npc_movements,
            faction_actions=faction_actions,
        )

    async def get_state(self, world_id: str) -> SimulationState:
        w = self._world(world_id)
        return SimulationState(
            world_id=world_id,
            game_time=w["game_time"],
            weather=w["weather"],
            season=w["season"],
            npc_states=w.get("npc_states", {}),
            faction_states=w.get("faction_states", {}),
            economy=w.get("systems", {}).get("economy", {}),
            awi_metrics=self._compute_awi(w),
            connected=True,
        )

    def _compute_awi(self, w: dict) -> Dict[str, float]:
        npc_count = len(w.get("npc_states", {}))
        faction_count = len(w.get("faction_states", {}))
        event_count = len(w.get("events", []))
        return {
            "population_health": min(npc_count / 10.0, 1.0) * 100,
            "safety_order": 75.0,
            "governance_participation": min(faction_count / 5.0, 1.0) * 100,
            "economic_vitality": 50.0,
            "social_fabric": min(npc_count * faction_count / 25.0, 1.0) * 100,
            "cultural_output": min(event_count / 50.0, 1.0) * 100,
        }

    async def get_recent_events(self, world_id: str, count: int) -> List[SimulationEvent]:
        w = self._world(world_id)
        return w.get("events", [])[-count:]

    async def get_npc_decision(self, world_id: str, npc_id: str,
                               situation: str, player_action: str) -> dict:
        w = self._world(world_id)
        npc = w.get("npc_states", {}).get(npc_id, {})
        personality = npc.get("personality", "cautious")

        action_weights = {
            "observe": 0.3, "approach": 0.2, "retreat": 0.1,
            "speak": 0.2, "assist": 0.1, "ignore": 0.1,
        }
        if "aggressive" in personality.lower():
            action_weights["approach"] = 0.4
            action_weights["retreat"] = 0.05
        elif "friendly" in personality.lower():
            action_weights["speak"] = 0.4
            action_weights["assist"] = 0.3

        actions = list(action_weights.keys())
        weights = list(action_weights.values())
        chosen = random.choices(actions, weights=weights, k=1)[0]

        return {
            "npc_id": npc_id,
            "chosen_action": chosen,
            "reasoning": f"Based on {personality} personality",
            "emotional_state": npc.get("emotional_state", "neutral"),
        }

    async def get_dialogue_context(self, world_id: str, npc_id: str) -> dict:
        w = self._world(world_id)
        npc = w.get("npc_states", {}).get(npc_id, {})
        return {
            "npc_id": npc_id,
            "personality": npc.get("personality", ""),
            "emotional_state": npc.get("emotional_state", "neutral"),
            "location": npc.get("location", ""),
            "recent_events": [e.description for e in w.get("events", [])[-5:]
                              if isinstance(e, SimulationEvent)],
        }

    async def get_narrative_context(self, world_id: str, player_location: str) -> dict:
        w = self._world(world_id)
        return {
            "game_time": w["game_time"],
            "weather": w["weather"],
            "season": w["season"],
            "player_location": player_location,
            "nearby_npcs": [
                {"name": n.get("name", nid), "location": n.get("location", "")}
                for nid, n in w.get("npc_states", {}).items()
                if n.get("location", "") == player_location
            ],
            "recent_events": [
                e.description for e in w.get("events", [])[-5:]
                if isinstance(e, SimulationEvent)
            ],
        }

    async def mcts_search(self, world_id: str, faction_id: str) -> dict:
        w = self._world(world_id)
        faction = w.get("faction_states", {}).get(faction_id, {})
        actions = FACTION_ACTIONS
        # Simple weighted scoring instead of real MCTS
        scored = []
        for action in actions:
            score = random.uniform(0.3, 1.0)
            if action in (faction.get("preferred_actions", [])):
                score += 0.2
            scored.append({"action": action, "score": min(score, 1.0)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {
            "faction_id": faction_id,
            "best_action": scored[0]["action"],
            "best_score": scored[0]["score"],
            "evaluated_actions": scored[:5],
        }

    async def register_system(self, world_id: str, system_type: str, config: dict) -> bool:
        w = self._world(world_id)
        w["systems"][system_type] = config
        return True

    async def get_awi_metrics(self, world_id: str) -> dict:
        w = self._world(world_id)
        return self._compute_awi(w)

    async def stop(self, world_id: str) -> bool:
        self._worlds.pop(world_id, None)
        return True

    async def health_check(self) -> bool:
        return True


# ============================================================================
# SIMULATION CLIENT SINGLETON
# ============================================================================

_instance: Optional["SimulationClient"] = None


class SimulationClient:
    """Auto-detecting simulation client. Tries Prometheus, falls back to local."""

    def __init__(self):
        self._backend: Optional[SimulationBackend] = None
        self._backend_name: str = "none"
        self._last_state: Optional[SimulationState] = None
        self._detected: bool = False

    async def _detect_backend(self):
        if self._detected:
            return
        self._detected = True
        prometheus = PrometheusBackend()
        if await prometheus.health_check():
            self._backend = prometheus
            self._backend_name = "prometheus"
            logger.info("Simulation backend: Prometheus (%s)", prometheus.url)
        else:
            self._backend = LocalBackend()
            self._backend_name = "local"
            logger.info("Simulation backend: Local (standalone mode)")

    async def _get_backend(self) -> SimulationBackend:
        await self._detect_backend()
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def last_state(self) -> Optional[SimulationState]:
        return self._last_state

    async def start(self, project_data: dict) -> SimulationState:
        backend = await self._get_backend()
        state = await backend.start(project_data)
        self._last_state = state
        return state

    async def tick(self, world_id: str, minutes: int = 60) -> TickResult:
        backend = await self._get_backend()
        result = await backend.tick(world_id, minutes)
        # Update cached state
        if self._last_state and self._last_state.world_id == world_id:
            self._last_state.game_time = result.game_time
            self._last_state.weather = result.weather
            self._last_state.events = result.events
        return result

    async def get_state(self, world_id: str) -> SimulationState:
        backend = await self._get_backend()
        state = await backend.get_state(world_id)
        self._last_state = state
        return state

    async def get_recent_events(self, world_id: str, count: int = 20) -> List[SimulationEvent]:
        backend = await self._get_backend()
        return await backend.get_recent_events(world_id, count)

    async def get_npc_decision(self, world_id: str, npc_id: str,
                               situation: str = "", player_action: str = "") -> dict:
        backend = await self._get_backend()
        return await backend.get_npc_decision(world_id, npc_id, situation, player_action)

    async def get_dialogue_context(self, world_id: str, npc_id: str) -> dict:
        backend = await self._get_backend()
        return await backend.get_dialogue_context(world_id, npc_id)

    async def get_narrative_context(self, world_id: str, player_location: str = "") -> dict:
        backend = await self._get_backend()
        return await backend.get_narrative_context(world_id, player_location)

    async def mcts_search(self, world_id: str, faction_id: str) -> dict:
        backend = await self._get_backend()
        return await backend.mcts_search(world_id, faction_id)

    async def register_system(self, world_id: str, system_type: str, config: dict) -> bool:
        backend = await self._get_backend()
        return await backend.register_system(world_id, system_type, config)

    async def get_awi_metrics(self, world_id: str) -> dict:
        backend = await self._get_backend()
        return await backend.get_awi_metrics(world_id)

    async def stop(self, world_id: str) -> bool:
        backend = await self._get_backend()
        result = await backend.stop(world_id)
        if self._last_state and self._last_state.world_id == world_id:
            self._last_state = None
        return result

    def sync_to_world_state(self, world_state, graph=None):
        """Write cached simulation state into Saga WorldState fields."""
        if not self._last_state:
            return
        s = self._last_state
        world_state.prometheus_world_id = s.world_id
        world_state.prometheus_active = s.connected
        world_state.simulation_time = s.game_time
        world_state.simulation_weather = s.weather
        world_state.last_sim_events = [
            e.description for e in s.events[-5:]
        ]
        if graph:
            graph.save()


def get_simulation_client() -> SimulationClient:
    """Get or create the global SimulationClient singleton."""
    global _instance
    if _instance is None:
        _instance = SimulationClient()
    return _instance
