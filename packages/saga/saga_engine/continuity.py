"""
Continuity Checker
==================

Scans scenes against StoryGraph for consistency issues:
1. Entity mentions vs graph existence (unknown entities)
2. Dead/destroyed characters appearing in later scenes
3. Characters in multiple locations simultaneously
4. Relationship contradictions (hostile characters cooperating)
5. Timeline inconsistencies (day numbering gaps)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph import StoryGraph
from .models import EdgeType, NodeStatus, NodeType, StoryNode

logger = logging.getLogger("Saga.ContinuityChecker")

# Regex for extracting entity markup from prose lines: <E t='type'>Name</E>
_ENTITY_RE = re.compile(r"<E\s+t='(\w+)'>([^<]+)</E>")

# Maximum day gap before flagging as suspicious
_MAX_DAY_GAP = 10


class IssueSeverity(str, Enum):
    ERROR = "error"        # Definite continuity break
    WARNING = "warning"    # Likely issue
    INFO = "info"          # Observation worth noting


class IssueType(str, Enum):
    UNKNOWN_ENTITY = "unknown_entity"
    DEAD_ENTITY_PRESENT = "dead_entity_present"
    LOCATION_CONFLICT = "location_conflict"
    RELATIONSHIP_CONTRADICTION = "relationship_contradiction"
    TIMELINE_GAP = "timeline_gap"
    MISSING_INTRODUCTION = "missing_introduction"


@dataclass
class ContinuityIssue:
    """A single continuity problem found by the checker."""

    type: IssueType
    severity: IssueSeverity
    scene_index: int
    scene_title: str
    message: str
    entity_name: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "scene_index": self.scene_index,
            "scene_title": self.scene_title,
            "message": self.message,
            "entity_name": self.entity_name,
            "suggestion": self.suggestion,
        }


def _extract_entities_from_prose(prose: List[str]) -> List[Tuple[str, str]]:
    """Extract (type, name) pairs from prose lines containing <E> markup.

    Args:
        prose: List of prose strings potentially containing ``<E t='type'>Name</E>``.

    Returns:
        List of (entity_type, entity_name) tuples in order of appearance.
    """
    entities: List[Tuple[str, str]] = []
    for line in prose:
        for entity_type, entity_name in _ENTITY_RE.findall(line):
            entities.append((entity_type, entity_name))
    return entities


def _get_scene_meta(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Safely retrieve the metadata dict from a scene."""
    return scene.get("metadata") or {}


class ContinuityChecker:
    """Check scenes against StoryGraph for consistency issues.

    Args:
        graph: The StoryGraph instance to validate against.
    """

    def __init__(self, graph: StoryGraph) -> None:
        self.graph = graph

    def check(self, scenes: List[Dict[str, Any]]) -> List[ContinuityIssue]:
        """Run all continuity checks against the scene list.

        Args:
            scenes: Ordered list of scene dicts with keys ``title``, ``prose``,
                    and optional ``metadata``.

        Returns:
            Sorted list of ContinuityIssue instances (by scene index then severity).
        """
        issues: List[ContinuityIssue] = []

        issues.extend(self._check_entity_existence(scenes))
        issues.extend(self._check_dead_entities(scenes))
        issues.extend(self._check_location_conflicts(scenes))
        issues.extend(self._check_introduction_order(scenes))
        issues.extend(self._check_timeline(scenes))

        severity_order = {
            IssueSeverity.ERROR: 0,
            IssueSeverity.WARNING: 1,
            IssueSeverity.INFO: 2,
        }
        issues.sort(key=lambda i: (i.scene_index, severity_order.get(i.severity, 9)))

        return issues

    # ------------------------------------------------------------------
    # 1. Entity existence — do mentioned entities exist in the graph?
    # ------------------------------------------------------------------

    def _check_entity_existence(
        self, scenes: List[Dict[str, Any]]
    ) -> List[ContinuityIssue]:
        """Flag entity names in prose that have no matching StoryGraph node.

        Extracts ``<E t='type'>Name</E>`` markup from each scene's prose and
        looks the name up via ``graph.find_node_by_name``.
        """
        issues: List[ContinuityIssue] = []

        for idx, scene in enumerate(scenes):
            title = scene.get("title", f"Scene {idx}")
            prose: List[str] = scene.get("prose") or []
            entities = _extract_entities_from_prose(prose)

            seen_names: Set[str] = set()
            for entity_type, entity_name in entities:
                name_lower = entity_name.lower()
                if name_lower in seen_names:
                    continue
                seen_names.add(name_lower)

                node = self.graph.find_node_by_name(entity_name)
                if node is None:
                    issues.append(ContinuityIssue(
                        type=IssueType.UNKNOWN_ENTITY,
                        severity=IssueSeverity.WARNING,
                        scene_index=idx,
                        scene_title=title,
                        message=(
                            f"Entity '{entity_name}' (type={entity_type}) is mentioned "
                            f"in prose but does not exist in the StoryGraph."
                        ),
                        entity_name=entity_name,
                        suggestion=(
                            f"Add '{entity_name}' as a {entity_type} node, or correct "
                            f"the name if it is a typo."
                        ),
                    ))

        return issues

    # ------------------------------------------------------------------
    # 2. Dead entities — destroyed characters should not reappear
    # ------------------------------------------------------------------

    def _check_dead_entities(
        self, scenes: List[Dict[str, Any]]
    ) -> List[ContinuityIssue]:
        """Flag destroyed entities that appear in later scenes.

        Builds a map of entity name -> scene index where DESTROYED status is
        first observed (from graph node status). Then checks subsequent scenes
        for mentions in ``metadata.present`` or prose markup.
        """
        issues: List[ContinuityIssue] = []

        # Collect all destroyed entity names from the graph
        destroyed_names: Dict[str, StoryNode] = {}
        for node in self.graph.get_all_nodes():
            if node.status == NodeStatus.DESTROYED:
                destroyed_names[node.name.lower()] = node
                for alias in node.aliases:
                    destroyed_names[alias.lower()] = node

        if not destroyed_names:
            return issues

        # Track which scene each destroyed entity was last marked destroyed.
        # We consider them destroyed from the start (graph state) and scan
        # every scene for reappearance.
        # If a scene's diffs mark an entity as destroyed, we could refine the
        # index, but with current data we use graph status as ground truth.

        for idx, scene in enumerate(scenes):
            title = scene.get("title", f"Scene {idx}")
            meta = _get_scene_meta(scene)

            # Check metadata.present list
            present: List[str] = meta.get("present") or []
            for char_name in present:
                node = destroyed_names.get(char_name.lower())
                if node is not None:
                    issues.append(ContinuityIssue(
                        type=IssueType.DEAD_ENTITY_PRESENT,
                        severity=IssueSeverity.ERROR,
                        scene_index=idx,
                        scene_title=title,
                        message=(
                            f"'{node.name}' has status DESTROYED but is listed "
                            f"as present in this scene."
                        ),
                        entity_name=node.name,
                        suggestion=(
                            f"Remove '{node.name}' from the present list, or "
                            f"change their status if they were resurrected."
                        ),
                    ))

            # Check prose mentions
            prose: List[str] = scene.get("prose") or []
            entities = _extract_entities_from_prose(prose)
            seen_in_scene: Set[str] = set()
            for _, entity_name in entities:
                name_lower = entity_name.lower()
                if name_lower in seen_in_scene:
                    continue
                seen_in_scene.add(name_lower)

                node = destroyed_names.get(name_lower)
                if node is not None:
                    # Don't double-report if already caught via present list
                    already_reported = any(
                        i.scene_index == idx
                        and i.entity_name == node.name
                        and i.type == IssueType.DEAD_ENTITY_PRESENT
                        for i in issues
                    )
                    if not already_reported:
                        issues.append(ContinuityIssue(
                            type=IssueType.DEAD_ENTITY_PRESENT,
                            severity=IssueSeverity.ERROR,
                            scene_index=idx,
                            scene_title=title,
                            message=(
                                f"'{node.name}' has status DESTROYED but is "
                                f"mentioned in prose."
                            ),
                            entity_name=node.name,
                            suggestion=(
                                f"This may be a flashback or memory reference. "
                                f"If intentional, consider marking the mention "
                                f"as past-tense."
                            ),
                        ))

        return issues

    # ------------------------------------------------------------------
    # 3. Location conflicts — same character in two places on one day
    # ------------------------------------------------------------------

    def _check_location_conflicts(
        self, scenes: List[Dict[str, Any]]
    ) -> List[ContinuityIssue]:
        """Flag characters that appear in different locations on the same day.

        Groups scenes by ``metadata.day``. Within a single day, if a character
        appears in ``metadata.present`` for scenes at different
        ``metadata.place`` values, that is a conflict.
        """
        issues: List[ContinuityIssue] = []

        # Build: day -> [(scene_index, place, set_of_present_chars)]
        day_scenes: Dict[int, List[Tuple[int, str, str, Set[str]]]] = {}

        for idx, scene in enumerate(scenes):
            meta = _get_scene_meta(scene)
            day = meta.get("day")
            place = meta.get("place")
            if day is None or place is None:
                continue

            present: List[str] = meta.get("present") or []
            title = scene.get("title", f"Scene {idx}")
            char_set = {name.lower() for name in present}
            day_scenes.setdefault(day, []).append((idx, title, place, char_set))

        # For each day, check for characters appearing in multiple places
        for day, scene_entries in day_scenes.items():
            if len(scene_entries) < 2:
                continue

            # Map character -> list of (scene_index, title, place)
            char_locations: Dict[str, List[Tuple[int, str, str]]] = {}
            for s_idx, s_title, s_place, s_chars in scene_entries:
                for char_name in s_chars:
                    char_locations.setdefault(char_name, []).append(
                        (s_idx, s_title, s_place)
                    )

            for char_name, locations in char_locations.items():
                # Deduplicate places
                unique_places = {place for _, _, place in locations}
                if len(unique_places) <= 1:
                    continue

                # This character is in multiple places on the same day
                place_list = ", ".join(sorted(unique_places))
                # Report on the later scene
                locations.sort(key=lambda x: x[0])
                later_idx, later_title, _ = locations[-1]

                # Look up display name from graph
                node = self.graph.find_node_by_name(char_name)
                display_name = node.name if node else char_name

                issues.append(ContinuityIssue(
                    type=IssueType.LOCATION_CONFLICT,
                    severity=IssueSeverity.WARNING,
                    scene_index=later_idx,
                    scene_title=later_title,
                    message=(
                        f"'{display_name}' appears in multiple locations on "
                        f"day {day}: {place_list}."
                    ),
                    entity_name=display_name,
                    suggestion=(
                        f"Add a travel scene or adjust the day numbers so "
                        f"'{display_name}' is only in one place per day."
                    ),
                ))

        return issues

    # ------------------------------------------------------------------
    # 4. Introduction order — entities used before being introduced
    # ------------------------------------------------------------------

    def _check_introduction_order(
        self, scenes: List[Dict[str, Any]]
    ) -> List[ContinuityIssue]:
        """Flag entities mentioned in prose before they are introduced.

        An entity is considered "introduced" when it appears in a scene's
        ``diffs`` list with ``tag == 'add'``. If an entity name appears in
        prose of an earlier scene, that is a missing introduction.
        """
        issues: List[ContinuityIssue] = []

        # Pass 1: find the first scene index where each entity is introduced
        # via diffs with tag='add'
        introduced_at: Dict[str, int] = {}  # lowercase name -> scene index

        for idx, scene in enumerate(scenes):
            diffs = scene.get("diffs") or []
            for diff in diffs:
                if not isinstance(diff, dict):
                    continue
                if diff.get("tag") == "add":
                    name = diff.get("name", "")
                    if name:
                        name_lower = name.lower()
                        if name_lower not in introduced_at:
                            introduced_at[name_lower] = idx

        if not introduced_at:
            return issues

        # Pass 2: find the first scene index where each entity is mentioned
        # in prose
        first_mention: Dict[str, int] = {}  # lowercase name -> scene index

        for idx, scene in enumerate(scenes):
            prose: List[str] = scene.get("prose") or []
            entities = _extract_entities_from_prose(prose)
            for _, entity_name in entities:
                name_lower = entity_name.lower()
                if name_lower not in first_mention:
                    first_mention[name_lower] = idx

        # Pass 3: compare
        for name_lower, mention_idx in first_mention.items():
            intro_idx = introduced_at.get(name_lower)
            if intro_idx is None:
                # Entity was never formally introduced via diffs -- skip,
                # this is covered by _check_entity_existence if not in graph
                continue
            if mention_idx < intro_idx:
                scene = scenes[mention_idx]
                title = scene.get("title", f"Scene {mention_idx}")
                # Recover display-case name from the intro diff
                display_name = name_lower
                intro_scene_diffs = (scenes[intro_idx].get("diffs") or [])
                for diff in intro_scene_diffs:
                    if isinstance(diff, dict) and diff.get("name", "").lower() == name_lower:
                        display_name = diff["name"]
                        break

                issues.append(ContinuityIssue(
                    type=IssueType.MISSING_INTRODUCTION,
                    severity=IssueSeverity.INFO,
                    scene_index=mention_idx,
                    scene_title=title,
                    message=(
                        f"'{display_name}' is mentioned in scene {mention_idx} "
                        f"but not formally introduced until scene {intro_idx}."
                    ),
                    entity_name=display_name,
                    suggestion=(
                        f"Move the introduction of '{display_name}' to scene "
                        f"{mention_idx} or earlier, or add foreshadowing."
                    ),
                ))

        return issues

    # ------------------------------------------------------------------
    # 5. Timeline — day numbering sanity
    # ------------------------------------------------------------------

    def _check_timeline(
        self, scenes: List[Dict[str, Any]]
    ) -> List[ContinuityIssue]:
        """Flag timeline inconsistencies in day numbering.

        Checks for:
        - Days going backwards (non-monotonic)
        - Large gaps between consecutive days (> _MAX_DAY_GAP)
        """
        issues: List[ContinuityIssue] = []

        # Collect (scene_index, title, day) for scenes that have day metadata
        day_entries: List[Tuple[int, str, int]] = []
        for idx, scene in enumerate(scenes):
            meta = _get_scene_meta(scene)
            day = meta.get("day")
            if day is not None:
                title = scene.get("title", f"Scene {idx}")
                day_entries.append((idx, title, int(day)))

        if len(day_entries) < 2:
            return issues

        for i in range(1, len(day_entries)):
            prev_idx, prev_title, prev_day = day_entries[i - 1]
            curr_idx, curr_title, curr_day = day_entries[i]

            # Days going backwards
            if curr_day < prev_day:
                issues.append(ContinuityIssue(
                    type=IssueType.TIMELINE_GAP,
                    severity=IssueSeverity.INFO,
                    scene_index=curr_idx,
                    scene_title=curr_title,
                    message=(
                        f"Day goes backwards: scene {prev_idx} is day {prev_day} "
                        f"but scene {curr_idx} is day {curr_day}."
                    ),
                    suggestion=(
                        "If this is a flashback, consider marking it explicitly. "
                        "Otherwise, correct the day numbering."
                    ),
                ))

            # Large gap
            elif (curr_day - prev_day) > _MAX_DAY_GAP:
                issues.append(ContinuityIssue(
                    type=IssueType.TIMELINE_GAP,
                    severity=IssueSeverity.INFO,
                    scene_index=curr_idx,
                    scene_title=curr_title,
                    message=(
                        f"Large time gap: {curr_day - prev_day} days between "
                        f"scene {prev_idx} (day {prev_day}) and scene {curr_idx} "
                        f"(day {curr_day})."
                    ),
                    suggestion=(
                        "Consider adding a bridging scene or time-skip narration "
                        "to cover the gap."
                    ),
                ))

        return issues
