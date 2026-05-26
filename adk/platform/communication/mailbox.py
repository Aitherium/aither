import os
import json
import datetime
import uuid
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# ARTIFACT SYSTEM
# ============================================================================

# CONSOLIDATION: Use unified Artifact base class
try:
    from lib.models.artifact import (
        Artifact as BaseArtifact,
        ArtifactType as UnifiedArtifactType,
        MailboxArtifact,
    )
    _UNIFIED_ARTIFACT_AVAILABLE = True
except ImportError:
    _UNIFIED_ARTIFACT_AVAILABLE = False
    BaseArtifact = None
    UnifiedArtifactType = None
    MailboxArtifact = None

# Backward compatibility: Keep ArtifactType enum for ADK
class ArtifactType(Enum):
    """Types of artifacts that can be referenced in messages."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CODE = "code"
    FILE = "file"
    LINK = "link"
    MESSAGE = "message"  # Reference to another message
    MEMORY = "memory"    # Reference to a memory entry


if _UNIFIED_ARTIFACT_AVAILABLE:
    @dataclass
    class Artifact(MailboxArtifact):
        """
        Represents a referenceable artifact in the mailbox system.
        
        CONSOLIDATION: Now inherits from lib.models.artifact.MailboxArtifact
        to use the unified artifact model across AitherOS.
        
        Artifacts can be images, files, code snippets, links, or message references.
        """
        def __post_init__(self):
            super().__post_init__()
            # Map ADK ArtifactType to unified ArtifactType if needed
            if hasattr(self, 'type') and isinstance(self.type, ArtifactType):
                type_map = {
                    ArtifactType.IMAGE: UnifiedArtifactType.IMAGE,
                    ArtifactType.VIDEO: UnifiedArtifactType.VIDEO,
                    ArtifactType.AUDIO: UnifiedArtifactType.AUDIO,
                    ArtifactType.CODE: UnifiedArtifactType.CODE,
                    ArtifactType.FILE: UnifiedArtifactType.FILE,
                    ArtifactType.LINK: UnifiedArtifactType.OTHER,  # No LINK in unified
                    ArtifactType.MESSAGE: UnifiedArtifactType.OTHER,
                    ArtifactType.MEMORY: UnifiedArtifactType.OTHER,
                }
                self.artifact_type = type_map.get(self.type, UnifiedArtifactType.UNKNOWN)
else:
    # Fallback if unified artifact not available (shouldn't happen in normal operation)
    @dataclass
    class Artifact:
        """
        Represents a referenceable artifact in the mailbox system.
        Artifacts can be images, files, code snippets, links, or message references.
        """
        id: str
        type: ArtifactType
        name: str
        path: Optional[str] = None  # File path for images/files
        content: Optional[str] = None  # Content for code/text
        url: Optional[str] = None  # URL for links
        metadata: Optional[Dict[str, Any]] = None  # Additional metadata
        created_at: Optional[str] = None
        created_by: Optional[str] = None  # Which agent created this
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, ArtifactType) else self.type,
            "name": self.name,
            "path": self.path,
            "content": self.content,
            "url": self.url,
            "metadata": self.metadata or {},
            "created_at": self.created_at or datetime.datetime.now().isoformat(),
            "created_by": self.created_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Artifact':
        """Create Artifact from dictionary."""
        artifact_type = data.get("type", "file")
        if isinstance(artifact_type, str):
            try:
                artifact_type = ArtifactType(artifact_type)
            except ValueError:
                artifact_type = ArtifactType.FILE
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=artifact_type,
            name=data.get("name", "Unknown"),
            path=data.get("path"),
            content=data.get("content"),
            url=data.get("url"),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            created_by=data.get("created_by")
        )
    
    def to_reference(self) -> str:
        """Get the #reference format for this artifact."""
        return f"#{self.id[:8]}"
    
    def to_display(self) -> str:
        """Get a display string for the artifact."""
        type_emoji = {
            ArtifactType.IMAGE: "",
            ArtifactType.VIDEO: "[MOVIE]",
            ArtifactType.AUDIO: "[LOUD]",
            ArtifactType.CODE: "[PC]",
            ArtifactType.FILE: "[FILE]",
            ArtifactType.LINK: "[LINK]",
            ArtifactType.MESSAGE: "[MSG]",
            ArtifactType.MEMORY: "[BRAIN]"
        }
        emoji = type_emoji.get(self.type, "[CLIP]")
        return f"{emoji} {self.name} ({self.to_reference()})"


class ArtifactStore:
    """
    Central store for all artifacts created by agents.
    Allows referencing artifacts across messages with #id syntax.
    """
    
    def __init__(self, storage_path: str = "artifacts.json"):
        self.storage_path = storage_path
        self.artifacts: Dict[str, Artifact] = {}
        self.load()
    
    def load(self):
        """Load artifacts from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.artifacts = {
                        k: Artifact.from_dict(v) for k, v in data.items()
                    }
            except Exception as e:
                print(f"Error loading artifacts: {e}")
                self.artifacts = {}
    
    def save(self):
        """Save artifacts to disk."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                data = {k: v.to_dict() for k, v in self.artifacts.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving artifacts: {e}")
    
    def add(self, artifact: Artifact) -> str:
        """Add an artifact and return its ID."""
        self.artifacts[artifact.id] = artifact
        self.save()
        return artifact.id
    
    def create_image_artifact(self, path: str, name: str = None, 
                               created_by: str = None) -> Artifact:
        """Create an image artifact from a file path."""
        artifact = Artifact(
            id=str(uuid.uuid4()),
            type=ArtifactType.IMAGE,
            name=name or os.path.basename(path),
            path=path,
            created_by=created_by,
            created_at=datetime.datetime.now().isoformat()
        )
        self.add(artifact)
        return artifact
    
    def create_code_artifact(self, content: str, name: str, 
                              language: str = None, created_by: str = None) -> Artifact:
        """Create a code snippet artifact."""
        artifact = Artifact(
            id=str(uuid.uuid4()),
            type=ArtifactType.CODE,
            name=name,
            content=content,
            metadata={"language": language} if language else None,
            created_by=created_by,
            created_at=datetime.datetime.now().isoformat()
        )
        self.add(artifact)
        return artifact
    
    def create_link_artifact(self, url: str, name: str = None,
                              created_by: str = None) -> Artifact:
        """Create a link artifact."""
        artifact = Artifact(
            id=str(uuid.uuid4()),
            type=ArtifactType.LINK,
            name=name or url[:50],
            url=url,
            created_by=created_by,
            created_at=datetime.datetime.now().isoformat()
        )
        self.add(artifact)
        return artifact
    
    def create_message_reference(self, message_id: str, subject: str,
                                   created_by: str = None) -> Artifact:
        """Create a reference to another message."""
        artifact = Artifact(
            id=str(uuid.uuid4()),
            type=ArtifactType.MESSAGE,
            name=f"Re: {subject[:40]}",
            metadata={"message_id": message_id},
            created_by=created_by,
            created_at=datetime.datetime.now().isoformat()
        )
        self.add(artifact)
        return artifact
    
    def get(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID (supports short IDs)."""
        # Try exact match first
        if artifact_id in self.artifacts:
            return self.artifacts[artifact_id]
        
        # Try short ID match (first 8 chars)
        for full_id, artifact in self.artifacts.items():
            if full_id.startswith(artifact_id) or artifact_id.startswith(full_id[:8]):
                return artifact
        
        return None
    
    def get_by_type(self, artifact_type: ArtifactType) -> List[Artifact]:
        """Get all artifacts of a specific type."""
        return [a for a in self.artifacts.values() if a.type == artifact_type]
    
    def get_by_creator(self, created_by: str) -> List[Artifact]:
        """Get all artifacts created by a specific agent."""
        return [a for a in self.artifacts.values() 
                if a.created_by and a.created_by.lower() == created_by.lower()]
    
    def get_recent(self, limit: int = 10) -> List[Artifact]:
        """Get most recent artifacts."""
        sorted_artifacts = sorted(
            self.artifacts.values(),
            key=lambda x: x.created_at or "",
            reverse=True
        )
        return sorted_artifacts[:limit]
    
    def resolve_references(self, content: str) -> str:
        """
        Resolve #artifact references in content to display format.
        Example: "Check #abc123" -> "Check  selfie.png (#abc123)"
        """
        pattern = r'#([a-f0-9]{8})'
        
        def replace_ref(match):
            ref_id = match.group(1)
            artifact = self.get(ref_id)
            if artifact:
                return artifact.to_display()
            return match.group(0)  # Keep original if not found
        
        return re.sub(pattern, replace_ref, content)
    
    def extract_references(self, content: str) -> List[str]:
        """Extract all #artifact references from content."""
        pattern = r'#([a-f0-9]{8})'
        return re.findall(pattern, content)


# Global artifact store instance
_artifact_store: Optional[ArtifactStore] = None

def get_artifact_store(storage_path: str = None) -> ArtifactStore:
    """Get the global artifact store instance."""
    global _artifact_store
    if _artifact_store is None:
        path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Saga", "artifacts.json"
        )
        _artifact_store = ArtifactStore(path)
    return _artifact_store


# ============================================================================
# MAILBOX CLASS
# ============================================================================

class Mailbox:
    def __init__(self, storage_path: str = "mailbox.json"):
        self.storage_path = storage_path
        self.messages: List[Dict] = []
        self.load()

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
            except Exception as e:
                print(f"Error loading mailbox: {e}")
                self.messages = []

    def save(self):
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2)
        except Exception as e:
            print(f"Error saving mailbox: {e}")

    def send_message(self, sender: str, recipient: str, subject: str, content: str, 
                     reply_to: Optional[str] = None, mentions: Optional[List[str]] = None,
                     artifacts: Optional[List[str]] = None, 
                     artifact_refs: Optional[List[Dict]] = None):
        """
        Send a message to the mailbox.
        
        Args:
            sender: The agent sending the message
            recipient: The recipient ("user", "all", or agent name)
            subject: Message subject
            content: Message content
            reply_to: Optional - who the agent is directly responding to (for @mentions)
            mentions: Optional - list of agents/users to @mention in the response
            artifacts: Optional - list of artifact IDs referenced in this message
            artifact_refs: Optional - list of artifact dicts to create and attach
        """
        # Create any new artifacts
        created_artifact_ids = []
        if artifact_refs:
            store = get_artifact_store()
            for ref in artifact_refs:
                artifact_type = ref.get("type", "file")
                if artifact_type == "image":
                    artifact = store.create_image_artifact(
                        path=ref.get("path", ""),
                        name=ref.get("name"),
                        created_by=sender
                    )
                elif artifact_type == "code":
                    artifact = store.create_code_artifact(
                        content=ref.get("content", ""),
                        name=ref.get("name", "code"),
                        language=ref.get("language"),
                        created_by=sender
                    )
                elif artifact_type == "link":
                    artifact = store.create_link_artifact(
                        url=ref.get("url", ""),
                        name=ref.get("name"),
                        created_by=sender
                    )
                else:
                    # Generic artifact
                    artifact = Artifact(
                        id=str(uuid.uuid4()),
                        type=ArtifactType.FILE,
                        name=ref.get("name", "artifact"),
                        path=ref.get("path"),
                        content=ref.get("content"),
                        url=ref.get("url"),
                        metadata=ref.get("metadata"),
                        created_by=sender,
                        created_at=datetime.datetime.now().isoformat()
                    )
                    store.add(artifact)
                created_artifact_ids.append(artifact.id)
        
        # Combine existing artifact refs with newly created ones
        all_artifacts = list(artifacts or []) + created_artifact_ids
        
        # Extract any #references from the content
        store = get_artifact_store()
        content_refs = store.extract_references(content)
        for ref in content_refs:
            if ref not in all_artifacts:
                all_artifacts.append(ref)
        
        message = {
            "id": str(uuid.uuid4()),
            "sender": sender,
            "recipient": recipient,
            "subject": subject,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
            "read": False,
            "reply_to": reply_to,  # Track who we're responding to
            "mentions": mentions or [],  # List of @mentioned entities
            "artifacts": all_artifacts  # List of artifact IDs
        }
        
        # PUSH ACTIVITY: Agent is sending a message
        try:
            from lib.core.FluxEmitter import inject_agent_activity
            inject_agent_activity(sender.lower(), {
                "state": "communicating",
                "task": f"Sending message to {recipient}: {subject[:50]}...",
                "will": "default",
                "mailbox": {"recipient": recipient, "subject": subject[:50]},
            })
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to push mailbox activity: {e}")
        self.messages.append(message)
        self.save()
        return message
    
    def format_response_with_mentions(self, sender: str, content: str, 
                                       reply_to: Optional[str] = None,
                                       additional_mentions: Optional[List[str]] = None,
                                       artifact_ids: Optional[List[str]] = None) -> str:
        """
        Format a response with proper @mentions at the beginning and #artifact references.
        Ensures the agent properly addresses who they're responding to.
        
        Args:
            sender: The agent sending the response
            content: The response content
            reply_to: Who this is a reply to (will be @mentioned)
            additional_mentions: Additional entities to @mention
            artifact_ids: Artifact IDs to append as references
        
        Returns:
            Formatted content with @mentions prepended and artifacts appended
        """
        mentions = []
        
        # Always @mention who we're replying to
        if reply_to:
            # Normalize the mention (admin -> Admin, user -> Admin, etc.)
            mention_name = "Admin" if reply_to.lower() in ["user", "admin", "human"] else reply_to.title()
            mentions.append(f"@{mention_name}")
        
        # Add any additional mentions
        if additional_mentions:
            for m in additional_mentions:
                mention_name = "Admin" if m.lower() in ["user", "admin", "human"] else m.title()
                if f"@{mention_name}" not in mentions:
                    mentions.append(f"@{mention_name}")
        
        # Check if content already starts with any of these mentions
        content_stripped = content.strip()
        for mention in mentions:
            if content_stripped.lower().startswith(mention.lower()):
                # Already has the mention, don't duplicate
                content_stripped = content_stripped[len(mention):].strip()
                break
        
        # Build final content
        result = ""
        
        # Prepend mentions
        if mentions:
            result = " ".join(mentions) + " "
        
        result += content_stripped
        
        # Append artifact references
        if artifact_ids:
            store = get_artifact_store()
            artifact_displays = []
            for aid in artifact_ids:
                artifact = store.get(aid)
                if artifact:
                    artifact_displays.append(artifact.to_display())
            
            if artifact_displays:
                result += "\n\n[CLIP] **Attachments:**\n" + "\n".join(f"  * {d}" for d in artifact_displays)
        
        return result
    
    def get_message_with_resolved_artifacts(self, message_id: str) -> Optional[Dict]:
        """
        Get a message with all #artifact references resolved to display format.
        """
        self.load()
        for m in self.messages:
            if m['id'] == message_id:
                store = get_artifact_store()
                resolved = m.copy()
                resolved['content_resolved'] = store.resolve_references(m.get('content', ''))
                
                # Also resolve artifact list to full objects
                if m.get('artifacts'):
                    resolved['artifact_objects'] = [
                        store.get(aid).to_dict() if store.get(aid) else {"id": aid, "error": "not found"}
                        for aid in m['artifacts']
                    ]
                
                return resolved
        return None
    
    def get_artifacts_for_message(self, message_id: str) -> List[Artifact]:
        """Get all artifacts attached to a message."""
        self.load()
        for m in self.messages:
            if m['id'] == message_id:
                artifact_ids = m.get('artifacts', [])
                store = get_artifact_store()
                return [store.get(aid) for aid in artifact_ids if store.get(aid)]
        return []

    def get_messages(self, recipient: str = None, unread_only: bool = False, 
                     search_query: str = None, sender: str = None, 
                     sort_by: str = "timestamp", sort_order: str = "asc") -> List[Dict]:
        """
        Get messages with optional filtering and sorting.
        
        Args:
            recipient: Filter by recipient
            unread_only: Only return unread messages
            search_query: Search in subject and content (case-insensitive)
            sender: Filter by sender name
            sort_by: "timestamp", "sender", "subject", "read"
            sort_order: "asc" or "desc"
        """
        # Always reload from disk to get latest messages from background threads
        self.load()
        
        filtered = self.messages
        if recipient:
            filtered = [m for m in filtered if m['recipient'] == recipient or m['recipient'] == 'all']

        if unread_only:
            filtered = [m for m in filtered if not m['read']]
        
        if sender:
            filtered = [m for m in filtered if sender.lower() in m.get('sender', '').lower()]
        
        if search_query:
            query_lower = search_query.lower()
            filtered = [
                m for m in filtered 
                if query_lower in m.get('subject', '').lower() 
                or query_lower in m.get('content', '').lower()
            ]

        # Sort messages
        reverse_order = (sort_order.lower() == "desc")
        
        if sort_by == "timestamp":
            sorted_messages = sorted(filtered, key=lambda x: x.get('timestamp', ''), reverse=reverse_order)
        elif sort_by == "sender":
            sorted_messages = sorted(filtered, key=lambda x: x.get('sender', '').lower(), reverse=reverse_order)
        elif sort_by == "subject":
            sorted_messages = sorted(filtered, key=lambda x: x.get('subject', '').lower(), reverse=reverse_order)
        elif sort_by == "read":
            # Unread first if ascending, read first if descending
            sorted_messages = sorted(filtered, key=lambda x: x.get('read', False), reverse=not reverse_order)
        else:
            sorted_messages = filtered

        return sorted_messages

    def mark_as_read(self, message_id: str):
        for m in self.messages:
            if m['id'] == message_id:
                m['read'] = True
                self.save()
                return True
        return False

    def clear_messages(self):
        self.messages = []
        self.save()

    def get_unread_count(self, recipient: str = None) -> int:
        # get_messages already reloads from disk
        return len(self.get_messages(recipient, unread_only=True))
    
    def refresh(self):
        """Force reload from disk."""
        self.load()
