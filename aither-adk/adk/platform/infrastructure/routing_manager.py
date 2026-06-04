"""
AitherOS Routing Manager
========================
Dynamic agent routing configuration with runtime control.

Features:
- Load/save routing config from YAML
- Enable/disable agents at runtime
- Add custom agents
- /route command integration
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from threading import Lock

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "Saga" / "config" / "routing.yaml"


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    enabled: bool = True
    description: str = ""
    type: str = "builtin"  # builtin, custom, external
    patterns: List[str] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    priority: int = 0
    use_mcp: bool = True
    endpoint: Optional[str] = None  # For external agents
    
    def __post_init__(self):
        if isinstance(self.keywords, list):
            self.keywords = set(self.keywords)


@dataclass 
class RoutingConfig:
    """Global routing configuration."""
    enabled: bool = True
    default_agent: str = "Aither"
    confidence_threshold: float = 0.5
    gemini_router_fallback: bool = False
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)


class RoutingManager:
    """
    Manages agent routing configuration with runtime control.
    
    Thread-safe singleton that provides:
    - Dynamic agent enable/disable
    - Custom agent registration
    - Pattern-based routing
    - Persistence to YAML
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: Optional[Path] = None):
        if self._initialized:
            return
            
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._config = RoutingConfig()
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._load_config()
        self._initialized = True
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self._config_path.exists():
            # Create default config
            self._save_config()
            return
            
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            # Parse routing settings
            routing = data.get('routing', {})
            self._config.enabled = routing.get('enabled', True)
            self._config.default_agent = routing.get('default_agent', 'Aither')
            self._config.confidence_threshold = routing.get('confidence_threshold', 0.5)
            self._config.gemini_router_fallback = routing.get('gemini_router_fallback', False)
            
            # Parse agents
            self._config.agents = {}
            for name, agent_data in data.get('agents', {}).items():
                self._config.agents[name] = AgentConfig(
                    name=name,
                    enabled=agent_data.get('enabled', True),
                    description=agent_data.get('description', ''),
                    type=agent_data.get('type', 'builtin'),
                    patterns=agent_data.get('patterns', []),
                    keywords=set(agent_data.get('keywords', [])),
                    priority=agent_data.get('priority', 0),
                    use_mcp=agent_data.get('use_mcp', True),
                    endpoint=agent_data.get('endpoint')
                )
            
            # Parse custom agents
            for name, agent_data in data.get('custom_agents', {}).items():
                if agent_data:  # Skip empty entries
                    self._config.agents[name] = AgentConfig(
                        name=name,
                        enabled=agent_data.get('enabled', True),
                        description=agent_data.get('description', ''),
                        type='custom',
                        patterns=agent_data.get('patterns', []),
                        keywords=set(agent_data.get('keywords', [])),
                        priority=agent_data.get('priority', 10),
                        use_mcp=agent_data.get('use_mcp', True),
                        endpoint=agent_data.get('endpoint')
                    )
            
            # Parse aliases
            self._config.aliases = data.get('aliases', {})
            
            # Compile patterns
            self._compile_patterns()
            
        except Exception as e:
            print(f"Warning: Failed to load routing config: {e}")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for all agents."""
        self._compiled_patterns = {}
        for name, agent in self._config.agents.items():
            if agent.enabled:
                self._compiled_patterns[name] = [
                    re.compile(p, re.IGNORECASE) for p in agent.patterns
                ]
    
    def _save_config(self) -> None:
        """Save current configuration to YAML file."""
        # Build data structure
        data = {
            'routing': {
                'enabled': self._config.enabled,
                'default_agent': self._config.default_agent,
                'confidence_threshold': self._config.confidence_threshold,
                'gemini_router_fallback': self._config.gemini_router_fallback,
            },
            'agents': {},
            'custom_agents': {},
            'aliases': self._config.aliases,
        }
        
        for name, agent in self._config.agents.items():
            agent_data = {
                'enabled': agent.enabled,
                'description': agent.description,
                'type': agent.type,
                'patterns': agent.patterns,
                'keywords': list(agent.keywords),
                'priority': agent.priority,
                'use_mcp': agent.use_mcp,
            }
            if agent.endpoint:
                agent_data['endpoint'] = agent.endpoint
                
            if agent.type == 'custom':
                data['custom_agents'][name] = agent_data
            else:
                data['agents'][name] = agent_data
        
        # Ensure directory exists
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self._config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    @property
    def enabled(self) -> bool:
        """Check if routing is enabled globally."""
        return self._config.enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable routing globally."""
        self._config.enabled = value
        self._save_config()
    
    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name or alias."""
        # Check aliases first
        resolved_name = self._config.aliases.get(name.lower(), name)
        return self._config.agents.get(resolved_name)
    
    def get_enabled_agents(self) -> List[AgentConfig]:
        """Get all enabled agents sorted by priority."""
        agents = [a for a in self._config.agents.values() if a.enabled]
        return sorted(agents, key=lambda a: -a.priority)  # Higher priority first
    
    def enable_agent(self, name: str) -> bool:
        """Enable an agent by name."""
        agent = self.get_agent(name)
        if agent:
            agent.enabled = True
            self._compile_patterns()
            self._save_config()
            return True
        return False
    
    def disable_agent(self, name: str) -> bool:
        """Disable an agent by name."""
        agent = self.get_agent(name)
        if agent:
            agent.enabled = False
            self._compile_patterns()
            self._save_config()
            return True
        return False
    
    def add_agent(
        self,
        name: str,
        description: str = "",
        patterns: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        priority: int = 10,
        endpoint: Optional[str] = None,
        use_mcp: bool = True
    ) -> AgentConfig:
        """Add a custom agent."""
        agent = AgentConfig(
            name=name,
            enabled=True,
            description=description,
            type='custom',
            patterns=patterns or [],
            keywords=set(keywords or []),
            priority=priority,
            use_mcp=use_mcp,
            endpoint=endpoint
        )
        self._config.agents[name] = agent
        self._compile_patterns()
        self._save_config()
        return agent
    
    def remove_agent(self, name: str) -> bool:
        """Remove a custom agent."""
        if name in self._config.agents:
            agent = self._config.agents[name]
            if agent.type != 'builtin':
                del self._config.agents[name]
                self._compile_patterns()
                self._save_config()
                return True
        return False
    
    def add_alias(self, alias: str, agent_name: str) -> bool:
        """Add an alias for an agent."""
        if agent_name in self._config.agents:
            self._config.aliases[alias.lower()] = agent_name
            self._save_config()
            return True
        return False
    
    def get_default_agent(self) -> str:
        """Get the default agent name."""
        return self._config.default_agent
    
    def set_default_agent(self, name: str) -> bool:
        """Set the default agent."""
        if name in self._config.agents:
            self._config.default_agent = name
            self._save_config()
            return True
        return False
    
    def get_patterns(self, agent_name: str) -> List[re.Pattern]:
        """Get compiled patterns for an agent."""
        return self._compiled_patterns.get(agent_name, [])
    
    def get_keywords(self, agent_name: str) -> Set[str]:
        """Get keywords for an agent."""
        agent = self.get_agent(agent_name)
        return agent.keywords if agent else set()
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current routing status."""
        return {
            'enabled': self._config.enabled,
            'default_agent': self._config.default_agent,
            'confidence_threshold': self._config.confidence_threshold,
            'gemini_fallback': self._config.gemini_router_fallback,
            'agents': {
                name: {
                    'enabled': agent.enabled,
                    'type': agent.type,
                    'priority': agent.priority,
                    'patterns': len(agent.patterns),
                    'keywords': len(agent.keywords),
                }
                for name, agent in self._config.agents.items()
            },
            'aliases': self._config.aliases,
        }


# Singleton accessor
_routing_manager: Optional[RoutingManager] = None


def get_routing_manager() -> RoutingManager:
    """Get the global routing manager instance."""
    global _routing_manager
    if _routing_manager is None:
        _routing_manager = RoutingManager()
    return _routing_manager
