import os
from aither_adk.memory.memory import MemoryManager
from aither_adk.infrastructure.tasks import TaskManager
from aither_adk.ai.ollama_service import OllamaService
from aither_adk.ai.comfyui_service import ComfyUIService
from aither_adk.ui.console import safe_print
from aither_adk.infrastructure.dependencies import check_and_report_dependencies
from aither_adk.infrastructure.services import get_service_status, is_aithernode_running, is_ollama_running, is_comfyui_running

def common_on_startup(agent, agent_root_dir, use_local_models=False):
    """
    Common startup routine for agents.
    
    FAST PATH: If AitherNode is running, skip most initialization since
    those services are already available via the MCP server.

    Args:
        agent: The agent instance.
        agent_root_dir: The root directory of the specific agent (where memory/ exists).
        use_local_models: Boolean indicating if local models are being used.

    Returns:
        dict: A dictionary of initialized components (memory_manager, task_manager).
    """
    # Quick service status check (uses cached socket checks)
    services = get_service_status()
    
    # If AitherNode is running, skip heavy initialization
    if services.aithernode:
        # Build list of connected services
        connected = ["AitherNode:8080"]
        if services.aitherllm:
            connected.append("MicroScheduler:8150")
        if services.ollama:
            connected.append("Ollama:11434")
        if services.comfyui:
            connected.append("ComfyUI:8188")
        if services.reasoning:
            connected.append("Reasoning:8093")
        
        safe_print(f"[dim][ZAP] Connected to shared AitherOS services: {', '.join(connected)}[/]")
        
        # Still need local memory for this agent instance
        memory_file = os.path.join(agent_root_dir, "memory", "long_term_memory.json")
        memory_manager = MemoryManager(memory_file)
        
        tasks_file = os.path.join(agent_root_dir, "memory", "tasks.json")
        task_manager = TaskManager(tasks_file)
        
        return {
            'memory_manager': memory_manager,
            'task_manager': task_manager,
            'dependency_info': {'degraded': False, 'missing_required': [], 'report': ''},
            'services': services
        }
    
    # SLOW PATH: Full initialization when services not running
    safe_print(f"[dim] Full startup: Initializing services locally...[/]")
    
    # Check Dependencies
    dep_info = check_and_report_dependencies()
    if dep_info['degraded']:
        safe_print("\n[bold yellow]Dependency Check:[/bold yellow]")
        safe_print(dep_info['report'])
        safe_print("")

    if dep_info['missing_required']:
        safe_print("[bold red]CRITICAL: Cannot proceed without required dependencies.[/]")

    # Initialize Memory
    memory_file = os.path.join(agent_root_dir, "memory", "long_term_memory.json")
    memory_manager = MemoryManager(memory_file)

    # Initialize Task Manager
    tasks_file = os.path.join(agent_root_dir, "memory", "tasks.json")
    task_manager = TaskManager(tasks_file)

    # Check Ollama Connectivity if using local models (only if not already running)
    if use_local_models and not services.ollama:
        OllamaService.ensure_running()
        safe_print(f"Ollama Status: {OllamaService.get_status()}")
        
        # Ensure the configured local model is available
        if hasattr(agent, 'model') and agent.model:
             from aither_adk.ai.models import is_local_model
             if is_local_model(agent.model):
                 OllamaService.ensure_model(agent.model)
    elif services.ollama:
        safe_print(f"[dim][DONE] Ollama already running[/]")

    # Check ComfyUI Connectivity (only if not already running)
    if not services.comfyui:
        ComfyUIService.ensure_reachable()
    else:
        safe_print(f"[dim][DONE] ComfyUI already running[/]")

    return {
        'memory_manager': memory_manager,
        'task_manager': task_manager,
        'dependency_info': dep_info,
        'services': services
    }
