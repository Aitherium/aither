"""
AitherOS Unified Tool Loader (Service-Aware)
=============================================
Centralized tool imports for all AitherOS agents.

FAST PATH: If AitherNode is already running on port 8080, uses MCP client
to call tools via HTTP - NO heavy module imports, NO re-initialization.
Target: <1s import time.

SLOW PATH: If AitherNode is not running, imports modules directly.
Expected: ~10-15s import time.

PERFORMANCE NOTES:
- google.adk.tools.FunctionTool import takes ~1.5s
- google.cloud.aiplatform import takes ~4s  
- AitherOS.AitherNode.server import takes ~6s (initializes services)
- All these are SKIPPED in fast path
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
# Allow importing from apps/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../apps')))

# Check service status FIRST (fast socket check, <10ms)
# DO NOT import google.adk, google.cloud, or AitherNode here!
from aither_adk.infrastructure.services import is_aithernode_running, get_service_status

_AITHERNODE_RUNNING = is_aithernode_running()

# =============================================================================
# FAST PATH: Use HTTP client when AitherNode is already running
# Skip ALL heavy imports - use lightweight HTTP stubs instead
# Target import time: <1s (vs ~15s for slow path)
# =============================================================================
if _AITHERNODE_RUNNING:
    # Import ONLY lightweight HTTP client stubs - NO google.adk, NO AitherNode
    from aither_adk.tools.mcp_client_tools import (
        # Memory tools (stubs that call HTTP API)
        remember, recall, add_to_working_memory, get_current_context,
        clear_context, list_memory_entries,
        # RAG/Mind tools
        search_knowledge, think, summarize_text,
        # Image/Vision tools
        analyze_image_content, generate_image, refine_image, create_animation,
        compare_images, ask_about_image, extract_text_from_image,
        get_vision_status, unload_vision_model, list_workflows,
        # Ollama tools
        list_ollama_models, chat_ollama,
        # Persona tools
        list_personas, get_persona_details, update_persona,
        generate_persona_profile_picture,
        # Infrastructure tools
        run_script, mcp_get_service_status, get_service_summary,
        # RBAC tools
        rbac_list_users, rbac_get_user, rbac_check_permission, rbac_summary,
        # Neurons/Cortex tools
        think_with_neurons, fire_neuron,
        # Tool lists (lazy-loaded)
        mcp_server_tools,
        # Local generation stubs
        generate_local, ollama_tools, get_vision_backend_status,
        generate_local_response, is_ollama_available,
        # Empty lists for tools not available via HTTP
        civitai_tools, huggingface_tools, video_tools, animation_tools, dataset_tools,
    )
    
    # Stub out aither_tools - these also import google.adk which is slow
    # The actual PowerShell execution happens via run_script MCP tool
    aither_tools = []
    
    # Stub out agent-specific tools that import heavy modules
    # google_genai_client imports google.cloud.aiplatform (~4s)
    generate_google_image = None
    refine_image_with_fal = None
    generate_image_with_fal = None
    generate_video_with_fal = None
    generate_3d_model_from_text = None
    generate_3d_model_from_image = None
    generate_narrative_response = None
    continue_scene = None
    infrastructure_tools = []
    
    # Personal assistant tools work locally (no MCP needed) - import them directly
    # These are lightweight and don't require heavy imports
    try:
        from aither_adk.tools.tools.personal_assistant_tools import (
            personal_assistant_tools, HAS_DDGS, HAS_CLIPBOARD,
            web_search, fetch_webpage_content, get_current_time,
            get_system_stats, get_weather,
        )
    except ImportError:
        personal_assistant_tools = []
        HAS_DDGS = False
        HAS_CLIPBOARD = False
        web_search = None
        fetch_webpage_content = None
        get_current_time = None
        get_system_stats = None
        get_weather = None

# =============================================================================
# SLOW PATH: Direct imports when AitherNode is NOT running
# Expected import time: ~10-15s (initializes all services locally)
# =============================================================================
else:
    # Set lazy load flag to minimize initialization
    os.environ["AITHERNODE_LAZY_LOAD"] = "1"
    
    # Import FunctionTool here (slow ~1.5s, but needed for slow path)
    from google.adk.tools import FunctionTool
    
    # Import aither_tools (also slow, imports FunctionTool)
    from aither_adk.tools.tools.aither_tools import aither_tools
    
    # Initialize as empty first
    remember = recall = add_to_working_memory = get_current_context = clear_context = list_memory_entries = None
    analyze_image_content = generate_image = refine_image = create_animation = compare_images = ask_about_image = extract_text_from_image = get_vision_status = unload_vision_model = list_workflows = None
    list_ollama_models = chat_ollama = None
    list_personas = get_persona_details = update_persona = generate_persona_profile_picture = upload_persona_profile_picture = remove_persona_profile_picture = generate_all_persona_profiles = None
    run_script = mcp_get_service_status = get_service_summary = None
    rbac_list_users = rbac_get_user = rbac_create_user = rbac_update_user = rbac_delete_user = rbac_list_groups = rbac_create_group = rbac_update_group = rbac_delete_group = rbac_add_user_to_group = rbac_remove_user_from_group = rbac_list_roles = rbac_create_role = rbac_delete_role = rbac_check_permission = rbac_get_user_permissions = rbac_summary = None

    try:
        from apps.AitherNode.tools.mcp.mcp_memory import remember, recall, add_to_working_memory, list_memory_entries
        from apps.AitherNode.tools.mcp.mcp_context import get_current_context, clear_context
    except ImportError as e:
        print(f"Warning: Memory slow path: {e}")

    try:
        from apps.AitherNode.tools.mcp.mcp_vision import analyze_image_content, compare_images, ask_about_image, extract_text_from_image, get_vision_status, unload_vision_model
        from apps.AitherNode.tools.mcp.mcp_generation import generate_image, refine_image, create_animation, list_workflows
    except ImportError as e:
        print(f"Warning: Vision slow path: {e}")

    try:
        from apps.AitherNode.tools.mcp.mcp_ollama import list_ollama_models, chat_ollama
    except ImportError as e:
        print(f"Warning: Ollama slow path: {e}")

    try:
        from apps.AitherNode.tools.mcp.mcp_persona import list_personas, get_persona_details, update_persona, generate_persona_profile_picture, upload_persona_profile_picture, remove_persona_profile_picture, generate_all_persona_profiles
    except ImportError as e:
        print(f"Warning: Persona slow path: {e}")

    try:
        from apps.AitherNode.tools.mcp.mcp_commands import run_script
        from apps.AitherNode.tools.mcp.mcp_services import get_service_status as mcp_get_service_status, get_service_summary
    except ImportError as e:
        print(f"Warning: Infra slow path: {e}")

    try:
        from apps.AitherNode.tools.mcp.mcp_rbac import rbac_list_users, rbac_get_user, rbac_create_user, rbac_update_user, rbac_delete_user, rbac_list_groups, rbac_create_group, rbac_update_group, rbac_delete_group, rbac_add_user_to_group, rbac_remove_user_from_group, rbac_list_roles, rbac_create_role, rbac_delete_role, rbac_check_permission, rbac_get_user_permissions, rbac_summary
    except ImportError as e:
        print(f"Warning: RBAC slow path: {e}")
    
    # These imports are optional - stub out if not available
    try:
        from AitherOS.AitherNode.AitherCanvas import generate_local
    except ImportError:
        generate_local = None
    
    try:
        from AitherOS.AitherNode.ollama_tools import ollama_tools
    except ImportError:
        ollama_tools = []
    
    try:
        from AitherOS.AitherNode.vision_tools import get_vision_backend_status
    except ImportError:
        get_vision_backend_status = None
    
    try:
        from AitherOS.AitherNode.local_llm import generate_local_response, is_ollama_available
    except ImportError:
        generate_local_response = None
        is_ollama_available = lambda: False
    
    try:
        from AitherOS.AitherNode.tools.civitai_tools import civitai_tools
    except ImportError:
        civitai_tools = []
    
    try:
        from AitherOS.AitherNode.tools.huggingface_tools import huggingface_tools
    except ImportError:
        huggingface_tools = []
    
    try:
        from AitherOS.AitherNode.tools.video_tools import video_tools
    except ImportError:
        video_tools = []
    
    try:
        from AitherOS.AitherNode.tools.dataset_tools import dataset_tools
    except ImportError:
        dataset_tools = []
    
    try:
        from AitherOS.AitherNode.tools.animation_tools import animation_tools
    except ImportError:
        animation_tools = []
    
    # Build MCP tools list from direct imports
    mcp_server_tools = [
        # Memory
        FunctionTool(remember),
        FunctionTool(recall),
        FunctionTool(add_to_working_memory),
        FunctionTool(get_current_context),
        FunctionTool(clear_context),
        FunctionTool(list_memory_entries),
        # Vision
        FunctionTool(analyze_image_content),
        FunctionTool(compare_images),
        FunctionTool(ask_about_image),
        FunctionTool(extract_text_from_image),
        FunctionTool(get_vision_status),
        # Image Generation
        FunctionTool(generate_image),
        FunctionTool(refine_image),
        FunctionTool(create_animation),
        FunctionTool(list_workflows),
        # Ollama
        FunctionTool(list_ollama_models),
        FunctionTool(chat_ollama),
        # Personas
        FunctionTool(list_personas),
        FunctionTool(get_persona_details),
        FunctionTool(update_persona),
        FunctionTool(generate_persona_profile_picture),
        # Infrastructure
        FunctionTool(run_script),
        FunctionTool(mcp_get_service_status),
        FunctionTool(get_service_summary),
        # RBAC
        FunctionTool(rbac_list_users),
        FunctionTool(rbac_get_user),
        FunctionTool(rbac_check_permission),
        FunctionTool(rbac_summary),
    ]

# =============================================================================
# AGENT-SPECIFIC TOOLS - Only imported in SLOW PATH
# These import google.cloud.aiplatform, google.adk, etc. which are slow
# =============================================================================
if not _AITHERNODE_RUNNING:
    from aither_adk.tools.tools.google_genai_client import generate_google_image
    from aither_adk.tools.tools.fal_tools import refine_image_with_fal, generate_image_with_fal, generate_video_with_fal
    from aither_adk.tools.tools.meshy_tools import generate_3d_model_from_text, generate_3d_model_from_image
    from aither_adk.tools.tools.narrative_tools import generate_narrative_response, continue_scene
    from aither_adk.tools.tools.personal_assistant_tools import personal_assistant_tools, HAS_DDGS, HAS_CLIPBOARD
    from aither_adk.tools.tools.infrastructure_tools import infrastructure_tools

# =============================================================================
# AWARENESS TOOLS - Environment/Sensation/Temporal Awareness
# These are lightweight HTTP calls - import in BOTH paths
# =============================================================================
try:
    from aither_adk.tools.awareness_tools import (
        awareness_tools,
        emit_sensation,
        get_affect_state,
        get_active_sensations,
        subscribe_to_pulse,
        get_pain_dashboard,
        emit_pulse_event,
        get_temporal_context,
        track_operation_duration,
        end_operation_tracking,
        get_environment_awareness,
    )
except ImportError:
    awareness_tools = []
    emit_sensation = None
    get_affect_state = None
    get_active_sensations = None
    subscribe_to_pulse = None
    get_pain_dashboard = None
    emit_pulse_event = None
    get_temporal_context = None
    track_operation_duration = None
    end_operation_tracking = None
    get_environment_awareness = None

# =============================================================================
# FLOW TOOLS - GitHub CI/CD Integration (AitherFlow)
# Lightweight HTTP calls to AitherFlow service (port 8142)
# =============================================================================
try:
    from aither_adk.tools.flow_tools import (
        flow_tools,
        github_list_workflows,
        github_trigger_workflow,
        github_get_workflow_runs,
        github_cancel_workflow,
        github_rerun_workflow,
        github_run_ci_tests,
        github_run_security_scan,
        github_ci_status,
        github_list_prs,
        github_ai_review_pr,
        github_merge_pr,
        github_comment_on_pr,
        github_get_pr_diff,
        github_get_pr_files,
        github_enable_auto_merge,
        github_list_issues,
        github_create_issue,
        github_assign_issue_to_agent,
        github_create_release,
        github_list_releases,
        github_list_labels,
        github_create_label,
        github_delete_label,
        github_list_milestones,
        github_create_milestone,
        github_close_milestone,
        github_get_branch_protection,
        github_update_branch_protection,
        github_update_secret,
        github_list_projects,
    )
except ImportError:
    flow_tools = []
    github_list_workflows = None
    github_trigger_workflow = None
    github_get_workflow_runs = None
    github_cancel_workflow = None
    github_rerun_workflow = None
    github_run_ci_tests = None
    github_run_security_scan = None
    github_ci_status = None
    github_list_prs = None
    github_ai_review_pr = None
    github_merge_pr = None
    github_comment_on_pr = None
    github_get_pr_diff = None
    github_get_pr_files = None
    github_enable_auto_merge = None
    github_list_issues = None
    github_create_issue = None
    github_assign_issue_to_agent = None
    github_create_release = None
    github_list_releases = None
    github_list_labels = None
    github_create_label = None
    github_delete_label = None
    github_list_milestones = None
    github_create_milestone = None
    github_close_milestone = None
    github_get_branch_protection = None
    github_update_branch_protection = None
    github_update_secret = None
    github_list_projects = None

# =============================================================================
# REINFORCEMENT LEARNING TOOLS - Deep RL / Training Pipeline
# Connects agents to AitherHarvest/Judge/Trainer/Evolution
# =============================================================================
try:
    from aither_adk.tools.reinforcement_tools import (
        reinforcement_tools,
        record_interaction_outcome,
        submit_preference_pair,
        capture_reasoning_trace,
        request_quality_judgement,
        get_training_metrics,
        trigger_training_export,
        report_model_improvement,
        InteractionCapture,
    )
except ImportError:
    reinforcement_tools = []
    record_interaction_outcome = None
    submit_preference_pair = None
    capture_reasoning_trace = None
    request_quality_judgement = None
    get_training_metrics = None
    trigger_training_export = None
    report_model_improvement = None
    InteractionCapture = None

# =============================================================================
# ALL TOOLS COMBINED
# =============================================================================
all_tools = (
    list(aither_tools) +
    list(mcp_server_tools) +
    list(civitai_tools) +
    list(huggingface_tools) +
    list(video_tools) +
    list(animation_tools) +
    list(dataset_tools) +
    list(personal_assistant_tools) +
    list(infrastructure_tools) +
    list(awareness_tools) +       # Environment awareness tools (AitherSense/Pulse/TimeSense)
    list(reinforcement_tools) +   # Deep RL tools (Harvest/Judge/Trainer/Evolution)
    list(flow_tools)              # GitHub CI/CD tools (AitherFlow)
)
