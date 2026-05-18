import argparse
import logging
import os

from dotenv import load_dotenv
from google.adk import Runner
from google.adk.apps import App
from google.adk.artifacts import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import (
    InMemoryCredentialService,
)
from google.adk.sessions import InMemorySessionService
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import merge_key_bindings
from prompt_toolkit.styles import Style as PromptStyle
from rich.rule import Rule

from aither_adk.ai.models import get_best_orchestration_model, get_fallback_model
from aither_adk.ai.models import select_model as common_select_model
from aither_adk.communication.mailbox import Mailbox
from aither_adk.infrastructure.auth import configure_auth
from aither_adk.infrastructure.profiling import profile
from aither_adk.infrastructure.runner import process_turn
from aither_adk.infrastructure.services import cleanup_clients
from aither_adk.infrastructure.utils import configure_logging
from aither_adk.ui.commands import SlashCommandCompleter, handle_command
from aither_adk.ui.console import (
    console,
    create_keybindings,
    print_banner,
    safe_print,
)

logger = logging.getLogger(__name__)

async def run_agent_app(
    create_agent_fn,
    models_list,
    personas=None,
    app_name="agents",
    description="Aither Agent",
    extra_args_setup=None,
    extra_args_handler=None,
    on_startup=None,
    prompt_style=None,
    bottom_toolbar_callback=None,
    rich_toolbar_callback=None,
    session_stats=None,
    debug_mode=False,
    get_banner_info=None,
    mailbox=None,
    extra_keybindings=None
):
    """
    Common entry point for running Aither agents.

    Args:
        create_agent_fn: Function that takes (model_name) and returns an Agent instance.
        models_list: List of available model names.
        personas: Dict of personas for autocomplete.
        app_name: Name of the app.
        description: Description for argparse.
        extra_args_setup: Callback(parser) to add extra arguments.
        extra_args_handler: Callback(args) to handle extra arguments. Can return modified models_list.
        on_startup: Callback(agent) to run after agent creation but before loop.
        prompt_style: PromptStyle object.
        debug_mode: Boolean to enable debug mode.
        get_banner_info: Callback() that returns a dict of system info to display in the banner.
        extra_keybindings: Optional KeyBindings object to merge.
    """
    # Load .env
    load_dotenv()
    configure_logging()

    # Initialize mailbox if not provided
    if mailbox is None:
        # Create default mailbox in the common directory
        default_mailbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mailbox.json")
        mailbox = Mailbox(default_mailbox_path)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default=None, help="Model to use.")
    parser.add_argument("--select-model", action="store_true", help="Force interactive model selection.")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Prompt to run non-interactively.")

    if extra_args_setup:
        extra_args_setup(parser)

    args, unknown = parser.parse_known_args()

    # Handle extra args
    skip_auth = False
    if extra_args_handler:
        # Handler can return a tuple (models_list, skip_auth) or just models_list or None
        result = extra_args_handler(args)
        if result:
            if isinstance(result, tuple):
                models_list = result[0]
                if len(result) > 1:
                    skip_auth = result[1]
            elif isinstance(result, list):
                models_list = result

    # Auth
    if not skip_auth and not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
         configure_auth()

    # Consolidate keys
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        # Unset GEMINI_API_KEY to prevent "Both are set" warning from library
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

    # Select Model - use intelligent default if not specified
    # Priority: --model arg > DEFAULT_MODEL env > best orchestration model > models_list[0]
    default_model = os.getenv("DEFAULT_MODEL") or get_best_orchestration_model() or (models_list[0] if models_list else "gemini-2.5-flash")

    model_name = None
    if args.prompt:
        model_name = args.model if args.model else default_model
    elif args.select_model:
        model_name = common_select_model(models_list, args.model)
    else:
        model_name = args.model if args.model else default_model

    if not args.prompt:
        banner_info = None
        if get_banner_info:
            banner_info = get_banner_info()
        print_banner(model_name, system_info=banner_info)

    try:
        # Initialize Agent (outside spinner to allow clean output from startup checks)
        with console.status(f"[bold green]Initializing Agent with {model_name}...[/]", spinner="dots"):
            agent = create_agent_fn(model_name)

        # Run startup tasks (may print to console)
        context = {}
        if on_startup:
            result = on_startup(agent)
            if isinstance(result, dict):
                context = result

        with console.status("[bold green]Starting Session...[/]", spinner="dots"):
            session_service = InMemorySessionService()
            artifact_service = InMemoryArtifactService()
            credential_service = InMemoryCredentialService()

            app = App(name=app_name, root_agent=agent)

            runner = Runner(
                app=app,
                session_service=session_service,
                artifact_service=artifact_service,
                credential_service=credential_service
            )

            user_id = "user"
            session = await runner.session_service.create_session(app_name=app.name, user_id=user_id)
            session_id = session.id

        if not args.prompt:
            console.print(f"[info]Session created: {session_id}[/]")
            console.print("[bold green]>>> Ready. Use Arrow keys to move. Enter for newline. Ctrl+Enter to submit.[/]\n")

        if session_stats is None:
            session_stats = {"total_cost": 0.0, "total_input": 0, "total_output": 0}

        memory_manager = context.get('memory_manager')

        async def process_turn_with_fallback(user_input):
            nonlocal agent, app, runner, model_name
            max_retries = 3
            current_try = 0

            while current_try < max_retries:
                try:
                    await profile(process_turn, name="process_turn")(runner, user_id, session_id, user_input, model_name, session_stats, root_agent=agent, debug_mode=debug_mode, memory_manager=memory_manager, toolbar_renderer=rich_toolbar_callback, mailbox=mailbox)
                    break # Success
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str:
                        safe_print(f"\n[warning][WARN] Quota exceeded for {model_name}.[/]")

                        fallback_model = get_fallback_model(model_name)
                        if fallback_model:
                            safe_print(f"[bold yellow][SYNC] Auto-switching to fallback model: {fallback_model}[/]")

                            # Re-initialize Agent and Runner with new model
                            model_name = fallback_model
                            agent = create_agent_fn(model_name)
                            app = App(name=app_name, root_agent=agent)

                            # Re-create runner but KEEP the session service to preserve history
                            runner = Runner(
                                app=app,
                                session_service=session_service,
                                artifact_service=artifact_service,
                                credential_service=credential_service
                            )

                            current_try += 1
                            continue
                        else:
                            safe_print("[red][FAIL] No fallback model available. Aborting.[/]")
                            raise e
                    else:
                        raise e

        if args.prompt:
            await process_turn_with_fallback(args.prompt)
            return

        # Interactive Loop
        kb = create_keybindings()
        if extra_keybindings:
            kb = merge_key_bindings([kb, extra_keybindings])

        session = PromptSession(
            key_bindings=kb,
            multiline=True,
            style=prompt_style if prompt_style else PromptStyle.from_dict({'prompt': 'ansicyan bold'}),
            completer=SlashCommandCompleter(models_list, personas if personas else {})
        )

        def toggle_debug_mode():
            nonlocal debug_mode
            debug_mode = not debug_mode
            safe_print(f"[bold yellow]Debug Mode: {debug_mode}[/]")

        # Add spacing before starting the loop to prevent UI overlap
        console.print("\n")

        while True:
            try:
                console.print(Rule(style="dim"))
                # Static elegant prompt (reverted animation for stability)
                user_input = await session.prompt_async(
                    HTML("<b><style color='#ffb3d9'>*</style><style color='#00ffff'> </style></b>"),
                    bottom_toolbar=bottom_toolbar_callback
                )

                if not user_input.strip():
                    continue

                if user_input.strip().startswith("/"):
                    result = await handle_command(
                        user_input,
                        session_stats,
                        agent,
                        runner,
                        session_id,
                        models_list,
                        personas if personas else {},
                        toggle_debug_mode,
                        memory_manager=memory_manager,
                        prompt_session=session,
                        mailbox=mailbox
                    )

                    # Update model name in case it was changed by command
                    if hasattr(agent, 'model'):
                        model_name = agent.model

                    if result == "RECREATE_AGENT":
                        # Recreate agent with new safety settings
                        with console.status("[bold green]Reinitializing agent with new safety settings...[/]", spinner="dots"):
                            agent = create_agent_fn(model_name)
                            app = App(name=app_name, root_agent=agent)
                            runner = Runner(
                                app=app,
                                session_service=session_service,
                                artifact_service=artifact_service,
                                credential_service=credential_service
                            )
                        console.print("[green][DONE] Agent reinitialized with new safety settings[/]")
                        continue

                    if result == "EXIT_SIGNAL":
                        # Unload all models before exit
                        try:
                            import requests

                            from lib.core.AitherPorts import ollama_url as get_ollama_url
                            ollama_url = get_ollama_url()
                            ps_resp = requests.get(f"{ollama_url}/api/ps", timeout=5)
                            if ps_resp.status_code == 200:
                                loaded_models = ps_resp.json().get("models", [])
                                if loaded_models:
                                    console.print("[dim]Unloading models...[/]")
                                    for m in loaded_models:
                                        m_name = m.get("name", "")
                                        requests.post(
                                            f"{ollama_url}/api/generate",
                                            json={"model": m_name, "prompt": "", "keep_alive": "0"},
                                            timeout=10
                                        )
                                    console.print(f"[dim]Unloaded {len(loaded_models)} models[/]")
                        except Exception as exc:
                            logger.debug(f"Ollama model unload failed: {exc}")

                        console.print("[bold yellow]Goodbye![/]")
                        break
                    continue

                # Check safety mode and handle overrides
                use_gemini_router = True
                effective_input = user_input
                was_override = False
                config = None

                try:
                    from aither_adk.ai.safety_mode import (
                        get_level_emoji,
                        get_level_name,
                        get_safety_manager,
                    )
                    safety = get_safety_manager()
                    config, effective_input, was_override = safety.get_effective_config(user_input)

                    if was_override:
                        safe_print("[bold #ffb3d9][UNLOCK] Override active - unrestricted mode for this turn[/]")
                except ImportError:
                    pass
                except Exception as e:
                    safe_print(f"[dim]Safety mode error: {e}[/]")

                # ========================================================================
                # FAST PATH: Direct image generation for high-confidence image requests
                # Bypasses LLM routing entirely for ~15 second generation
                # ========================================================================
                try:
                    from aither_adk.infrastructure.local_router import get_local_router
                    local_router = get_local_router()
                    decision = local_router.route(effective_input)

                    # FAST PATH: High confidence ArtistAgent routing = direct image generation
                    if decision and decision.agent == "ArtistAgent" and decision.confidence >= 0.8:
                        import time as time_mod
                        fast_start = time_mod.time()
                        safe_print(f"[dim][ZAP] Fast path: {decision.reason}[/]")

                        try:
                            # Import fast generation function
                            from aither_adk.infrastructure.fast_image import fast_generate_image

                            # Call fast path directly - no LLM, no agent transfer
                            result = await fast_generate_image(effective_input, config)

                            elapsed = time_mod.time() - fast_start
                            if result.get("success"):
                                paths = result.get("paths", [])
                                safe_print(f"[green][DONE] Image generated in {elapsed:.1f}s[/]")
                                for path in paths:
                                    safe_print(f"[bold cyan][PHOTO] {path}[/]")
                            else:
                                safe_print(f"[yellow][WARN] {result.get('error', 'Unknown error')}[/]")

                            use_gemini_router = False
                            continue  # Skip to next prompt

                        except ImportError:
                            # Fast path not available, fall through to normal routing
                            pass
                        except Exception as e:
                            safe_print(f"[dim]Fast path failed ({e}), using normal routing[/]")
                except ImportError:
                    pass
                except Exception as exc:
                    logger.debug(f"Safety mode check failed: {exc}")

                # Normal routing path (slower, goes through LLM)
                try:
                    from aither_adk.infrastructure.local_router import get_local_router
                    local_router = get_local_router()
                    decision = local_router.route(effective_input)

                    # Ensure decision is valid
                    if decision is None:
                        raise ValueError("Router returned None - check local_router.route()")

                    # Decide based on safety mode + routing confidence
                    should_local_route = decision.confidence >= 0.6

                    # In unrestricted mode or with override, always use local routing for NSFW
                    if was_override or (config and hasattr(config, 'allow_explicit') and config.allow_explicit):
                        if decision.agent == "ArtistAgent" or local_router.scene_nsfw_level(effective_input) >= 2:
                            should_local_route = True

                    if should_local_route:
                        # Quiet routing

                        # Handle MultiAgent routing (multiple @mentions for chat)
                        if decision.agent == "MultiAgent":
                            try:
                                from aither_adk.communication.multi_agent_chat import (
                                    dispatch_multi_agent,
                                    get_mentioned_agents,
                                )

                                agents = get_mentioned_agents(user_input)
                                # Quiet dispatch

                                # Find mailbox path
                                mailbox_path = os.path.join(
                                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "Saga", "mailbox.json"
                                )

                                dispatch_multi_agent(user_input, mailbox_path)
                                safe_print(f"[green][DONE] {len(agents)} agents responded! Check /inbox[/]")
                                use_gemini_router = False
                            except Exception as e:
                                safe_print(f"[yellow]Multi-agent dispatch failed: {e}[/]")
                        else:
                            # Check if the agent has transfer_to_agent tool and call it directly
                            # This bypasses Gemini Router and avoids safety blocks
                            if hasattr(agent, 'tools'):
                                for tool in agent.tools:
                                    tool_name = getattr(tool, '__name__', str(tool))
                                    if 'transfer_to_agent' in tool_name.lower():
                                        # Call the transfer function directly
                                        result = tool(decision.agent, user_input)
                                        safe_print(f"[dim]{result}[/]")
                                        use_gemini_router = False
                                        break
                except ImportError:
                    pass  # Local router not available, use normal flow
                except Exception:
                    # Quiet fallback - continue to Gemini router
                    pass

                if use_gemini_router:
                    await process_turn_with_fallback(user_input)
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                if debug_mode:
                    import traceback
                    traceback.print_exc()
    except Exception as e:
        console.print(f"[red]Fatal Error: {e}[/]")
        if debug_mode:
            import traceback
            traceback.print_exc()
    finally:
        # Clean up HTTP client sessions
        await cleanup_clients()
