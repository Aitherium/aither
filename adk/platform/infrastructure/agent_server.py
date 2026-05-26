import asyncio
import sys
import aiohttp
from aiohttp import web
from aither_adk.ui.console import safe_print
from aither_adk.infrastructure.runner import process_turn
from google.adk.apps import App
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService

async def run_persistent_server(create_agent_fn, port, initial_model, debug_mode=False, mailbox=None):
    """
    Runs the agent as a persistent HTTP server.
    """
    safe_print(f"[bold green]Starting Persistent Agent Server on port {port}...[/]")
    
    # Initialize Agent
    agent = create_agent_fn(initial_model)
    
    # Initialize Services
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    credential_service = InMemoryCredentialService()
    app = App(name="agent_service", root_agent=agent)
    runner = Runner(app=app, session_service=session_service, artifact_service=artifact_service, credential_service=credential_service)
    
    # Create a single session for persistence
    user_id = "user"
    session = await runner.session_service.create_session(app_name=app.name, user_id=user_id)
    session_id = session.id
    
    # Session stats container
    server_session_stats = {"total_cost": 0.0, "total_input": 0, "total_output": 0}
    
    async def handle_prompt(request):
        try:
            data = await request.json()
            prompt = data.get("prompt")
            requested_model = data.get("model")
            
            if not prompt:
                return web.json_response({"error": "No prompt provided"}, status=400)
            
            # Update model if requested and different
            if requested_model and requested_model != agent.model:
                safe_print(f"[dim]Switching model to: {requested_model}[/]")
                agent.model = requested_model
            
            safe_print(f"[dim]Received prompt: {prompt} (Model: {agent.model})[/]")
            
            await process_turn(
                runner, 
                user_id, 
                session_id, 
                prompt, 
                agent.model, 
                server_session_stats, 
                root_agent=agent, 
                debug_mode=debug_mode, 
                mailbox=mailbox
            )
            
            final_response = server_session_stats.get("last_response", "No response generated.")
            return web.json_response({"response": final_response})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    server_app = web.Application()
    server_app.add_routes([web.post('/prompt', handle_prompt)])
    
    runner_site = web.AppRunner(server_app)
    await runner_site.setup()
    site = web.TCPSite(runner_site, 'localhost', port)
    await site.start()
    
    # Keep alive
    while True:
        await asyncio.sleep(3600)

async def try_connect_to_server(port, prompt, model=None):
    """
    Tries to connect to a running agent server.
    Returns True if successful (and prints response), False otherwise.
    """
    try:
        payload = {"prompt": prompt}
        if model:
            payload["model"] = model

        async with aiohttp.ClientSession() as client:
            async with client.post(f"http://localhost:{port}/prompt", json=payload, timeout=300) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(result.get("response", ""))
                    return True
                else:
                    return False
    except Exception:
        return False
