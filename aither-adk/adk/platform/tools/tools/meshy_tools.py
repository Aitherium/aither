import os
import requests
import time
import json
from google.adk.tools import ToolContext

MESHY_API_KEY = os.getenv("MESHY_API_KEY")
BASE_URL = "https://api.meshy.ai/openapi"

def _get_headers():
    if not MESHY_API_KEY:
        raise ValueError("MESHY_API_KEY not found in environment variables.")
    return {
        "Authorization": f"Bearer {MESHY_API_KEY}",
        "Content-Type": "application/json"
    }

def generate_3d_model_from_text(prompt: str, art_style: str = "realistic", tool_context: ToolContext = None):
    """
    Generates a 3D model from a text prompt using Meshy API (v2).
    This initiates a 'preview' task.

    Args:
        prompt (str): The text description of the 3D model.
        art_style (str): The style of the model. Options: "realistic", "sculpture". Default: "realistic".
        tool_context (ToolContext): Context for logging/saving.

    Returns:
        dict: The result containing model URLs (glb, fbx, etc.) or error.
    """
    print(f"Generating 3D model for prompt: '{prompt}' with style: '{art_style}'...")

    url = f"{BASE_URL}/v2/text-to-3d"
    payload = {
        "mode": "preview",
        "prompt": prompt,
        "art_style": art_style,
        "should_remesh": True,
        "ai_model": "latest" # Use Meshy 6 Preview if available
    }

    try:
        response = requests.post(url, headers=_get_headers(), json=payload)
        response.raise_for_status()
        data = response.json()

        task_id = data.get("result")
        if not task_id:
            return {"status": "error", "error": f"No task ID returned. Response: {data}"}

        print(f"Task started: {task_id}. Polling for completion...")
        return _poll_meshy_task(task_id, "text-to-3d", tool_context)

    except Exception as e:
        return {"status": "error", "error": str(e)}

def generate_3d_model_from_image(image_url: str, tool_context: ToolContext = None):
    """
    Generates a 3D model from an image URL using Meshy API (v1).

    Args:
        image_url (str): The URL of the source image.
        tool_context (ToolContext): Context for logging/saving.

    Returns:
        dict: The result containing model URLs.
    """
    print(f"Generating 3D model from image: {image_url}...")

    url = f"{BASE_URL}/v1/image-to-3d"
    payload = {
        "image_url": image_url,
        "enable_pbr": True,
        "should_remesh": True,
        "should_texture": True,
        "ai_model": "latest"
    }

    try:
        response = requests.post(url, headers=_get_headers(), json=payload)
        response.raise_for_status()
        data = response.json()

        task_id = data.get("result")
        if not task_id:
            return {"status": "error", "error": f"No task ID returned. Response: {data}"}

        print(f"Task started: {task_id}. Polling for completion...")
        return _poll_meshy_task(task_id, "image-to-3d", tool_context)

    except Exception as e:
        return {"status": "error", "error": str(e)}

def _poll_meshy_task(task_id: str, task_type: str, tool_context: ToolContext = None):
    """
    Polls the Meshy API for task completion.
    """
    # Determine endpoint version based on task type
    version = "v2" if task_type == "text-to-3d" else "v1"
    url = f"{BASE_URL}/{version}/{task_type}/{task_id}"

    start_time = time.time()
    timeout = 600 # 10 minutes timeout (3D generation can be slow)

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, headers=_get_headers())
            response.raise_for_status()
            data = response.json()

            status = data.get("status")
            progress = data.get("progress", 0)

            if status == "SUCCEEDED":
                print(f"Task {task_id} succeeded!")
                return {
                    "status": "success",
                    "model_urls": data.get("model_urls"),
                    "thumbnail_url": data.get("thumbnail_url"),
                    "video_url": data.get("video_url"),
                    "metadata": data
                }
            elif status == "FAILED":
                return {"status": "error", "error": f"Task failed: {data.get('task_error', {}).get('message', 'Unknown error')}"}
            elif status == "EXPIRED":
                return {"status": "error", "error": "Task expired."}
            elif status == "CANCELED":
                return {"status": "error", "error": "Task canceled."}

            print(f"Task {task_id} in progress: {status} ({progress}%)")
            time.sleep(5) # Poll every 5 seconds

        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(5)

    return {"status": "error", "error": "Timeout waiting for Meshy task completion."}
