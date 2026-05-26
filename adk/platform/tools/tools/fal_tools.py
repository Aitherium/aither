import os
import fal_client
import pathlib
import time
import base64
import requests
from google.adk.tools import ToolContext
from google.genai import types
# from config_manager import OUTPUT_DIR # Removed dependency

# Default output directory
DEFAULT_OUTPUT_DIR = os.getenv("AITHER_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))

async def generate_image_with_fal(prompt: str, tool_context: ToolContext, image_size: str = "landscape_4_3", style: str = "realistic", model: str = "fal-ai/bytedance/seedream/v4/text-to-image", loras: list = None, negative_prompt: str = None):
    """
    Generates a new image using Fal.ai with style control.

    Args:
        prompt (str): The prompt to generate the image.
        tool_context (ToolContext): Context for saving artifacts.
        image_size (str): The aspect ratio/size. Options: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9".
        style (str): The desired style. Options: "realistic" (default), "anime", "cinematic", "3d".
        model (str): The Fal.ai model endpoint to use. Defaults to "fal-ai/bytedance/seedream/v4/text-to-image".
                     Other options: "fal-ai/stable-diffusion-v35-medium", "fal-ai/flux/dev", "fal-ai/nano-banana-pro", "fal-ai/imagen4/preview/ultra".
        loras (list): Optional list of LoRAs to apply.
        negative_prompt (str): Optional negative prompt to append to the default.
    """

    # Style Presets
    style_prompts = {
        "realistic": ", hyperrealistic, 8k, highly detailed, raw photo, dslr, soft lighting, film grain, detailed background, wide angle, full body shot",
        "anime": ", anime style, anime art, vibrant colors, cel shading, high quality, masterpiece, detailed background",
        "cinematic": ", cinematic lighting, movie scene, dramatic atmosphere, 4k, detailed, depth of field",
        "3d": ", 3d render, unreal engine 5, octane render, highly detailed, ray tracing"
    }

    # Apply style suffix if not already present in prompt
    suffix = style_prompts.get(style.lower(), "")
    enhanced_prompt = prompt
    if suffix and suffix.strip().split(',')[1].strip() not in prompt:
        enhanced_prompt = f"{prompt}{suffix}"

    print(f"Submitting to {model} with prompt: {enhanced_prompt} (Style: {style})")

    # Default Negative Prompt (Universal)
    default_negative_prompt = "low quality, worst quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name, deformed, ugly, mutation, mutated, disfigured"

    if negative_prompt:
        final_negative_prompt = f"{default_negative_prompt}, {negative_prompt}"
    else:
        final_negative_prompt = default_negative_prompt

    arguments = {
        "prompt": enhanced_prompt,
        "negative_prompt": final_negative_prompt,
        "image_size": image_size,
        "num_inference_steps": 40,
        "guidance_scale": 3.5, # Flux prefers 3.5
        "enable_safety_checker": False
    }

    if loras:
        arguments["loras"] = loras

    try:
        handler = fal_client.submit(
            model,
            arguments=arguments,
        )

        result = handler.get()

        if not result or "images" not in result or not result["images"]:
             return {"status": "error", "error": "No images returned from Fal.ai."}

        # Result contains a URL to the generated image
        generated_image_url = result["images"][0]["url"]

        # Download the image
        import requests
        response = requests.get(generated_image_url)
        if response.status_code != 200:
             return {"status": "error", "error": "Failed to download generated image from Fal.ai."}

        image_bytes = response.content

        # Save to disk
        output_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        filename = f"fal_{int(time.time())}.png"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # Save to artifact service
        await tool_context.save_artifact(
            filename,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        )

        return {
            "status": "success",
            "detail": f"Image generated successfully via Fal.ai ({style}) and saved to {filepath}",
            "filename": str(filepath),
            "provider": f"Fal.ai (Flux/Dev - {style})"
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

# Alias for backward compatibility if needed, or just ensure the tool name matches what the agent expects
generate_image = generate_image_with_fal

async def refine_image_with_fal(image_path: str, prompt: str, tool_context: ToolContext, strength: float = 0.50, style: str = "realistic", model: str = "fal-ai/stable-diffusion-v35-medium"):
    """
    Refines an existing image using Fal.ai with style awareness.

    Args:
        image_path (str): Path to the source image file (must exist locally).
        prompt (str): The prompt to guide the refinement.
        tool_context (ToolContext): Context for saving artifacts.
        strength (float): 0.0 to 1.0. The strength of the transformation.
        style (str): The desired style. Options: "realistic" (default), "anime", "cinematic", "3d".
        model (str): The Fal.ai model endpoint to use. Defaults to "fal-ai/stable-diffusion-v35-medium".
    """

    if not os.path.exists(image_path):
        return {"status": "error", "error": f"Source image not found at {image_path}"}

    # Upload image to Fal
    try:
        print(f"Uploading {image_path} to Fal.ai...")
        image_url = fal_client.upload_file(image_path)
    except Exception as e:
        return {"status": "error", "error": f"Failed to upload image to Fal.ai: {e}"}

    print(f"Image uploaded: {image_url}")

    # Style Presets (Same as generate)
    style_prompts = {
        "realistic": ", detailed skin texture, hyperrealistic, 8k, raw photo, dslr, soft lighting, film grain, sharp focus, masterpiece, best quality",
        "anime": ", anime style, anime art, vibrant colors, cel shading, high quality, masterpiece",
        "cinematic": ", cinematic lighting, movie scene, dramatic atmosphere, 4k, detailed",
        "3d": ", 3d render, unreal engine 5, octane render, highly detailed"
    }

    # Apply style suffix
    suffix = style_prompts.get(style.lower(), "")
    enhanced_prompt = prompt

    # Only append if not already present and prompt is short
    if suffix and len(prompt) < 200:
        enhanced_prompt = f"{prompt}{suffix}"

    print(f"Submitting to {model} with prompt: {enhanced_prompt} (Style: {style})")

    try:
        handler = fal_client.submit(
            model,
            arguments={
                "prompt": enhanced_prompt,
                "image_url": image_url,
                "strength": strength,
                "num_inference_steps": 40,
                "guidance_scale": 3.5,
                "enable_safety_checker": False,
                "sync_mode": True
            },
        )

        result = handler.get()

        if not result or "images" not in result or not result["images"]:
             return {"status": "error", "error": "No images returned from Fal.ai."}

        # Result contains a URL to the generated image
        generated_image_url = result["images"][0]["url"]

        # Download the image
        import requests
        response = requests.get(generated_image_url)
        if response.status_code != 200:
             return {"status": "error", "error": "Failed to download generated image from Fal.ai."}

        image_bytes = response.content

        # Save to disk
        output_path = pathlib.Path(os.getcwd()) / OUTPUT_DIR
        output_path.mkdir(exist_ok=True)

        timestamp = int(time.time())
        filename = f"fal_refined_{style}_{timestamp}.png"
        file_path = output_path / filename

        with open(file_path, "wb") as f:
            f.write(image_bytes)

        # Save to artifact service
        await tool_context.save_artifact(
            filename,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        )

        return {
            "status": "success",
            "detail": f"Image refined successfully via Fal.ai ({style}) and saved to {file_path}",
            "filename": str(file_path),
            "provider": f"Fal.ai ({model})"
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

async def generate_video_with_fal(prompt: str, tool_context: ToolContext, image_url: str = None, seconds: int = 5, model: str = None):
    """
    Generates a video using Fal.ai.

    Args:
        prompt (str): The description of the video motion and scene.
        tool_context (ToolContext): Context for saving artifacts.
        image_url (str): Optional. If provided, creates an image-to-video animation (start frame).
        seconds (int): Duration in seconds (default 5).
        model (str, optional): Specific model to use. Defaults to Hunyuan (text) or Kling (image).
    """
    # Use Hunyuan Video for Text-to-Video (Open weights, less censored)
    # Use Kling for Image-to-Video (High quality)

    if model:
        print(f"Using specified model: {model}")
        arguments = {
            "prompt": prompt
        }
        if image_url:
            arguments["image_url"] = image_url
        # Add other common args if they fit, but model specific args might vary
        if "duration" not in arguments and model != "fal-ai/fast-svd/text-to-video": # fast-svd might not take duration
             arguments["duration"] = str(seconds) if isinstance(seconds, int) else seconds

    elif image_url:
        # Default to Kling for Image-to-Video if not specified
        if not model:
            model = "fal-ai/kling-video/v1/image-to-video"

        print(f"Submitting Image-to-Video to {model}...")
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": str(seconds),
            "aspect_ratio": "16:9"
        }
    else:
        # Default to Hunyuan for Text-to-Video
        if not model:
            model = "fal-ai/hunyuan-video"

        # Enhance video prompt
        enhanced_prompt = prompt
        if "cinematic" not in prompt.lower():
             enhanced_prompt = f"cinematic, 4k, high quality, {prompt}, slow motion, highly detailed"

        arguments = {
            "prompt": enhanced_prompt,
            "duration": seconds,
            "resolution": "720p",
            "aspect_ratio": "16:9"
        }
        print(f"Submitting Text-to-Video to {model} with prompt: {enhanced_prompt}")

    try:
        handler = fal_client.submit(
            model,
            arguments=arguments,
        )

        result = handler.get()

        if not result or "video" not in result:
             print(f"[Fal Error] No video returned. Result: {result}")
             return {"status": "error", "error": "No video returned from Fal.ai."}

        # Result contains a URL to the generated video
        video_url = result["video"]["url"]

        # Download the video
        import requests
        response = requests.get(video_url)
        if response.status_code != 200:
             return {"status": "error", "error": "Failed to download generated video."}

        video_bytes = response.content

        # Save to disk
        output_path = pathlib.Path(os.getcwd()) / OUTPUT_DIR
        output_path.mkdir(exist_ok=True)

        timestamp = int(time.time())
        filename = f"fal_video_{timestamp}.mp4"
        file_path = output_path / filename

        with open(file_path, "wb") as f:
            f.write(video_bytes)

        # Save to artifact service
        await tool_context.save_artifact(
            filename,
            types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
        )

        return {
            "status": "success",
            "detail": f"Video generated successfully and saved to {file_path}",
            "filename": str(file_path),
            "provider": f"Fal.ai ({model})"
        }

    except Exception as e:
        print(f"[Fal Error] Exception occurred: {str(e)}")
        return {"status": "error", "error": str(e)}
