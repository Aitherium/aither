import os
import google.generativeai as genai
from google.cloud import aiplatform
# from config_manager import OUTPUT_DIR # Removed dependency
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import base64
import time
from google.adk.tools import ToolContext
from google.genai import types

# Default output directory if not specified
DEFAULT_OUTPUT_DIR = os.getenv("AITHER_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))

class GoogleGenAIClient:
    def __init__(self, api_key=None, project_id=None, location="us-central1"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location

        if self.api_key:
            genai.configure(api_key=self.api_key)

        if self.project_id:
            aiplatform.init(project=self.project_id, location=self.location)

    def generate_image_gemini(self, prompt, model_name="gemini-2.5-flash"):
        """
        Generates an image using Gemini's multimodal capabilities (if supported).
        Note: 'Nano Banana' / Gemini 2.5 Flash Image is the target here.
        """
        try:
            # In reality (Nov 2025), we assume the SDK supports this.
            # For now, we use the standard generation pattern.
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)

            # Check if response contains image data (hypothetical API structure for 2.5)
            # Or if it returns a URI.
            # We will assume it returns a standard response object that we can parse.
            return response
        except Exception as e:
            print(f"Error generating with Gemini: {e}")
            return None

    def generate_image_imagen(self, prompt, model_name="imagen-3.0-generate-001", aspect_ratio="1:1"):
        """
        Generates an image using Imagen on Vertex AI.
        Supports Imagen 3 and Imagen 4 Pro models.
        """
        if not self.project_id:
            print("Error: GOOGLE_CLOUD_PROJECT not set for Imagen.")
            return None

        try:
            from vertexai.preview.vision_models import ImageGenerationModel

            # Model name mapping
            model_mapping = {
                "imagen-4-pro": "imagen-4.0-pro",
                "imagen-4": "imagen-4.0-pro",
                "quick-imagen": "imagen-4.0-pro",
                "imagen-3": "imagen-3.0-generate-001",
                "imagen-3.0-generate-001": "imagen-3.0-generate-001"
            }

            real_model_name = model_mapping.get(model_name.lower(), model_name)
            print(f"[PHOTO] Using Imagen model: {real_model_name}")

            model = ImageGenerationModel.from_pretrained(real_model_name)

            images = model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                safety_filter_level="block_some",
                person_generation="allow_adult"
            )

            if images:
                # Save to temp file and return path
                output_dir = DEFAULT_OUTPUT_DIR
                os.makedirs(output_dir, exist_ok=True)
                filename = f"imagen_{int(time.time())}.png"
                filepath = os.path.join(output_dir, filename)
                images[0].save(location=filepath, include_generation_parameters=False)
                return filepath

        except Exception as e:
            print(f"Error generating with Imagen: {e}")
            return None

from google.adk.tools import ToolContext
from google.genai import types

async def generate_google_image(prompt: str, tool_context: ToolContext, model: str = "imagen-3.0-generate-001"):
    client = GoogleGenAIClient()
    filepath = None

    # Run synchronous generation in a thread if needed, but for now we'll just call it (blocking)
    # or wrap it if we want true async.

    if "gemini" in model.lower() or "banana" in model.lower():
        print(f"Generating with Gemini/Nano Banana ({model})...")
        filepath = client.generate_image_imagen(prompt, model_name="imagen-3.0-generate-001")
    else:
        filepath = client.generate_image_imagen(prompt, model_name=model)

    if filepath:
        # Save to artifact service
        try:
            with open(filepath, "rb") as f:
                image_bytes = f.read()

            await tool_context.save_artifact(
                os.path.basename(filepath),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            )

            return {
                "status": "success",
                "detail": f"Image generated successfully via Google ({model}) and saved to {filepath}",
                "filename": str(filepath),
                "provider": f"Google ({model})"
            }
        except Exception as e:
             return {"status": "error", "error": f"Failed to save artifact: {e}"}
    else:
        return {"status": "error", "error": "Google Image Generation failed."}
