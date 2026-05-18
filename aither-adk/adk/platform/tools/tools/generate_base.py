import argparse
import os
import sys
from dotenv import load_dotenv

# Add tools directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add repo root to path to allow importing AitherNode
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from AitherOS.AitherNode.AitherCanvas import generate_local
from .google_genai_client import generate_google_image

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate base images using various sources.")
    parser.add_argument("prompt", type=str, help="The prompt for image generation")
    parser.add_argument("--source", type=str, default="local", choices=["local", "google", "imagen", "gemini"], help="Source: local (ComfyUI) or google (Imagen/Gemini)")
    parser.add_argument("--model", type=str, default=None, help="Specific model name (e.g., 'imagen 4 pro', 'nano banana pro')")

    args = parser.parse_args()

    print(f"Generating image with prompt: '{args.prompt}'")
    print(f"Source: {args.source}")

    output_paths = []

    if args.source == "local":
        print("Using Local ComfyUI...")
        output_paths = generate_local(args.prompt)
    elif args.source in ["google", "imagen", "gemini"]:
        model_name = args.model if args.model else "imagen-3.0-generate-001"
        if args.source == "gemini":
            model_name = "gemini-2.5-flash" # Default for gemini source

        print(f"Using Google Cloud ({model_name})...")
        path = generate_google_image(args.prompt, model=model_name)
        if path:
            output_paths = [path]

    if output_paths:
        print(f"Success! Images saved to:")
        for p in output_paths:
            print(f" - {p}")
    else:
        print("Generation failed.")

if __name__ == "__main__":
    main()
