import requests
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ModelSearchResult:
    name: str
    source: str  # "civitai" or "huggingface"
    url: str
    download_url: str
    tags: List[str]
    description: str
    filesize_mb: Optional[float] = None
    nsfw: bool = False

class ModelBrowser:
    """
    Browses and searches for models on Civitai and Hugging Face.
    """
    
    def __init__(self):
        self.civitai_api_url = "https://civitai.com/api/v1/models"
        self.hf_api_url = "https://huggingface.co/api/models"

    def search_civitai(self, query: str, limit: int = 5, nsfw: bool = False) -> List[ModelSearchResult]:
        """Searches Civitai for models."""
        params = {
            "query": query,
            "limit": limit,
            "types": "Checkpoint",
            "sort": "Highest Rated"
        }
        if not nsfw:
            params["nsfw"] = "false"

        try:
            response = requests.get(self.civitai_api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                # Get the latest version's download URL
                model_versions = item.get("modelVersions", [])
                if not model_versions:
                    continue
                
                latest_version = model_versions[0]
                files = latest_version.get("files", [])
                if not files:
                    continue
                
                # Prefer SafeTensor format
                download_url = files[0].get("downloadUrl")
                size_kb = files[0].get("sizeKB", 0)
                
                for f in files:
                    if "safetensors" in f.get("name", "").lower():
                        download_url = f.get("downloadUrl")
                        size_kb = f.get("sizeKB", 0)
                        break

                results.append(ModelSearchResult(
                    name=item.get("name"),
                    source="civitai",
                    url=f"https://civitai.com/models/{item.get('id')}",
                    download_url=download_url,
                    tags=item.get("tags", []),
                    description=item.get("description", "")[:200] + "...",
                    filesize_mb=size_kb / 1024 if size_kb else None,
                    nsfw=item.get("nsfw", False)
                ))
            return results
        except Exception as e:
            print(f"Error searching Civitai: {e}")
            return []

    def search_huggingface(self, query: str, limit: int = 5) -> List[ModelSearchResult]:
        """Searches Hugging Face for models."""
        params = {
            "search": query,
            "limit": limit,
            "filter": "text-to-image",
            "sort": "downloads"
        }
        
        try:
            response = requests.get(self.hf_api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data:
                model_id = item.get("modelId")
                results.append(ModelSearchResult(
                    name=model_id,
                    source="huggingface",
                    url=f"https://huggingface.co/{model_id}",
                    download_url=f"https://huggingface.co/{model_id}/resolve/main/model.safetensors", # Assumption, might need refinement
                    tags=item.get("tags", []),
                    description=f"Hugging Face model: {model_id}",
                    filesize_mb=None, # HF API doesn't give size in search easily
                    nsfw=False # HF tags might indicate, but assume false for now
                ))
            return results
        except Exception as e:
            print(f"Error searching Hugging Face: {e}")
            return []

    def download_model(self, url: str, save_path: str, progress_callback=None):
        """Downloads a model from a URL."""
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_length = r.headers.get('content-length')
                
                with open(save_path, 'wb') as f:
                    if total_length is None: # no content length header
                        f.write(r.content)
                    else:
                        dl = 0
                        total_length = int(total_length)
                        for data in r.iter_content(chunk_size=4096):
                            dl += len(data)
                            f.write(data)
                            if progress_callback:
                                progress_callback(dl, total_length)
            return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
