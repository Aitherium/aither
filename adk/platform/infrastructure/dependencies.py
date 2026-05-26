import importlib
import sys
from typing import List, Dict, Tuple

class DependencyChecker:
    def __init__(self):
        self.optional_dependencies = {
            "duckduckgo_search": "Internet Search (duckduckgo-search)",
            "pyperclip": "Clipboard Access",
            "psutil": "System Monitoring",
            "PIL": "Image Processing (Pillow)",
            "cv2": "Computer Vision (opencv-python)",
            "httpx": "Async HTTP Client (for Council API)"
        }
        self.required_dependencies = {
            "rich": "Rich Text Interface",
            "prompt_toolkit": "Interactive CLI",
            "google.genai": "Google GenAI SDK",
            "yaml": "YAML Parsing (PyYAML)",
            "dotenv": "Environment Variables (python-dotenv)"
        }

    def check(self) -> Tuple[Dict[str, bool], List[str]]:
        """
        Checks for required and optional dependencies.
        Returns:
            status_map: Dict of package_name -> is_installed
            missing_required: List of missing required packages
        """
        status_map = {}
        missing_required = []

        # Check Required
        for package, name in self.required_dependencies.items():
            if not self._is_installed(package):
                status_map[package] = False
                missing_required.append(f"{name} ({package})")
            else:
                status_map[package] = True

        # Check Optional
        for package, name in self.optional_dependencies.items():
            status_map[package] = self._is_installed(package)

        return status_map, missing_required

    def _is_installed(self, package_name: str) -> bool:
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False

    def get_status_report(self) -> str:
        status_map, missing_required = self.check()
        report = []

        if missing_required:
            report.append("[bold red]CRITICAL: Missing Required Dependencies![/]")
            for missing in missing_required:
                report.append(f"  - {missing}")
            report.append("")

        report.append("[bold]Optional Capabilities:[/bold]")
        for package, name in self.optional_dependencies.items():
            installed = status_map.get(package, False)
            icon = "[DONE]" if installed else "[WARN] "
            style = "green" if installed else "yellow"
            msg = "Installed" if installed else "Missing (Functionality degraded)"
            report.append(f"  {icon} [{style}]{name}[/]: {msg}")

        return "\n".join(report)

def check_and_report_dependencies():
    checker = DependencyChecker()
    status_map, missing_required = checker.check()

    return {
        "status_map": status_map,
        "missing_required": missing_required,
        "report": checker.get_status_report(),
        "degraded": len(missing_required) > 0 or not all(status_map.get(p) for p in checker.optional_dependencies)
    }
