import subprocess
import os
from rich.console import Console

console = Console()

def run_command(command, cwd=None):
    """Runs a shell command and returns the output."""
    try:
        import shlex
        result = subprocess.run(
            shlex.split(command) if isinstance(command, str) else command,
            cwd=cwd,
            shell=False,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def run_tofu_init(directory: str):
    """Runs 'tofu init' in the specified directory."""
    console.print(f"[bold blue]Running tofu init in {directory}...[/bold blue]")
    return run_command("tofu init", cwd=directory)

def run_tofu_plan(directory: str):
    """Runs 'tofu plan' in the specified directory."""
    console.print(f"[bold blue]Running tofu plan in {directory}...[/bold blue]")
    return run_command("tofu plan", cwd=directory)

def run_tofu_apply(directory: str):
    """Runs 'tofu apply -auto-approve' in the specified directory."""
    console.print(f"[bold blue]Running tofu apply in {directory}...[/bold blue]")
    return run_command("tofu apply -auto-approve", cwd=directory)

def run_tofu_destroy(directory: str):
    """Runs 'tofu destroy -auto-approve' in the specified directory."""
    console.print(f"[bold red]Running tofu destroy in {directory}...[/bold red]")
    return run_command("tofu destroy -auto-approve", cwd=directory)

def read_tofu_file(file_path: str):
    """Reads the content of a .tf file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_tofu_file(file_path: str, content: str):
    """Writes content to a .tf file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"

def list_files(directory: str):
    """Lists files in a directory."""
    try:
        return os.listdir(directory)
    except Exception as e:
        return f"Error listing files: {e}"

infrastructure_tools = [
    run_tofu_init,
    run_tofu_plan,
    run_tofu_apply,
    run_tofu_destroy,
    read_tofu_file,
    write_tofu_file,
    list_files
]
