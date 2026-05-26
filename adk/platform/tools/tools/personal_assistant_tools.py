import os
import datetime
import json
import subprocess
import requests
from typing import List, Optional
from google.adk.tools import FunctionTool

# Try to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None

# Try to import ddgs (new package name) or duckduckgo_search (old package name)
try:
    try:
        from ddgs import DDGS
    except ImportError:
        from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

# Try to import pyperclip
try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

# Try to import psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_current_time() -> str:
    """
    Returns the current date and time.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_terminal_command(command: str) -> str:
    """
    Executes a terminal command and returns the output.
    Use this to run system commands, git, python scripts, etc.

    Args:
        command: The command to execute (e.g., 'ls -la', 'git status').
    """
    try:
        import shlex
        result = subprocess.run(
            shlex.split(command) if isinstance(command, str) else command,
            shell=False,
            capture_output=True,
            text=True,
            timeout=60
        )

        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")

        if not output:
            return "Command executed successfully with no output."

        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

def list_directory(path: str = ".") -> str:
    """
    Lists the contents of a directory.

    Args:
        path: The path to the directory (default is current directory).
    """
    try:
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist."

        items = os.listdir(path)
        # Add type info (dir/file)
        result = []
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                result.append(f"[DIR]  {item}")
            else:
                result.append(f"[FILE] {item}")

        return "\n".join(sorted(result))
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def read_file(path: str) -> str:
    """
    Reads the content of a text file.

    Args:
        path: The path to the file.
    """
    try:
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Truncate if too long?
            if len(content) > 10000:
                return content[:10000] + "\n...[Truncated]..."
            return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(path: str, content: str) -> str:
    """
    Writes content to a file. Overwrites if exists.

    Args:
        path: The path to the file.
        content: The content to write.
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to '{path}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"

def web_search(query: str, max_results: int = 5) -> str:
    """
    Performs a web search using DuckDuckGo.

    Args:
        query: The search query.
        max_results: Number of results to return (default 5).
    """
    if not HAS_DDGS:
        return "Error: duckduckgo-search library not installed."

    try:
        results = DDGS().text(query, max_results=max_results)
        formatted_results = []
        if results:
            for r in results:
                formatted_results.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n")
            return "\n---\n".join(formatted_results)
        else:
            return "No results found."
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def send_to_inbox(recipient: str, subject: str, content: str) -> str:
    """
    Sends a message to the user's inbox.
    Use this to store search results, long texts, or information for later review.

    Args:
        recipient: The recipient (usually 'user').
        subject: A brief subject line.
        content: The body of the message.
    """
    try:
        # We need to access the global mailbox instance.
        # Since tools are stateless functions, we might need to import it or rely on a singleton.
        # For now, we'll try to import the mailbox from the main agent module if possible,
        # or instantiate a new one pointing to the same file.
        from AitherOS.agents.common.mailbox import Mailbox
        # Assuming default path is used or we can find it.
        # Ideally, the mailbox path should be consistent.
        # Let's assume it's in the current working directory or a standard location.
        mailbox = Mailbox() # Defaults to mailbox.json in CWD
        mailbox.send_message("Agent", recipient, subject, content)
        return f"Message sent to {recipient}'s inbox."
    except Exception as e:
        return f"Error sending to inbox: {str(e)}"


def fetch_webpage_content(url: str) -> str:
    """
    Fetches and extracts text content from a webpage URL.
    Useful for reading articles, documentation, or specific pages found via search.

    Args:
        url: The URL to fetch.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        text = soup.get_text()

        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # Truncate if too long
        if len(text) > 20000:
            return text[:20000] + "\n...[Truncated]..."
        return text
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"

def get_clipboard_content() -> str:
    """
    Reads the current content of the system clipboard.
    Useful when the user says "I copied the error message" or "Read my clipboard".
    """
    if not HAS_CLIPBOARD:
        return "Error: Clipboard access not available (pyperclip not installed or headless environment)."
    try:
        return pyperclip.paste()
    except Exception as e:
        return f"Error reading clipboard: {str(e)}"

def set_clipboard_content(text: str) -> str:
    """
    Writes text to the system clipboard.
    Useful for copying code snippets, summaries, or generated text for the user.

    Args:
        text: The text to copy to clipboard.
    """
    if not HAS_CLIPBOARD:
        return "Error: Clipboard access not available."
    try:
        pyperclip.copy(text)
        return "Successfully copied to clipboard."
    except Exception as e:
        return f"Error writing to clipboard: {str(e)}"

def get_system_stats() -> str:
    """
    Returns current system statistics (CPU, Memory, Disk, Battery).
    Useful for checking system health.
    """
    if not HAS_PSUTIL:
        return "Error: psutil library not installed."

    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        stats = [
            f"CPU Usage: {cpu_percent}%",
            f"Memory: {memory.percent}% used ({memory.used // (1024*1024)}MB / {memory.total // (1024*1024)}MB)",
            f"Disk Usage: {disk.percent}% used ({disk.free // (1024*1024*1024)}GB free)",
        ]

        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery:
                stats.append(f"Battery: {battery.percent}% ({'Charging' if battery.power_plugged else 'Discharging'})")

        return "\n".join(stats)
    except Exception as e:
        return f"Error getting system stats: {str(e)}"

def get_weather(location: str) -> str:
    """
    Gets the current weather for a location using wttr.in.

    Args:
        location: City name or location (e.g. "London", "New York").
    """
    try:
        url = f"https://wttr.in/{location}?format=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
        else:
            return f"Error fetching weather: HTTP {response.status_code}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

def manage_todo_list(action: str, item: Optional[str] = None) -> str:
    """
    Manages a simple todo list stored in 'todo.json'.

    Args:
        action: 'add', 'list', 'remove', 'clear'
        item: The task description (required for 'add', optional for 'remove' if using index)
    """
    todo_file = "todo.json"

    try:
        todos = []
        if os.path.exists(todo_file):
            with open(todo_file, 'r') as f:
                todos = json.load(f)

        if action == "list":
            if not todos:
                return "Todo list is empty."
            return "\n".join([f"{i+1}. {task}" for i, task in enumerate(todos)])

        elif action == "add":
            if not item:
                return "Error: Item description required for 'add'."
            todos.append(item)
            with open(todo_file, 'w') as f:
                json.dump(todos, f)
            return f"Added task: {item}"

        elif action == "remove":
            if not item:
                return "Error: Item index or name required for 'remove'."

            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(todos):
                    removed = todos.pop(idx)
                    with open(todo_file, 'w') as f:
                        json.dump(todos, f)
                    return f"Removed task: {removed}"
                else:
                    return "Error: Invalid task index."
            else:
                if item in todos:
                    todos.remove(item)
                    with open(todo_file, 'w') as f:
                        json.dump(todos, f)
                    return f"Removed task: {item}"
                else:
                    return "Error: Task not found."

        elif action == "clear":
            if os.path.exists(todo_file):
                os.remove(todo_file)
            return "Todo list cleared."

        else:
            return "Error: Unknown action. Use 'add', 'list', 'remove', or 'clear'."

    except Exception as e:
        return f"Error managing todo list: {str(e)}"


def transfer_to_agent(agent_name: str) -> str:
    """
    Transfers the conversation to another agent.
    Use this when the user asks to switch personas or when a task requires a different specialist.

    Args:
        agent_name: The name of the agent to transfer to (e.g., 'Aither', 'CoderAgent', 'ArtistAgent').
    """
    return f"[SYSTEM_TRANSFER_REQUEST:{agent_name}]"


# Export all tools in one consolidated list
personal_assistant_tools = [
    FunctionTool(transfer_to_agent),
    FunctionTool(get_current_time),
    FunctionTool(run_terminal_command),
    FunctionTool(list_directory),
    FunctionTool(read_file),
    FunctionTool(write_file),
    FunctionTool(web_search),
    FunctionTool(fetch_webpage_content),
    FunctionTool(get_clipboard_content),
    FunctionTool(set_clipboard_content),
    FunctionTool(get_system_stats),
    FunctionTool(get_weather),
    FunctionTool(manage_todo_list),
    FunctionTool(send_to_inbox)
]
