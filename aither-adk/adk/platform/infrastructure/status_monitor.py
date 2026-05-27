import threading
import time
import logging

# Import the actual data fetching functions
from aither_adk.infrastructure.system_utils import get_system_load, get_git_status

# We need to handle the import of vision_tools carefully as it might not be available
try:
    from AitherOS.AitherNode.vision_tools import get_vision_backend_status
except ImportError:
    def get_vision_backend_status(): return "Unknown"

class StatusMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StatusMonitor, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        self.initialized = True
        self.system_load = "Init..."
        self.git_status = "Init..."
        self.vision_status = "Init..."
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                # Run checks
                # We can run them sequentially in this thread.
                # Since this thread is separate from main, blocking here is fine.

                # System Load
                load = get_system_load()

                # Git Status
                git = get_git_status()

                # Vision Status
                vision = get_vision_backend_status()

                with self.lock:
                    self.system_load = load
                    self.git_status = git
                    self.vision_status = vision

            except Exception as e:
                # logging.error(f"StatusMonitor error: {e}")
                pass

            time.sleep(2)

    def get_system_load(self):
        with self.lock:
            return self.system_load

    def get_git_status(self):
        with self.lock:
            return self.git_status

    def get_vision_status(self):
        with self.lock:
            return self.vision_status

monitor = StatusMonitor()
