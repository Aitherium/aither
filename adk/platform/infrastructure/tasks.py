import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class Task:
    def __init__(self, title: str, status: str = "todo", created_at: str = None, completed_at: str = None, id: int = 0):
        self.id = id
        self.title = title
        self.status = status # todo, in-progress, done
        self.created_at = created_at or datetime.now().isoformat()
        self.completed_at = completed_at

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class TaskManager:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tasks: List[Task] = []
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.tasks = [Task.from_dict(t) for t in data]
            except Exception as e:
                print(f"Error loading tasks: {e}")
                self.tasks = []

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump([t.to_dict() for t in self.tasks], f, indent=2)
        except Exception as e:
            print(f"Error saving tasks: {e}")

    def add_task(self, title: str) -> Task:
        new_id = 1
        if self.tasks:
            new_id = max(t.id for t in self.tasks) + 1

        task = Task(title=title, id=new_id)
        self.tasks.append(task)
        self.save()
        return task

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        if status:
            return [t for t in self.tasks if t.status == status]
        return self.tasks

    def update_status(self, task_id: int, status: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                if status == "done":
                    task.completed_at = datetime.now().isoformat()
                self.save()
                return task
        return None

    def remove_task(self, task_id: int) -> bool:
        initial_len = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.id != task_id]
        if len(self.tasks) < initial_len:
            self.save()
            return True
        return False

    def clear_completed(self):
        self.tasks = [t for t in self.tasks if t.status != "done"]
        self.save()
