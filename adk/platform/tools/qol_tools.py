"""
Quality of Life Tools: Notebook, Calendar, Timers, Stopwatch, Alarms
"""

import datetime
import json
import os
import threading
import uuid
from typing import Dict, List, Optional

# Storage paths (writable, avoids read-only FS in Docker)
try:
    from aither_adk.paths import get_saga_subdir
    QOL_DIR = get_saga_subdir("data", "qol", create=True)
except ImportError:
    QOL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Saga", "data", "qol")
    try:
        os.makedirs(QOL_DIR, exist_ok=True)
    except OSError:
        pass

NOTEBOOK_FILE = os.path.join(QOL_DIR, "notebook.json")
CALENDAR_FILE = os.path.join(QOL_DIR, "calendar.json")
TIMERS_FILE = os.path.join(QOL_DIR, "timers.json")
STOPWATCH_FILE = os.path.join(QOL_DIR, "stopwatch.json")
ALARMS_FILE = os.path.join(QOL_DIR, "alarms.json")


# ============================================================
# NOTEBOOK
# ============================================================

class Notebook:
    def __init__(self):
        self.notes: List[Dict] = []
        self.load()

    def load(self):
        if os.path.exists(NOTEBOOK_FILE):
            try:
                with open(NOTEBOOK_FILE, 'r') as f:
                    self.notes = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.notes = []

    def save(self):
        try:
            with open(NOTEBOOK_FILE, 'w') as f:
                json.dump(self.notes, f, indent=2)
        except Exception as e:
            print(f"Error saving notebook: {e}")

    def add_note(self, content: str, tags: List[str] = None) -> Dict:
        note = {
            "id": str(uuid.uuid4()),
            "content": content,
            "tags": tags or [],
            "timestamp": datetime.datetime.now().isoformat(),
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.notes.append(note)
        self.save()
        return note

    def get_notes(self, tag: str = None, limit: int = None) -> List[Dict]:
        filtered = self.notes
        if tag:
            filtered = [n for n in filtered if tag.lower() in [t.lower() for t in n.get('tags', [])]]
        # Sort by timestamp descending (newest first)
        filtered = sorted(filtered, key=lambda x: x.get('timestamp', ''), reverse=True)
        if limit:
            filtered = filtered[:limit]
        return filtered

    def search_notes(self, query: str) -> List[Dict]:
        query_lower = query.lower()
        return [
            n for n in self.notes
            if query_lower in n.get('content', '').lower()
            or any(query_lower in t.lower() for t in n.get('tags', []))
        ]

    def delete_note(self, note_id: str) -> bool:
        original_len = len(self.notes)
        self.notes = [n for n in self.notes if n.get('id') != note_id]
        if len(self.notes) < original_len:
            self.save()
            return True
        return False


# ============================================================
# CALENDAR
# ============================================================

class Calendar:
    def __init__(self):
        self.events: List[Dict] = []
        self.load()

    def load(self):
        if os.path.exists(CALENDAR_FILE):
            try:
                with open(CALENDAR_FILE, 'r') as f:
                    self.events = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.events = []

    def save(self):
        try:
            with open(CALENDAR_FILE, 'w') as f:
                json.dump(self.events, f, indent=2)
        except Exception as e:
            print(f"Error saving calendar: {e}")

    def add_event(self, title: str, date: str, time: str = None, description: str = None, reminder: int = None) -> Dict:
        """Add event. Date format: YYYY-MM-DD, time format: HH:MM"""
        event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "date": date,
            "time": time or "00:00",
            "description": description or "",
            "reminder_minutes": reminder,
            "created": datetime.datetime.now().isoformat()
        }
        self.events.append(event)
        self.save()

        # PUSH ACTIVITY: Agent created a calendar event
        try:
            from lib.core.FluxEmitter import inject_agent_activity
            # Try to get agent_id from context or use "aither" as default
            agent_id = "aither"  # Default - could be enhanced to detect calling agent
            inject_agent_activity(agent_id.lower(), {
                "state": "scheduling",
                "task": f"Created calendar event: {title[:50]}...",
                "will": "default",
                "calendar": {"event_id": event.get("id"), "date": date, "time": time, "title": title[:50]},
            })
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to push calendar activity: {e}")

        return event

    def get_events(self, date: str = None, upcoming: bool = False) -> List[Dict]:
        """Get events. If date provided, filter by date. If upcoming=True, show future events."""
        filtered = self.events
        if date:
            filtered = [e for e in filtered if e.get('date') == date]
        elif upcoming:
            today = datetime.datetime.now().date().isoformat()
            filtered = [e for e in filtered if e.get('date') >= today]
            filtered = sorted(filtered, key=lambda x: (x.get('date', ''), x.get('time', '')))
        else:
            filtered = sorted(filtered, key=lambda x: (x.get('date', ''), x.get('time', '')))
        return filtered

    def delete_event(self, event_id: str) -> bool:
        original_len = len(self.events)
        self.events = [e for e in self.events if e.get('id') != event_id]
        if len(self.events) < original_len:
            self.save()
            return True
        return False


# ============================================================
# TIMERS
# ============================================================

class TimerManager:
    def __init__(self):
        self.timers: Dict[str, Dict] = {}
        self.active_timers: Dict[str, threading.Timer] = {}
        self.load()

    def load(self):
        if os.path.exists(TIMERS_FILE):
            try:
                with open(TIMERS_FILE, 'r') as f:
                    self.timers = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.timers = {}

    def save(self):
        try:
            with open(TIMERS_FILE, 'w') as f:
                json.dump(self.timers, f, indent=2)
        except Exception as e:
            print(f"Error saving timers: {e}")

    def create_timer(self, name: str, seconds: int, callback_msg: str = None) -> Dict:
        """Create a countdown timer."""
        timer_id = str(uuid.uuid4())
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)

        timer_data = {
            "id": timer_id,
            "name": name,
            "duration_seconds": seconds,
            "end_time": end_time.isoformat(),
            "callback_msg": callback_msg or f"Timer '{name}' finished!",
            "created": datetime.datetime.now().isoformat(),
            "status": "active"
        }

        self.timers[timer_id] = timer_data

        # Create actual timer thread
        def timer_callback():
            timer_data["status"] = "finished"
            self.save()
            if timer_id in self.active_timers:
                del self.active_timers[timer_id]

        timer_thread = threading.Timer(seconds, timer_callback)
        timer_thread.daemon = True
        timer_thread.start()
        self.active_timers[timer_id] = timer_thread

        self.save()
        return timer_data

    def get_timers(self, active_only: bool = False) -> List[Dict]:
        timers = list(self.timers.values())
        if active_only:
            timers = [t for t in timers if t.get('status') == 'active']
            # Update remaining time
            for timer in timers:
                end_time = datetime.datetime.fromisoformat(timer['end_time'])
                remaining = (end_time - datetime.datetime.now()).total_seconds()
                timer['remaining_seconds'] = max(0, int(remaining))
        return sorted(timers, key=lambda x: x.get('created', ''), reverse=True)

    def cancel_timer(self, timer_id: str) -> bool:
        if timer_id in self.active_timers:
            self.active_timers[timer_id].cancel()
            del self.active_timers[timer_id]
            if timer_id in self.timers:
                self.timers[timer_id]['status'] = 'cancelled'
                self.save()
            return True
        return False


# ============================================================
# STOPWATCH
# ============================================================

class StopwatchManager:
    def __init__(self):
        self.stopwatches: Dict[str, Dict] = {}
        self.load()

    def load(self):
        if os.path.exists(STOPWATCH_FILE):
            try:
                with open(STOPWATCH_FILE, 'r') as f:
                    self.stopwatches = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.stopwatches = {}

    def save(self):
        try:
            with open(STOPWATCH_FILE, 'w') as f:
                json.dump(self.stopwatches, f, indent=2)
        except Exception as e:
            print(f"Error saving stopwatch: {e}")

    def start_stopwatch(self, name: str = "default") -> Dict:
        """Start or resume a stopwatch."""
        if name not in self.stopwatches:
            self.stopwatches[name] = {
                "name": name,
                "start_time": datetime.datetime.now().isoformat(),
                "paused_time": None,
                "total_elapsed_seconds": 0,
                "status": "running"
            }
        else:
            sw = self.stopwatches[name]
            if sw.get('status') == 'paused':
                # Resume from paused
                sw['start_time'] = datetime.datetime.now().isoformat()
                sw['status'] = 'running'
            elif sw.get('status') == 'stopped':
                # Restart
                sw['start_time'] = datetime.datetime.now().isoformat()
                sw['total_elapsed_seconds'] = 0
                sw['status'] = 'running'

        self.save()
        return self.stopwatches[name]

    def pause_stopwatch(self, name: str = "default") -> Optional[Dict]:
        if name not in self.stopwatches:
            return None

        sw = self.stopwatches[name]
        if sw.get('status') == 'running':
            elapsed = (datetime.datetime.now() - datetime.datetime.fromisoformat(sw['start_time'])).total_seconds()
            sw['total_elapsed_seconds'] += elapsed
            sw['paused_time'] = datetime.datetime.now().isoformat()
            sw['status'] = 'paused'
            self.save()

        return sw

    def stop_stopwatch(self, name: str = "default") -> Optional[Dict]:
        if name not in self.stopwatches:
            return None

        sw = self.stopwatches[name]
        if sw.get('status') == 'running':
            elapsed = (datetime.datetime.now() - datetime.datetime.fromisoformat(sw['start_time'])).total_seconds()
            sw['total_elapsed_seconds'] += elapsed
            sw['status'] = 'stopped'
        elif sw.get('status') == 'paused':
            sw['status'] = 'stopped'

        self.save()
        return sw

    def get_elapsed(self, name: str = "default") -> float:
        """Get current elapsed time in seconds."""
        if name not in self.stopwatches:
            return 0.0

        sw = self.stopwatches[name]
        base_elapsed = sw.get('total_elapsed_seconds', 0)

        if sw.get('status') == 'running':
            current_elapsed = (datetime.datetime.now() - datetime.datetime.fromisoformat(sw['start_time'])).total_seconds()
            return base_elapsed + current_elapsed
        else:
            return base_elapsed

    def reset_stopwatch(self, name: str = "default") -> Optional[Dict]:
        if name in self.stopwatches:
            self.stopwatches[name] = {
                "name": name,
                "start_time": None,
                "paused_time": None,
                "total_elapsed_seconds": 0,
                "status": "stopped"
            }
            self.save()
            return self.stopwatches[name]
        return None


# ============================================================
# ALARMS
# ============================================================

class AlarmManager:
    def __init__(self):
        self.alarms: List[Dict] = []
        self.load()

    def load(self):
        if os.path.exists(ALARMS_FILE):
            try:
                with open(ALARMS_FILE, 'r') as f:
                    self.alarms = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.alarms = []

    def save(self):
        try:
            with open(ALARMS_FILE, 'w') as f:
                json.dump(self.alarms, f, indent=2)
        except Exception as e:
            print(f"Error saving alarms: {e}")

    def add_alarm(self, time: str, message: str = None, repeat: str = None) -> Dict:
        """
        Add alarm. Time format: HH:MM
        Repeat: 'daily', 'weekly', 'weekdays', or None for one-time
        """
        alarm = {
            "id": str(uuid.uuid4()),
            "time": time,
            "message": message or "Alarm!",
            "repeat": repeat,  # None, 'daily', 'weekly', 'weekdays'
            "enabled": True,
            "created": datetime.datetime.now().isoformat()
        }
        self.alarms.append(alarm)
        self.save()
        return alarm

    def get_alarms(self, enabled_only: bool = False) -> List[Dict]:
        alarms = self.alarms
        if enabled_only:
            alarms = [a for a in alarms if a.get('enabled', True)]
        return sorted(alarms, key=lambda x: x.get('time', ''))

    def toggle_alarm(self, alarm_id: str) -> bool:
        for alarm in self.alarms:
            if alarm.get('id') == alarm_id:
                alarm['enabled'] = not alarm.get('enabled', True)
                self.save()
                return True
        return False

    def delete_alarm(self, alarm_id: str) -> bool:
        original_len = len(self.alarms)
        self.alarms = [a for a in self.alarms if a.get('id') != alarm_id]
        if len(self.alarms) < original_len:
            self.save()
            return True
        return False


# ============================================================
# SINGLETONS
# ============================================================

_notebook: Optional[Notebook] = None
_calendar: Optional[Calendar] = None
_timer_manager: Optional[TimerManager] = None
_stopwatch_manager: Optional[StopwatchManager] = None
_alarm_manager: Optional[AlarmManager] = None

def get_notebook() -> Notebook:
    global _notebook
    if _notebook is None:
        _notebook = Notebook()
    return _notebook

def get_calendar() -> Calendar:
    global _calendar
    if _calendar is None:
        _calendar = Calendar()
    return _calendar

def get_timer_manager() -> TimerManager:
    global _timer_manager
    if _timer_manager is None:
        _timer_manager = TimerManager()
    return _timer_manager

def get_stopwatch_manager() -> StopwatchManager:
    global _stopwatch_manager
    if _stopwatch_manager is None:
        _stopwatch_manager = StopwatchManager()
    return _stopwatch_manager

def get_alarm_manager() -> AlarmManager:
    global _alarm_manager
    if _alarm_manager is None:
        _alarm_manager = AlarmManager()
    return _alarm_manager

