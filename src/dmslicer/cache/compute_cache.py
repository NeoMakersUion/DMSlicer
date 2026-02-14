from typing import Optional, Any
from collections import OrderedDict
import threading
import time
import pickle
from ..file_parser.workspace_utils import get_workspace_dir


class ComputeCache:
    def __init__(self, max_size: int = 10, ttl: int = 3600):
        self._max_size = max_size
        self._ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["time"] <= self._ttl:
                    self._cache.move_to_end(key)
                    return entry["value"]
                else:
                    del self._cache[key]
        try:
            ws_dir = get_workspace_dir()
            file_path = ws_dir / key / f"{key}_sorted_geom.pkl"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    value = pickle.load(f)
                self.set(key, value, save_to_disk=False)
                return value
        except Exception as e:
            print(f"[Cache] Error loading {key}: {e}")
            pass
        return None

    def set(self, key: str, value: Any, save_to_disk: bool = True) -> None:
        with self._lock:
            self._cache[key] = {"value": value, "time": time.time()}
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        if save_to_disk:
            try:
                ws_dir = get_workspace_dir()
                save_dir = ws_dir / key
                save_dir.mkdir(parents=True, exist_ok=True)
                file_path = save_dir / f"{key}_sorted_geom.pkl"
                with open(file_path, "wb") as f:
                    pickle.dump(value, f)
            except Exception as e:
                print(f"[Cache] Error saving {key}: {e}")

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
