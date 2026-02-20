
# =========================
# Terminal progress helpers
# =========================
import sys
import time
import threading
def show_spinner(msg: str, done_event: threading.Event, fps: float = 5.0) -> None:
    frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    delay = max(0.01, 1.0 / fps)
    try:
        idx = 0
        while not done_event.is_set():
            sys.stdout.write("\r" + (" " * 120) + "\r")
            sys.stdout.write(f"{frames[idx % len(frames)]} {msg}")
            sys.stdout.flush()
            idx += 1
            time.sleep(delay)
    finally:
        sys.stdout.write("\r" + (" " * 120) + "\r")
        sys.stdout.flush()


def show_progress_bar(current: int, total: int, width: int = 40) -> None:
    total = max(0, int(total))
    current = max(0, int(current))
    pct = 0.0 if total == 0 else round(current / total * 100, 2)
    filled = 0 if total == 0 else min(width, int(width * (current / total)))
    bar = "▇" * filled + "-" * (width - filled)
    line = f"[{bar}] {current}/{total} ({pct}%)"
    sys.stdout.write("\r" + (" " * 120) + "\r")
    sys.stdout.write(line)
    sys.stdout.flush()
