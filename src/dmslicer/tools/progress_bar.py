import sys
import time

class RecursionSpinner:
    def __init__(self, interval=0.1, max_dots=10):
        self.last_time = 0.0
        self.dot_count = 0
        self.interval = interval      # 最小刷新时间（秒）
        self.max_dots = max_dots

    def tick(self):
        now = time.time()
        if now - self.last_time < self.interval:
            return

        self.last_time = now
        self.dot_count = (self.dot_count + 1) % (self.max_dots + 1)

        dots = "." * self.dot_count
        sys.stdout.write(f"\r[BVH query] running{dots:<{self.max_dots}}")
        sys.stdout.flush()

    def done(self):
        sys.stdout.write("\r[BVH query] done" + " " * self.max_dots + "\r")
        sys.stdout.flush()
