import time


class FixedRate:
    def __init__(self, hz):
        self.period = 1.0 / hz
        self.next_deadline = time.monotonic()

    def sleep(self):
        self.next_deadline += self.period
        delay = self.next_deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_deadline = time.monotonic()
