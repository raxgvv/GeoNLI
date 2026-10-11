"""Fair serialization and exclusive placement for the shared GPU."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Optional


class FairReentrantLock:
    """A FIFO reentrant lock so a stream of VLM calls cannot starve SAM3."""

    def __init__(self):
        self._condition = threading.Condition()
        self._owner: Optional[int] = None
        self._depth = 0
        self._next_ticket = 0
        self._waiters = deque()

    def acquire(self):
        ident = threading.get_ident()
        with self._condition:
            if self._owner == ident:
                self._depth += 1
                return True
            ticket = self._next_ticket
            self._next_ticket += 1
            self._waiters.append(ticket)
            while self._owner is not None or self._waiters[0] != ticket:
                self._condition.wait()
            self._waiters.popleft()
            self._owner = ident
            self._depth = 1
            return True

    def release(self):
        ident = threading.get_ident()
        with self._condition:
            if self._owner != ident:
                raise RuntimeError("GPU inference lock released by a non-owner")
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False


GPU_INFERENCE_LOCK = FairReentrantLock()


class GPUModelScheduler:
    """Keep VLM and SAM3 resident on CPU/GPU exclusively, swapping on task change.

    Callers must hold ``GPU_INFERENCE_LOCK`` while activating a model.
    Models are loaded once and transferred only when switching task families.
    """

    def __init__(self):
        self._vlm: Any = None
        self._sam3: Any = None
        self._active: Optional[str] = None

    def register_vlm(self, backend: Any) -> None:
        self._vlm = backend

    def register_sam3(self, service: Any) -> None:
        self._sam3 = service

    def activate_vlm(self) -> float:
        import torch
        started = time.monotonic()
        if torch.cuda.is_available():
            if self._sam3 is not None and self._sam3.device != "cpu":
                self._sam3.move_to("cpu")
            if self._vlm is not None:
                self._vlm.move_to("cuda")
        self._active = "vlm"
        return time.monotonic() - started

    def prepare_sam3(self) -> float:
        """Free the GPU before SAM3's builder allocates or loads anything."""
        started = time.monotonic()
        if self._vlm is not None and self._vlm.device != "cpu":
            self._vlm.move_to("cpu")
        self._active = "sam3_loading"
        return time.monotonic() - started

    def activate_sam3(self, service: Any) -> float:
        import torch
        started = time.monotonic()
        if torch.cuda.is_available():
            if self._vlm is not None:
                self._vlm.move_to("cpu")
            service.move_to("cuda")
        else:
            service.move_to("cpu")
        self._active = "sam3"
        return time.monotonic() - started

    @property
    def active_model(self) -> Optional[str]:
        return self._active


GPU_MODEL_SCHEDULER = GPUModelScheduler()
