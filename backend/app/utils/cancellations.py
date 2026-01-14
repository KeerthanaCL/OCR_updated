import asyncio
from typing import Dict

class CancellationManager:
    def __init__(self):
        self.accept_requests = True
        self.cancel_event = asyncio.Event()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()

    async def register(self, job_id: str, task: asyncio.Task):
        async with self.lock:
            self.running_tasks[job_id] = task

    async def unregister(self, job_id: str):
        async with self.lock:
            self.running_tasks.pop(job_id, None)

    async def cancel_all(self):
        async with self.lock:
            self.accept_requests = False
            self.cancel_event.set()
            for task in self.running_tasks.values():
                task.cancel()

    async def reset(self):
        async with self.lock:
            self.accept_requests = True
            self.cancel_event.clear()
