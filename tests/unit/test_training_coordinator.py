"""Concurrency guard for the P0 in-process training coordinator."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from maintai.api.experiments import SingleTrainingCoordinator


class _FakeExperimentService:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run(self, experiment_id: str) -> dict:
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self._state_lock:
            self.active -= 1
        return {"id": experiment_id}


def test_training_coordinator_allows_only_one_active_run():
    coordinator = SingleTrainingCoordinator()
    service = _FakeExperimentService()
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda experiment_id: coordinator.run(service, experiment_id),
                ["e1", "e2", "e3"],
            )
        )
    assert [result["id"] for result in results] == ["e1", "e2", "e3"]
    assert service.max_active == 1
