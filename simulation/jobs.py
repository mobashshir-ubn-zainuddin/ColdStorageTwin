"""
Background execution of simulations for the web dashboard.

Runs are executed in daemon threads and kept in memory (the most recent few).
The web server must therefore run as a single process (e.g. `python app.py`
or `gunicorn -w 1 --threads 8 app:app`) so every request sees the same jobs.
"""

import threading
import time
import traceback
import uuid
from typing import Dict, Any, Optional

from simulation.scenario import Scenario
from simulation.results import ResultStore
from solver.coupled_solver import ColdRoomSolver

MAX_JOBS = 6


class Job:
    def __init__(self, scenario_dict: Dict[str, Any]):
        self.id = uuid.uuid4().hex[:12]
        self.scenario_dict = scenario_dict
        self.status = 'queued'
        self.progress = 0.0
        self.message = 'Queued'
        self.error: Optional[str] = None
        self.warnings = []
        self.created = time.time()
        self.store: Optional[ResultStore] = None
        self.solver: Optional[ColdRoomSolver] = None
        self._cancel = False

    def to_dict(self) -> Dict[str, Any]:
        return {'id': self.id, 'status': self.status, 'progress': self.progress, 'message': self.message,
                'error': self.error, 'warnings': self.warnings,
                'n_snapshots': len(self.store.snapshots) if self.store else 0}


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()

    def submit(self, scenario_dict: Dict[str, Any]) -> Job:
        # Validate synchronously so input errors are reported immediately
        scenario = Scenario.from_dict(scenario_dict)
        job = Job(scenario_dict)
        job.warnings = list(scenario.warnings)
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > MAX_JOBS:
                oldest = min(self.jobs.values(), key=lambda j: j.created)
                if oldest.status == 'running':
                    break
                self.jobs.pop(oldest.id)
        threading.Thread(target=self._run, args=(job, scenario), daemon=True).start()
        return job

    def _run(self, job: Job, scenario: Scenario) -> None:
        try:
            job.status = 'running'
            job.message = 'Initialising mesh and pressure operator'
            solver = ColdRoomSolver(scenario)
            job.solver = solver
            job.store = ResultStore(solver)

            def progress(frac, msg):
                job.progress = float(frac)
                job.message = msg

            solver.run(recorder=job.store, progress=progress, cancel=lambda: job._cancel)
            job.warnings = list(scenario.warnings)
            if solver.aborted and solver.aborted != 'cancelled':
                job.status = 'failed'
                job.error = solver.aborted
            else:
                job.status = 'cancelled' if solver.aborted == 'cancelled' else 'finished'
                job.progress = 1.0
                job.message = f'Finished in {job.store.wall_time:.1f} s ({solver.step_count} steps)'
        except Exception as exc:  # pragma: no cover - reported to the client
            job.status = 'failed'
            job.error = f'{type(exc).__name__}: {exc}'
            job.message = traceback.format_exc(limit=3)

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job:
            job._cancel = True
            return True
        return False


manager = JobManager()
