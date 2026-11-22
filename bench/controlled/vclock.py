"""A virtual-time asyncio event loop for controlled, host-independent benchmarks.

loop.time() returns simulated time. When the loop would block waiting for the
next timer, it advances the simulated clock to that timer instead of sleeping.
So `await asyncio.sleep(0.130)` inside a mock model "costs" exactly 130 ms of
simulated time, and every run of the same workload produces the same numbers
regardless of CPU steal, noisy neighbours or core count.

What this measures: latency as composed from *declared* costs (model
inference, judge, upstream LLM, network hops) by the *real* proxy code's
control flow -- what runs in sequence, what runs concurrently, what is
skipped, cached, raced or shed.
What it deliberately does not measure: the proxy's own CPU time. CPU work takes
zero simulated time; it is reported separately as deterministic operation
counts (bench/controlled/run.py).

Executor jobs (PII in a process pool) are real work. They still run in the
real executor -- so work that happens in a worker process in production happens
there here too -- but the loop waits for each one at submission. CPU work takes
zero simulated time anyway, and this makes completion order equal submission
order: with concurrent flows, the order in which real jobs finished would
otherwise decide, e.g., which request reaches the System-2 judge first and which
one is shed -- a wall-clock dependency the controlled runs must not have."""
import asyncio
import selectors
from concurrent.futures import ThreadPoolExecutor


class VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(selectors.DefaultSelector())
        self._vt = 0.0
        self._vc_default = None
        real_select = self._selector.select

        def select(timeout=None):
            events = real_select(0)
            if events:
                return events
            if timeout is None:
                raise RuntimeError("virtual clock deadlock: nothing scheduled and nothing in flight")
            self._vt += timeout
            return events

        self._selector.select = select

    def time(self):
        return self._vt

    def run_in_executor(self, executor, func, *args):
        if executor is None:
            if self._vc_default is None:
                self._vc_default = ThreadPoolExecutor(max_workers=1)
            executor = self._vc_default
        fut = self.create_future()
        try:
            fut.set_result(executor.submit(func, *args).result())
        except BaseException as exc:              # noqa: BLE001 -- propagate into the awaiting task
            fut.set_exception(exc)
        return fut

    async def shutdown_default_executor(self, timeout=None):
        # The stock version waits on a helper thread with a timeout, which the
        # virtual clock would skip straight past. Shut down synchronously instead.
        self._executor_shutdown_called = True
        if self._vc_default is not None:
            self._vc_default.shutdown(wait=True)
            self._vc_default = None
        if self._default_executor is not None:
            self._default_executor.shutdown(wait=True)
            self._default_executor = None


def run(coro):
    """asyncio.run() on a fresh virtual-clock loop."""
    with asyncio.Runner(loop_factory=VirtualClockLoop) as r:
        return r.run(coro)
