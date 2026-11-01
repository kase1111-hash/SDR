"""K-P2: ``gui/frame_pipeline.FramePipeline`` runs a display loop's per-frame
job on a thread of its own, one frame behind: each frame collects the last
frame's result (waiting only while that job still runs) and submits its
own. One job at a time, so results come back in order and the loop never
runs more than a frame ahead of the worker."""

import threading

import pytest

from sdr_module.gui import frame_pipeline
from sdr_module.gui.frame_pipeline import FramePipeline


def test_results_come_back_in_order_one_frame_behind():
    pipeline = FramePipeline()
    threads = []

    def job(n):
        threads.append(threading.get_ident())
        return n * 10

    shown = []
    for n in range(5):
        shown.append(pipeline.collect())  # the last frame's result
        pipeline.submit(lambda n=n: job(n))
    shown.append(pipeline.collect())
    pipeline.close()
    assert shown == [None, 0, 10, 20, 30, 40]
    assert threading.get_ident() not in threads
    assert not pipeline.running


def test_submit_returns_while_the_job_runs():
    pipeline = FramePipeline()
    gate, started = threading.Event(), threading.Event()

    def job():
        started.set()
        assert gate.wait(10)
        return "done"

    pipeline.submit(job)  # (returns at once)
    assert started.wait(5)
    assert pipeline.pending and not pipeline.ready()
    gate.set()
    assert pipeline.collect() == "done"
    assert not pipeline.pending
    pipeline.close()


def test_one_job_at_a_time():
    pipeline = FramePipeline()
    pipeline.submit(lambda: 1)
    with pytest.raises(RuntimeError):
        pipeline.submit(lambda: 2)  # (collect the last one first)
    assert pipeline.collect() == 1
    pipeline.close()


def test_an_error_in_a_job_reaches_the_caller():
    pipeline = FramePipeline()

    def job():
        raise ValueError("broken frame")

    pipeline.submit(job)
    with pytest.raises(ValueError, match="broken frame"):
        pipeline.collect()
    pipeline.submit(lambda: "next")  # (and frames go on)
    assert pipeline.collect() == "next"
    pipeline.close()


def test_ready_tells_without_waiting():
    pipeline = FramePipeline()
    assert not pipeline.ready()  # (nothing submitted)
    pipeline.submit(lambda: 7)
    pipeline.close()  # (the job finishes first; its result stays)
    assert pipeline.ready()
    assert pipeline.collect() == 7


def test_an_idle_thread_ends_by_itself(monkeypatch):
    monkeypatch.setattr(frame_pipeline, "IDLE_EXIT_S", 0.01)
    pipeline = FramePipeline()
    pipeline.submit(lambda: 1)
    assert pipeline.collect() == 1
    thread = pipeline._thread
    if thread is not None:
        thread.join(5)
    assert not pipeline.running  # (a pipeline dropped unclosed leaks nothing)
    pipeline.submit(lambda: 2)  # and starts again when needed
    assert pipeline.collect() == 2
    pipeline.close()
