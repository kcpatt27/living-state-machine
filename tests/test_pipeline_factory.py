"""Default pipeline wiring."""

from __future__ import annotations

from lsm.runtime.pipeline_factory import build_default_pipeline


def test_build_default_pipeline_wires_state_and_memory() -> None:
    pipe, backend, episodic, store, memory, outcome_log = build_default_pipeline(session_id="test")
    try:
        assert pipe.state_store is store
        assert pipe.state_updater is not None
        assert pipe.memory is memory
        assert pipe.habituation is not None
        assert pipe.association is not None
    finally:
        memory.close()
        outcome_log.close()
        episodic.close()
