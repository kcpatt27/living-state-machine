"""fact_correction_v1 scripted scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from lsm.eval.tasks.fact_correction_v1 import SCENARIOS, run_scenario


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_id", [s.scenario_id for s in SCENARIOS])
@pytest.mark.parametrize("updater_mode", ["facts_backed", "llm_summarizer"])
async def test_fact_correction_scenarios(
    tmp_path: Path,
    scenario_id: str,
    updater_mode: Literal["facts_backed", "llm_summarizer"],
) -> None:
    sc = next(s for s in SCENARIOS if s.scenario_id == scenario_id)
    sub = tmp_path / f"{scenario_id}-{updater_mode}"
    sub.mkdir()
    store, state = await run_scenario(sc, sub, updater_mode=updater_mode)
    from lsm.eval.tasks.fact_correction_v1 import assert_scenario

    failures = assert_scenario(sc, store, state, enforce_st_equals=updater_mode == "facts_backed")
    assert not failures, failures
