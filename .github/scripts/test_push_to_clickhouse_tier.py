# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""push-to-clickhouse's tier step, run as the workflow runs it.

The tier is a run_id hash input, so an empty one makes the v2 ingest skip the whole run.
The env below is what GitHub actually delivers: workflow_run.name is the run-name.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest
import yaml

_WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / "workflows" / "push-to-clickhouse.yaml"


def _tier_script() -> str:
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["ingest"]["steps"]
    return next(s["run"] for s in steps if s.get("id") == "extract-tier")


@pytest.mark.parametrize(
    ("workflow", "title", "tier"),
    [
        ("regression", "regression", "regression"),
        ("trunk", "trunk", "trunk"),
        ("integration-tests (integration)", "integration-tests (integration)", "integration"),
        ("integration-tests (perf)", "integration-tests (perf)", "perf"),
        ("test_each_commit", "regression", "regression"),
        ("integration-tests", "", ""),
    ],
)
def test_tier_from_display_title(tmp_path, workflow, title, tier):
    out = tmp_path / "out"
    env = {
        **os.environ,
        "TRIGGERING_WORKFLOW": workflow,
        "TRIGGERING_DISPLAY_TITLE": title,
        "GITHUB_OUTPUT": str(out),
    }
    subprocess.run(["bash", "-e", "-c", _tier_script()], env=env, check=True, capture_output=True)
    assert out.read_text().splitlines() == [f"tier={tier}"]
