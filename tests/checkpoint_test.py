# Copyright 2022-2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from pathlib import Path

import pytest

from netobs.adaptors.simple_hydrogen import SimpleHydrogen
from netobs.checkpoint import SavingCheckpointManager
from netobs.evaluate import evaluate_observable
from netobs.observables.energy import EnergyEstimator
from netobs.observables.force import AC
from netobs.options import NetObsOptions


@pytest.fixture
def adaptor():
    return SimpleHydrogen(1, [])


@pytest.fixture
def options():
    return NetObsOptions(
        steps=1,
        random_seed=42,
        mcmc_burn_in=0,
        mcmc_steps=1,
        network_restore="1",
    )


@pytest.fixture
def ckpt_mgr():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SavingCheckpointManager(tmpdir, tmpdir)


def test_save_and_restore(
    adaptor: SimpleHydrogen, options: NetObsOptions, ckpt_mgr: SavingCheckpointManager
):
    evaluate_observable(
        adaptor, EnergyEstimator, options=options, checkpoint_mgr=ckpt_mgr
    )
    ckpts = os.listdir(ckpt_mgr.save_path)
    assert len(ckpts) == 1
    assert ckpts[0].endswith("00.npz")

    options.steps += 1
    evaluate_observable(
        adaptor, EnergyEstimator, options=options, checkpoint_mgr=ckpt_mgr
    )
    ckpts = sorted(os.listdir(ckpt_mgr.save_path))
    assert len(ckpts) == 2
    assert ckpts[1].endswith("01.npz")


def test_restore_zb_evaluate_zv(
    adaptor: SimpleHydrogen, options: NetObsOptions, ckpt_mgr: SavingCheckpointManager
):
    options.estimator["zb"] = True
    _, values, _ = evaluate_observable(
        adaptor, AC, options=options, checkpoint_mgr=ckpt_mgr
    )
    assert "el" in values

    options.steps += 1
    options.estimator["zb"] = False

    with tempfile.TemporaryDirectory() as save_dir:
        ckpt_mgr.save_path = Path(save_dir)
        _, values, _ = evaluate_observable(
            adaptor, AC, options=options, checkpoint_mgr=ckpt_mgr
        )
        assert list(values.keys()) == ["hf_term"]
        assert len(os.listdir(ckpt_mgr.restore_path)) == 1
        assert len(os.listdir(save_dir)) == 1
