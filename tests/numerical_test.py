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

import pytest
from jax import numpy as jnp

from netobs.adaptors.simple_hydrogen import SimpleHydrogen
from netobs.evaluate import evaluate_observable
from netobs.helpers.digest import robust_mean
from netobs.observables.density import DensityEstimator
from netobs.observables.energy import EnergyEstimator
from netobs.observables.force import AC, SWCT, Bare
from netobs.observables.wf_change import WFChangeEstimator
from netobs.options import NetObsOptions

pytest.importorskip(
    "jax",
    minversion="0.4.0",
    reason="Different JAX version has different behaviors. Please use JAX 0.4.x",
)


@pytest.fixture
def adaptor():
    return SimpleHydrogen(0.999, [])


@pytest.fixture
def options():
    return NetObsOptions(
        steps=20,
        random_seed=42,
        mcmc_burn_in=100,
        mcmc_steps=10,
        network_restore="2048",
    )


def test_energy(adaptor: SimpleHydrogen, options: NetObsOptions, snapshot: str):
    digest, *_ = evaluate_observable(adaptor, EnergyEstimator, options=options)
    assert digest is not None
    assert "energy" in digest
    energy = robust_mean(digest["energy"])
    assert jnp.allclose(energy, -0.5, rtol=5e-4)
    assert [robust_mean(value) for value in digest.values()] == snapshot


def test_density(adaptor: SimpleHydrogen, options: NetObsOptions, snapshot):
    options.estimator["bins"] = [1, 2, 3]
    options.estimator["range"] = [[-3, 3], [-3, 3], [-3, 3]]
    options.steps = 1
    options.mcmc_burn_in = 0
    *_, state = evaluate_observable(adaptor, DensityEstimator, options=options)
    assert "map" in state
    assert jnp.sum(state["map"]) < 2048
    assert state["map"].shape == (1, 2, 3)
    assert state["map"] == snapshot


def test_bare(adaptor: SimpleHydrogen, options: NetObsOptions, snapshot: str):
    digest, *_ = evaluate_observable(adaptor, Bare, options=options)
    assert digest is not None
    assert "force" in digest
    force = robust_mean(digest["force"])
    assert jnp.allclose(force, 0, atol=0.2)
    assert force == snapshot


def test_antithetic(adaptor: SimpleHydrogen, options: NetObsOptions, snapshot: str):
    options.estimator["r_core"] = 0.5
    digest, *_ = evaluate_observable(adaptor, Bare, options=options)
    assert digest is not None
    assert "force" in digest
    force = robust_mean(digest["force"])
    assert jnp.allclose(force, 0, atol=1e-3)
    assert force == snapshot


def test_antithetic_zb(adaptor: SimpleHydrogen, options: NetObsOptions, snapshot):
    options.estimator["r_core"] = 0.5
    options.estimator["zb"] = True
    digest, *_ = evaluate_observable(adaptor, Bare, options=options)
    assert digest is not None
    assert "force" in digest
    force = robust_mean(digest["force"])
    assert jnp.allclose(force, 0, atol=1e-3)
    assert force == snapshot


def test_zv_noerror(adaptor: SimpleHydrogen, options: NetObsOptions):
    options.mcmc_burn_in = 0
    options.steps = 1
    digest, *_ = evaluate_observable(adaptor, AC, options=options)
    assert digest is not None
    assert "force" in digest


def test_zvzb(adaptor: SimpleHydrogen, options: NetObsOptions, snapshot: str):
    options.estimator["zb"] = True
    digest, *_ = evaluate_observable(adaptor, AC, options=options)
    assert digest is not None
    assert "force" in digest
    assert "force_zv" in digest
    force = robust_mean(digest["force"])
    assert jnp.allclose(force, 0, atol=1e-5)
    force_zv = robust_mean(digest["force_zv"])
    assert jnp.allclose(force_zv, 0, atol=1e-6)
    assert (force, force_zv) == snapshot


def test_swct(adaptor: SimpleHydrogen, options: NetObsOptions, snapshot: str):
    digest, *_ = evaluate_observable(adaptor, SWCT, options=options)
    assert digest is not None
    assert "force" in digest
    assert "force_no_warp" in digest
    force = robust_mean(digest["force"])
    assert jnp.allclose(force, 0, atol=1e-7)
    force_no_warp = robust_mean(digest["force_no_warp"])
    assert jnp.allclose(force_no_warp, 0, atol=1e-3)
    assert (force, force_no_warp) == snapshot


def test_wf_change(adaptor: SimpleHydrogen, options: NetObsOptions, snapshot: str):
    digest, *_ = evaluate_observable(adaptor, WFChangeEstimator, options=options)
    assert digest is not None
    assert "overlap" in digest
    overlap = digest["overlap"]
    assert jnp.allclose(overlap, 1, rtol=1e-4)
    assert overlap == snapshot
