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

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

import jax
from jax import numpy as jnp

from netobs.adaptors import NetworkAdaptor
from netobs.checkpoint import CheckpointManager
from netobs.helpers.digest import robust_mean_std
from netobs.logging import logger
from netobs.observables import Estimator
from netobs.options import NetObsOptions

# Utilities for paralleled random keys
broadcast_all_local_devices = jax.pmap(lambda x: x)
p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def make_different_rng_key_on_all_devices(
    rng: jnp.ndarray,
) -> jnp.ndarray:
    rng = jax.random.fold_in(rng, jax.process_index())
    rng = jax.random.split(rng, jax.device_count())
    return broadcast_all_local_devices(rng)


def evaluate_observable(
    network_adaptor: NetworkAdaptor,
    estimator_class: type[Estimator],
    checkpoint_mgr: Optional[CheckpointManager] = None,
    options: NetObsOptions | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run inference to evaluate observable for a system.

    Args:
        network_adaptor: Network adaptor.
        estimator_class: An estimator class. See estimators.py for detail.
        checkpoint_mgr: The checkpoint manager.
        options: The NetObsOptions. See the docstring of NetObsOptions for detail.

    Returns:
        * entries returned by `estimator.digest`
        * values and states returned by `estimator.evaluate`
    """
    if checkpoint_mgr is None:
        checkpoint_mgr = CheckpointManager()
    if options is None:
        options = NetObsOptions()

    params, data, system, aux_data = network_adaptor.restore(options.network_restore)

    key = jax.random.PRNGKey(options.random_seed)

    estimator = estimator_class(
        network_adaptor, system, options.estimator, options.observable
    )

    sharded_key = make_different_rng_key_on_all_devices(key)

    init_step, new_data, all_values, state, aux_data = checkpoint_mgr.restore(
        *estimator.empty_val_state(options.steps), aux_data
    )
    if new_data is not None:
        data = new_data

    metadata = {
        "version": 9,
        "estimator": estimator_class.__name__,
        **dataclasses.asdict(options),
    }

    if init_step >= options.steps:  # re-exporting data
        digest = estimator.digest(all_values, state)
        for k, v in digest.items():
            print(k, v, sep="\n")
        return digest, all_values, state

    logger.info("Start burning in %s steps", options.mcmc_burn_in)
    call_burnin_step = network_adaptor.make_burnin_step(
        options.mcmc_burn_in * options.mcmc_steps, system
    )
    sharded_key, subkeys = p_split(sharded_key)
    call_burnin_step(subkeys, params, data, aux_data)

    call_walking_step = network_adaptor.make_walking_step(options.mcmc_steps, system)

    time_start = None  # initialize later to be more accurate
    last_log = time.time()
    last_save = time.time()
    logger.info("Starting %s evaluation steps", options.steps - init_step)
    for i in range(init_step, options.steps):
        sharded_key, subkeys = p_split(sharded_key)
        obs_values, state = estimator.evaluate(
            i, params, subkeys, data, system, state, aux_data
        )
        all_values = {k: v.at[i].set(obs_values[k]) for k, v in all_values.items()}
        sharded_key, subkeys = p_split(sharded_key)
        data, aux_data = call_walking_step(subkeys, params, data, aux_data)

        # Logging and saving
        now = time.time()
        if time_start is None:
            time_start = now
        should_log = last_log < now - options.log_interval
        should_save = last_save < now - options.save_interval
        if not should_save and not should_log:
            continue
        all_values_yet = {k: v[: i + 1] for k, v in all_values.items()}
        digest = estimator.digest(all_values_yet, state)

        if should_save:
            last_save = now
            checkpoint_mgr.save(i, data, digest, all_values, state, aux_data, metadata)
        # Also log when should save
        last_log = now
        logger.info("Loop %s", i)
        log_digest(i, digest)

    if i == init_step or time_start is None:
        logger.warning("Not enough steps to calculate time per step.")
    else:
        logger.info(
            "Time per step: %.2fs", (time.time() - time_start) / (i - init_step)
        )
    digest = estimator.digest(all_values, state)
    log_digest(i, digest)
    checkpoint_mgr.save(i, data, digest, all_values, state, aux_data, metadata)
    return digest, all_values, state


def log_digest(i: int, digest: dict[str, jnp.ndarray]) -> None:
    for k, v in digest.items():
        if v.shape and v.shape[0] == i + 1:
            v, std_v = robust_mean_std(v)
            if v.shape:
                print(k, v, f"std {k}", std_v, sep="\n")
            else:
                print(k + ":", v, "\xb1", std_v)
        else:
            print(k, v, sep="\n")
