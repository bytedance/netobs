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

from typing import Any
from zipfile import BadZipFile

import jax
import numpy as np
from jax import numpy as jnp
from upath import UPath

from netobs.logging import logger


class CheckpointManager:
    """Base checkpoint manager doing nothing.

    Args:
        restore_path: UPath to restore from.
        save_path: UPath to save checkpoints to.
    """

    def __init__(
        self,
        restore_path: UPath | str = UPath("."),
        save_path: UPath | str = UPath("."),
    ) -> None:
        del restore_path, save_path

    def restore(
        self,
        empty_values: dict[str, jnp.ndarray],
        state: dict[str, Any],
        aux_data: dict[str, Any],
    ) -> tuple[int, jnp.ndarray | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Restore a netobs checkpoint.

        Args:
            empty_values: template for empty estimator values
            state: empty estimator state
            aux_data: initial network aux data

        Returns:
            Tuple of checkpoint parameters
              - Starting iteration
              - MCMC configuration
              - Observable value at each steps
              - Aux state for estimators
        """
        return 0, None, empty_values, state, aux_data

    def save(
        self,
        i: int,
        data: jnp.ndarray,
        digest: dict[str, jnp.ndarray],
        all_values: dict[str, jnp.ndarray],
        state: dict[str, Any],
        aux_data: dict[str, Any],
        metadata: dict,
    ) -> UPath | None:
        """Save a netobs checkpoint, optionally.

        Args:
            i: The current iteration
            data: MCMC configuration
            digest: Calculated results from values and state
            all_values: Observable parts at each steps
            state: Energy state
            aux_data: Network aux data
            metadata: Metadata.

        Returns:
            Optionally the path where the checkpoint was saved.
        """
        del i, data, digest, all_values, state, aux_data, metadata
        return None


class SavingCheckpointManager(CheckpointManager):
    """Create a checkpoint manager that actually saves something."""

    def __init__(
        self,
        restore_path: UPath | str = UPath("."),
        save_path: UPath | str = UPath("."),
    ) -> None:
        super().__init__()

        self.restore_path = UPath(restore_path)
        self.save_path = UPath(save_path)
        self.save_path.mkdir(exist_ok=True, parents=True)

    def restore(
        self,
        empty_values: dict[str, jnp.ndarray],
        state: dict[str, Any],
        aux_data: dict[str, Any],
    ) -> tuple[int, jnp.ndarray | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
        if not self.restore_path.exists():
            return super().restore(empty_values, state, aux_data)

        ckpt_files = sorted(
            list(self.restore_path.glob("netobs_ckpt_*.npz")), reverse=True
        )
        cards = jax.local_device_count()
        for ckpt_file in ckpt_files:
            try:
                with ckpt_file.open("rb") as f:
                    ckpt_content = jnp.load(f, allow_pickle=True)
                    init_step = int(ckpt_content["i"]) + 1
                    # Get required states from archive based on given `state`.
                    state = {
                        k: ckpt_content.get(f"state/{k}", v) for k, v in state.items()
                    }  # type: ignore
                    data = ckpt_content["data"]
                    aux_data = ckpt_content["aux_data"].item()
                    # Automatically reshapes data based on number of devices.
                    if cards != data.shape[0]:
                        data = data.reshape(cards, -1, data.shape[-1])
                        aux_data = jax.tree_map(
                            lambda x: jnp.repeat(x[:1], cards, axis=0), aux_data
                        )
                    # No restore of values is required
                    if not empty_values:
                        return (init_step, data, empty_values, state, aux_data)
                    # How many steps we have run?
                    first_key = list(empty_values.keys())[0]
                    check_key = "values/" + first_key
                    old_steps_to_run = len(ckpt_content[check_key])
                    old_steps_have_run = int(ckpt_content["i"]) + 1
                    steps = len(empty_values[first_key])

                    # Allow equal steps to reexport report
                    if old_steps_have_run > steps:
                        print("Calculation already done. Assuming re-export")
                    if old_steps_to_run != steps:
                        steps_to_set = min(steps, old_steps_have_run)  # type: ignore
                        all_values = {
                            k: v.at[:steps_to_set].set(
                                ckpt_content[f"values/{k}"][:steps_to_set]
                            )
                            for k, v in empty_values.items()
                        }

                    else:
                        all_values = {
                            k: jnp.array(ckpt_content[f"values/{k}"])
                            for k in empty_values.keys()
                        }

                    return (init_step, data, all_values, state, aux_data)
            except (OSError, EOFError, BadZipFile):
                logger.info("Error loading %s. Trying next checkpoint...", ckpt_file)

        if ckpt_files:
            logger.warn(
                "Really BAD NEWS. No checkpoint is readable. "
                "Please check the version of checkpoint. Evaluation will start from 0."
            )
        return super().restore(empty_values, state, aux_data)

    def save(
        self,
        i: int,
        data: jnp.ndarray,
        digest: dict[str, jnp.ndarray],
        all_values: dict[str, jnp.ndarray],
        state: dict[str, Any],
        aux_data: dict[str, Any],
        metadata: dict,
    ) -> UPath | None:
        ckpt_path = self.save_path / f"netobs_ckpt_{i:06}.npz"
        with (ckpt_path).open("wb") as f:
            np.savez_compressed(  # compressed npz is only available in numpy
                f,
                i=i,
                data=data,
                metadata=metadata,  # type: ignore
                aux_data=aux_data,  # type: ignore
                **{f"digest/{k}": v for k, v in digest.items()},
                **{f"values/{k}": v for k, v in all_values.items()},
                **{f"state/{k}": v for k, v in state.items()},
            )
        logger.info("Saved checkpoint to %s", ckpt_path)
        return ckpt_path
