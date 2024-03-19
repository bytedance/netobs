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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import ferminet.constants
import ferminet.hamiltonian
import ferminet.mcmc
import ferminet.networks
import ferminet.networks as fermi_network
import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing_extensions import TypedDict

from netobs.adaptors import NetworkAdaptor, WalkingStep
from netobs.helpers.asbl_config import absl_config
from netobs.helpers.grad import make_kinetic_energy
from netobs.logging import logger
from netobs.systems import System
from netobs.systems.molecule import MolecularSystem

if TYPE_CHECKING:
    from ml_collections import ConfigDict

FERMINET_VERSION: Literal["main", "jax"]

if hasattr(fermi_network, "FermiNetOptions"):
    FERMINET_VERSION = "main"
else:
    FERMINET_VERSION = "jax"
logger.info("Assuming running with FermiNet on %s.", FERMINET_VERSION)

if FERMINET_VERSION == "main":
    import ferminet.envelopes
    from ferminet.networks import FermiNetData


class FerminetVMCAuxData(TypedDict):
    mcmc_width: jnp.ndarray


def DEFAULT(*args):
    if FERMINET_VERSION == "main":
        return FermiNetMainAdaptor(*args)
    else:
        return FermiNetJAXAdaptor(*args)


def resolve_config(config: Any, args: list[str]) -> ConfigDict:
    if isinstance(config, str):
        cfg = absl_config(config, args)
    else:
        cfg = config
    try:
        from ferminet.base_config import resolve

        return resolve(cfg)
    except ImportError:
        return cfg


@register_pytree_node_class
class FermiNetJAXAdaptor(NetworkAdaptor[MolecularSystem]):
    def __init__(self, config: Any, args: list[str]) -> None:
        super().__init__(config, args)
        self.config = resolve_config(config, args)

    def restore(
        self, ckpt_file: str | None = None
    ) -> tuple[Any, jnp.ndarray, MolecularSystem, Any]:
        cfg = self.config
        if ckpt_file is None:
            ckpt_file = str(
                Path(cfg.log.restore_path)
                / f"qmcjax_ckpt_{cfg.optim.iterations-1:06d}.npz"
            )
        with open(ckpt_file, "rb") as f:
            ckpt_data = jnp.load(f, allow_pickle=True)
            data = ckpt_data["data"]
            params = ckpt_data["params"].tolist()
            mcmc_width = jnp.array(ckpt_data["mcmc_width"].tolist())

        if data.shape[0] != jax.local_device_count():
            raise ValueError(
                "Incorrect number of devices found. Expected {}, found {}.".format(
                    data.shape[0], jax.local_device_count()
                )
            )

        self.atoms = atoms = jnp.stack(
            [jnp.array(atom.coords) for atom in cfg.system.molecule]
        )
        charges = jnp.array([atom.charge for atom in cfg.system.molecule])

        self.kinetic_energy = make_kinetic_energy(
            self.call_network, ferminet.hamiltonian.local_kinetic_energy
        )

        self.fermi_net = partial(
            fermi_network.fermi_net,
            spins=cfg.system.electrons,
            charges=charges,
            envelope_type=cfg.network.envelope_type,
            full_det=cfg.network.full_det,
        )

        return (
            params,
            cast(jnp.ndarray, data),
            MolecularSystem(
                atoms=atoms,
                charges=charges,
                spins=cfg.system.electrons,
                ndim=cfg.system.ndim,
            ),
            FerminetVMCAuxData(mcmc_width=mcmc_width),
        )

    def call_signed_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: MolecularSystem
    ):
        return self.fermi_net(params, electrons, atoms=system["atoms"])

    def make_walking_step(
        self, batch_log_psi: Callable, steps: int, system: MolecularSystem
    ) -> WalkingStep[FerminetVMCAuxData]:
        mcmc_step = ferminet.mcmc.make_mcmc_step(
            lambda params, electrons, *_: batch_log_psi(params, electrons, system),
            batch_per_device=1,  # useless
            steps=steps,
        )

        def walk(
            key: jnp.ndarray,
            params: jnp.ndarray,
            electrons: jnp.ndarray,
            aux_data: FerminetVMCAuxData,
        ) -> tuple[jnp.ndarray, FerminetVMCAuxData]:
            new_data, _ = mcmc_step(params, electrons, key, aux_data["mcmc_width"])
            return new_data, aux_data

        return jax.pmap(walk)

    def call_local_kinetic_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: MolecularSystem,
    ) -> jnp.ndarray:
        del key
        return self.kinetic_energy(params, electrons, system)

    def call_local_potential_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: MolecularSystem,
    ) -> jnp.ndarray:
        del params, key
        return potential_energy(electrons, system)


@register_pytree_node_class
class FermiNetMainAdaptor(NetworkAdaptor):
    def __init__(self, config: str, args: list[str]) -> None:
        super().__init__(config, args)
        self.config = resolve_config(config, args)

    def restore(
        self, ckpt_file: str | None = None
    ) -> tuple[Any, jnp.ndarray, System, Any]:
        cfg = self.config
        if ckpt_file is None:
            ckpt_file = str(
                Path(cfg.log.restore_path)
                / f"qmcjax_ckpt_{cfg.optim.iterations-1:06d}.npz"
            )
        with open(ckpt_file, "rb") as f:
            ckpt_data = jnp.load(f, allow_pickle=True)
            data = ckpt_data["data"].item()["positions"]
            params = ckpt_data["params"].tolist()
            mcmc_width = jnp.array(ckpt_data["mcmc_width"].tolist())

        if data.shape[0] != jax.local_device_count():
            raise ValueError(
                "Incorrect number of devices found. Expected {}, found {}.".format(
                    data.shape[0], jax.local_device_count()
                )
            )

        atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
        charges = jnp.array([atom.charge for atom in cfg.system.molecule])

        net = fermi_network.make_fermi_net(
            cfg.system.electrons,
            charges,
            ndim=cfg.system.ndim,
            determinants=cfg.network.determinants,
            states=cfg.system.states,
            envelope=ferminet.envelopes.make_isotropic_envelope(),
            feature_layer=fermi_network.make_ferminet_features(
                charges, cfg.system.electrons, cfg.system.ndim
            ),
            jastrow=cfg.network.get("jastrow", "default"),
            bias_orbitals=cfg.network.bias_orbitals,
            full_det=cfg.network.full_det,
            rescale_inputs=cfg.network.get("rescale_inputs", False),
            complex_output=cfg.network.get("complex", False),
            **cfg.network.ferminet,
        )
        self.fermi_net = net.apply
        self.fermi_data = FermiNetData(
            positions=None, spins=cfg.system.electrons, atoms=atoms, charges=charges
        )

        self.kinetic_energy = ferminet.hamiltonian.local_kinetic_energy(net.apply)

        return (
            params,
            cast(jnp.ndarray, data),
            MolecularSystem(
                atoms=atoms,
                charges=charges,
                spins=cfg.system.electrons,
                ndim=cfg.system.ndim,
            ),
            FerminetVMCAuxData(mcmc_width=mcmc_width),
        )

    def call_signed_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: MolecularSystem
    ):
        return self.fermi_net(
            params, electrons, system["spins"], system["atoms"], system["charges"]
        )

    def make_walking_step(
        self, batch_log_psi: Callable, steps: int, system: MolecularSystem
    ) -> WalkingStep[FerminetVMCAuxData]:
        mcmc_step = ferminet.mcmc.make_mcmc_step(
            lambda params, electrons, *_: batch_log_psi(params, electrons, system),
            batch_per_device=1,  # useless
            steps=steps,
        )

        def walk(
            key: jnp.ndarray,
            params: jnp.ndarray,
            electrons: jnp.ndarray,
            aux_data: FerminetVMCAuxData,
        ) -> tuple[jnp.ndarray, FerminetVMCAuxData]:
            data = dataclasses.replace(self.fermi_data, positions=electrons)
            new_data, _ = mcmc_step(params, data, key, aux_data["mcmc_width"])
            return new_data["positions"], aux_data

        return jax.pmap(walk)

    def call_local_kinetic_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: MolecularSystem,
    ) -> jnp.ndarray:
        del key
        data = dataclasses.replace(
            self.fermi_data, positions=electrons, atoms=system["atoms"]
        )
        return self.kinetic_energy(params, data)

    def call_local_potential_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: MolecularSystem,
    ) -> jnp.ndarray:
        del params, key
        return potential_energy(electrons, system)


def potential_energy(x: jnp.ndarray, system: MolecularSystem) -> jnp.ndarray:
    """The potential energy, rewritten to avoid NaN in grad.

    Args:
        x: electron positions. Shape (nelectrons*ndim,).
        system: system containing atomic information.

    Returns:
        The potential energy for this electron configuration.
    """
    atoms, charges = system["atoms"], system["charges"]
    ae = jnp.reshape(x, [-1, 1, 3]) - atoms[None, ...]
    ee = jnp.reshape(x, [1, -1, 3]) - jnp.reshape(x, [-1, 1, 3])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    # Zero-valued diagonals will lead to NaNs in gradients.
    # But diagonals are not used, so there is no need to put them to zero as FermiNet
    r_ee = jnp.linalg.norm(ee + jnp.eye(ee.shape[0])[..., None], axis=-1, keepdims=True)

    v_ee = jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))
    v_ae = -jnp.sum(charges / r_ae[..., 0])
    # Diagonals are not used, so there is no need to put them to zero
    r_aa = jnp.linalg.norm(
        atoms[None, ...] - atoms[:, None] + jnp.eye(atoms.shape[0])[..., None], axis=-1
    )
    v_aa = jnp.sum(jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
    return v_ee + v_ae + v_aa
