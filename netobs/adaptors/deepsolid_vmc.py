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

from functools import partial
from pathlib import Path
from typing import Any, Callable

import DeepSolid.hamiltonian
import DeepSolid.hf
import DeepSolid.network
import DeepSolid.qmc
import DeepSolid.supercell
import jax
from DeepSolid.ewaldsum import EwaldSum
from DeepSolid.supercell import get_supercell_copies
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing_extensions import TypedDict

from netobs.adaptors import NetworkAdaptor, WalkingStep
from netobs.helpers.asbl_config import absl_config
from netobs.helpers.grad import grad_with_system, make_kinetic_energy
from netobs.systems.solid import MinimalImageDistance, SolidSystem


class DeepSolidVMCAuxData(TypedDict):
    """Auxiliary data for DeepSolid."""

    mcmc_width: jnp.ndarray


@register_pytree_node_class
class DeepSolidVMCAdaptor(NetworkAdaptor[SolidSystem]):
    def __init__(self, config: Any, args: list[str]) -> None:
        super().__init__(config, args)
        self.config = absl_config(config, args) if isinstance(config, str) else config

    def restore(
        self, ckpt_file: str | None = None, check_shape: bool = True
    ) -> tuple[Any, jnp.ndarray, SolidSystem, DeepSolidVMCAuxData]:
        cfg = self.config
        restore_path = Path(cfg.log.restore_path)
        ckpt_filename = (
            str(restore_path / f"qmcjax_ckpt_{cfg.optim.iterations-1:06d}.npz")
            if ckpt_file is None
            else ckpt_file
        )
        with open(ckpt_filename, "rb") as f:
            ckpt_data = jnp.load(f, allow_pickle=True)
            data = ckpt_data["data"]
            params = ckpt_data["params"].tolist()
            mcmc_width = jnp.array(ckpt_data["mcmc_width"].tolist())

        cards = jax.local_device_count()
        if check_shape and cards != data.shape[0]:
            print(
                f"Converting checkpoint with {data.shape[0]} devices to {cards} devices"
            )
            data = data.reshape(cards, -1, data.shape[-1])
            params = jax.tree_map(lambda x: jnp.repeat(x[:1], cards, axis=0), params)
            mcmc_width = jnp.repeat(mcmc_width[:1], cards, axis=0)

        simulation_cell = cfg.system.pyscf_cell

        hartree_fock = DeepSolid.hf.SCF(
            cell=simulation_cell,
            twist=jnp.array(cfg.network.twist),
        )
        hartree_fock.init_scf()
        klist = hartree_fock.klist

        self.atoms = atoms = simulation_cell.original_cell.atom_coords()
        charges = simulation_cell.original_cell.atom_charges()
        latvec = jnp.asarray(simulation_cell.original_cell.lattice_vectors())
        self.orig_dist = MinimalImageDistance(latvec)
        self.cell_size = simulation_cell.scale

        system = SolidSystem(
            atoms=atoms,
            charges=charges,
            spins=simulation_cell.nelec,
            ndim=cfg.system.ndim,
            latvec=simulation_cell.lattice_vectors(),
        )

        self.net_func = lambda params, electrons, atoms: DeepSolid.network.eval_func(
            params,
            electrons,
            atoms=atoms,
            klist=klist,
            simulation_cell=simulation_cell,
            spins=simulation_cell.nelec,
            envelope_type=cfg.network.detnet.envelope_type,
            full_det=cfg.network.detnet.full_det,
            distance_type=cfg.network.detnet.distance_type,
            method_name="eval_phase_and_slogdet",
        )

        self.local_kinetic = make_kinetic_energy(
            self.call_complex_network,
            DeepSolid.hamiltonian.local_kinetic_energy_real_imag,
        )
        self.local_potential = local_ewald_energy(simulation_cell)
        twist = jnp.mod(jnp.array(cfg.network.twist), 1.0)
        ks = jnp.dot(jnp.linalg.inv(simulation_cell.a), twist) * 2 * jnp.pi
        if simulation_cell._ecp:
            self.non_local_potential = DeepSolid.hamiltonian.non_local_energy(
                partial(self.call_signed_network, system=system),
                simulation_cell,
                minimal_imag_dis=MinimalImageDistance(simulation_cell.a),
                ecp_quadrature_id=cfg.system.ecp_quadrature_id,
                partition_number=sum(simulation_cell.nelec)
                if cfg.optim.ecp_partition_number == -1
                else cfg.optim.ecp_partition_number,
                ks=ks,
            )
        else:
            self.non_local_potential = lambda *_: 0

        return (
            params,
            data,
            system,
            DeepSolidVMCAuxData(mcmc_width=mcmc_width),
        )

    def call_local_kinetic_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: SolidSystem,
    ) -> jnp.ndarray:
        del key
        return sum(self.local_kinetic(params, electrons, system)) / self.cell_size

    def call_local_potential_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: SolidSystem,
    ) -> jnp.ndarray:
        del key
        ep = self.local_potential(electrons, system)
        # HACK: grad potential requires real-valued potential
        ecp = self.non_local_potential(params, electrons, system["atoms"]).real
        return (ep + ecp) / self.cell_size

    def call_signed_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: SolidSystem
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.net_func(params, electrons, system["atoms"])

    def call_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: SolidSystem
    ) -> jnp.ndarray:
        _, output = self.net_func(params, electrons, system["atoms"])
        return jnp.real(output)

    def call_complex_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: SolidSystem
    ) -> jnp.ndarray:
        _, output = self.net_func(params, electrons, system["atoms"])
        return output

    def make_walking_step(
        self, batch_log_psi: Callable, steps: int, system: SolidSystem
    ) -> WalkingStep[DeepSolidVMCAuxData]:
        mcmc_step = DeepSolid.qmc.make_mcmc_step(
            lambda params, electrons: batch_log_psi(params, electrons, system),
            batch_per_device=1,  # useless
            latvec=system["latvec"],
            steps=steps,
        )

        def walk(
            key: jnp.ndarray,
            params: jnp.ndarray,
            electrons: jnp.ndarray,
            aux_data: DeepSolidVMCAuxData,
        ) -> tuple[jnp.ndarray, DeepSolidVMCAuxData]:
            new_data, _ = mcmc_step(params, electrons, key, aux_data["mcmc_width"])
            return new_data, aux_data

        return jax.pmap(walk)

    def make_signed_network_grad(self, arg: str, jaxfun: Callable = jax.grad):
        def complex_f(params, electrons, system):
            sign, slogdet = self.call_signed_network(params, electrons, system)
            return jnp.log(sign) + slogdet

        grad_f_real = grad_with_system(
            lambda *args: complex_f(*args).real, arg, args_before=1, jaxfun=jaxfun
        )
        grad_f_imag = grad_with_system(
            lambda *args: complex_f(*args).imag, arg, args_before=1, jaxfun=jaxfun
        )
        return lambda *args: grad_f_real(*args) + grad_f_imag(*args) * 1j

    def make_local_energy_grad(self, arg: str, jaxfun: Callable = jax.grad):
        grad_local_energy_real = grad_with_system(
            lambda *args: self.call_local_energy(*args).real,
            arg,
            args_before=2,
            jaxfun=jaxfun,
        )
        grad_local_energy_imag = grad_with_system(
            lambda *args: self.call_local_energy(*args).imag,
            arg,
            args_before=2,
            jaxfun=jaxfun,
        )
        return (
            lambda *args: grad_local_energy_real(*args)
            + grad_local_energy_imag(*args) * 1j
        )


def local_ewald_energy(simulation_cell):
    ewald = EwaldSum(simulation_cell)
    nelec = sum(simulation_cell.nelec)
    ewald.dist = MinimalImageDistance(ewald.latvec)
    Rpts = get_supercell_copies(
        simulation_cell.original_cell.lattice_vectors(), simulation_cell.S
    )

    def _local_ewald_energy(data: jnp.ndarray, system: SolidSystem):
        ewald.atom_coords = jnp.reshape(system["atoms"] + Rpts[:, None, :], (-1, 3))
        ii = ewald.ewald_ion() + ewald.ii_const
        ee, ei = ewald.ewald_electron(data)
        ee += ewald.ee_const(nelec)
        ei += ewald.ei_const(nelec)
        return jnp.asarray(ee) + jnp.asarray(ei) + jnp.asarray(ii)

    return _local_ewald_energy


DEFAULT = DeepSolidVMCAdaptor
