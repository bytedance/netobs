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

# Original files: https://github.com/WagnerGroup/pyqmc/blob/master/pyqmc/ewald.py
# Copyright (c) 2019-2024 The PyQMC Developers
# SPDX-License-Identifier: MIT

from functools import partial

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from netobs.logging import logger
from netobs.observables import Estimator, Observable
from netobs.systems.solid import MinimalImageDistance, SolidSystem


class Stress(Observable[SolidSystem]):
    def shapeof(self, system) -> tuple[int, ...]:
        return (system["ndim"], system["ndim"])


@register_pytree_node_class
class StressEstimator(Estimator[SolidSystem]):
    observable_type = Stress

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.grad_f = adaptor.make_network_grad("electrons")
        self.hess_f = jax.hessian(adaptor.call_network, argnums=1)
        self.cellvol = jnp.linalg.det(jnp.asarray(self.system["latvec"]))
        self.nelec = sum(system["spins"])

        ewald = Ewald4Stress(system)
        self.batch_potential_part = jax.pmap(jax.vmap(ewald.stress))

    def empty_val_state(self, steps: int):
        term_shape = (steps, *self.observable.shape)
        dtype = self.options.get("dtype")
        return {
            "stress_kin": jnp.zeros(term_shape, dtype),
            "stress_pot": jnp.zeros(term_shape, dtype),
        }, {}

    @partial(jax.pmap, in_axes=(None, 0, 0, None))
    @partial(jax.vmap, in_axes=(None, None, 0, None))
    def batch_kinetic_part(
        self, params: jnp.ndarray, x: jnp.ndarray, system: SolidSystem
    ) -> jnp.ndarray:
        n = self.nelec
        grad_f = jnp.reshape(self.grad_f(params, x, system), (n, 3))
        hess_f = jnp.reshape(self.hess_f(params, x, system), (n, 3, n, 3))
        # (n, 3, 3)
        grad_grad_f_per_elec = jnp.einsum("ij,ik->ijk", grad_f, grad_f)
        # Take out 3x3 block diagonal from nx3xnx3 matrix
        hess_f_per_elec = jnp.moveaxis(jnp.diagonal(hess_f, axis1=0, axis2=2), 2, 0)
        return jnp.sum(grad_grad_f_per_elec + hess_f_per_elec, axis=0)

    def evaluate(self, i, params, key, data, system, state, aux_data):
        del i, aux_data, key
        return {
            "stress_kin": self.batch_kinetic_part(params, data, system),
            "stress_pot": self.batch_potential_part(data),
        }, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del state
        mean = jnp.mean(all_values["stress_kin"] + all_values["stress_pot"], axis=0)
        stress = -mean / self.cellvol
        return {"stress": stress, "trace": jnp.trace(stress)}


class Ewald4Stress:
    def __init__(self, system: SolidSystem, ewald_gmax=200, nlatvec=1):
        self.nelec = sum(system["spins"])
        self.atom_coords = system["atoms"]
        self.atom_charges = system["charges"]
        self.latvec = system["latvec"]
        self.dist = MinimalImageDistance(self.latvec)
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec):
        XYZ = jnp.meshgrid(*[jnp.arange(-nlatvec, nlatvec + 1)] * 3, indexing="ij")
        xyz = jnp.stack(XYZ, axis=-1).reshape((-1, 3))
        self.lattice_displacements = jnp.asarray(jnp.dot(xyz, self.latvec))

    def set_up_reciprocal_ewald_sum(self, ewald_gmax):
        cellvolume = jnp.linalg.det(self.latvec)
        recvec = jnp.linalg.inv(self.latvec).T

        # Determine alpha
        smallestheight = jnp.amin(1 / jnp.linalg.norm(recvec, axis=1))
        self.alpha = 5.0 / smallestheight
        logger.info(f"Setting Ewald alpha to {self.alpha.item()}")

        # Determine G points to include in reciprocal Ewald sum
        gptsXpos = jnp.meshgrid(
            jnp.arange(1, ewald_gmax + 1),
            *[jnp.arange(-ewald_gmax, ewald_gmax + 1)] * 2,
            indexing="ij",
        )
        zero = jnp.asarray([0])
        gptsX0Ypos = jnp.meshgrid(
            zero,
            jnp.arange(1, ewald_gmax + 1),
            jnp.arange(-ewald_gmax, ewald_gmax + 1),
            indexing="ij",
        )
        gptsX0Y0Zpos = jnp.meshgrid(
            zero, zero, jnp.arange(1, ewald_gmax + 1), indexing="ij"
        )
        gs = zip(
            *[
                select_big(x, cellvolume, recvec, self.alpha)
                for x in (gptsXpos, gptsX0Ypos, gptsX0Y0Zpos)
            ]
        )
        self.gpoints, self.gweight = [jnp.concatenate(x, axis=0) for x in gs]
        self.set_ewald_constants(cellvolume)

    def set_ewald_constants(self, cellvolume):
        self.i_sum = jnp.sum(self.atom_charges)
        ii_sum2 = jnp.sum(self.atom_charges**2)
        ii_sum = (self.i_sum**2 - ii_sum2) / 2

        self.ijconst = -jnp.pi / (cellvolume * self.alpha**2)
        self.squareconst = -self.alpha / jnp.sqrt(jnp.pi) + self.ijconst / 2

        self.ii_const = ii_sum * self.ijconst + ii_sum2 * self.squareconst
        self.e_single_test = -self.i_sum * self.ijconst + self.squareconst

        GdotR = jnp.dot(self.gpoints, jnp.asarray(self.atom_coords.T))
        self.ion_exp = jnp.dot(jnp.exp(1j * GdotR), self.atom_charges)

        # NOTE: The following is added in stress implementation
        gpoints = self.gpoints
        gweight = self.gweight[..., None, None]
        gpoints_outer = jnp.einsum("ij,ik->ijk", gpoints, gpoints)
        gsquared = jnp.sum(gpoints**2, axis=-1)[..., None, None]
        self.gweight_stress = gweight * (
            2 * gpoints_outer * (1 / gsquared + 1 / (4 * self.alpha**2)) - jnp.eye(3)
        )
        self.stress_ion_ion = self.stress_ion()

    def _real_cij(self, dists):
        r = dists[:, :, None, :] + self.lattice_displacements
        r = jnp.linalg.norm(r, axis=-1)
        cij = jnp.sum(jax.lax.erfc(self.alpha * r) / r, axis=-1)
        return cij

    # NOTE: The following methods are added by stress implementation
    def _real_deriv_r(self, r):
        a = self.alpha
        return (
            -2 * a * jnp.exp(-(a**2) * r**2) / (jnp.sqrt(jnp.pi) * r)
            - jax.lax.erfc(a * r) / r**2
        )

    def _real_stress_mat(self, dr, charge_ij=1):
        # (nlat, nParticleA, nParticleB, ndim)
        dr = dr[None, ...] + self.lattice_displacements[:, None, None, :]
        r = jnp.linalg.norm(dr, axis=-1)
        return jnp.transpose(  # Transpose makes triu easier
            jnp.einsum("...ij,...ik->...ijk", dr, dr)  # (nl, nA, nB, nd, nd)
            * (charge_ij * self._real_deriv_r(r) / r)[..., None, None],
            (0, 3, 4, 1, 2),
        )  # (nl, nd, nd, nA, nB)

    def stress_ion_real(self):
        if len(self.atom_charges) == 1:
            return 0
        ion_distances = self.dist.dist_matrix(self.atom_coords.ravel())
        charge_ij = self.atom_charges[..., None] * self.atom_charges[None, ...]
        stress_real_all_ion = self._real_stress_mat(ion_distances, charge_ij)
        return jnp.sum(jnp.triu(stress_real_all_ion, k=1), axis=(0, 3, 4))

    def stress_ion_recip(self):
        return jnp.einsum("ijk,i->jk", self.gweight_stress, jnp.abs(self.ion_exp) ** 2)

    def stress_ion(self):
        # `ii_const` depends on `ijconst` which has $\frac{1}{V}$ dependence.
        # `squareconst = sth. + ijconst / 2`, which means it also depend on $V$.
        # Originally, we have this for Ewald summation for energy:
        #   `ii_const = (i_sum**2 - ii_sum2) / 2 * ijconst + ii_sum2 * squareconst`
        ii_const = -(self.i_sum**2) / 2 * self.ijconst * jnp.eye(3)
        return self.stress_ion_real() + self.stress_ion_recip() + ii_const

    def stress_electron_real(self, configs):
        nelec = self.nelec
        ei_distances = self.dist.dist_i(self.atom_coords.ravel(), configs)
        ei_real_separated = jnp.sum(
            self._real_stress_mat(ei_distances, -self.atom_charges[None, :]),
            axis=(0, 3, 4),
        )
        ee_real_separated = jnp.array(0.0)
        if nelec > 1:
            ee_distances = self.dist.dist_matrix(configs)
            ee_real_separated = jnp.sum(
                jnp.triu(self._real_stress_mat(ee_distances), k=1),
                axis=(0, 3, 4),
            )
        return ei_real_separated + ee_real_separated

    def stress_electron_recip(self, configs):
        e_GdotR = jnp.einsum("ik,jk->ij", configs.reshape(-1, 3), self.gpoints)
        e_sin = jnp.sin(e_GdotR).sum(axis=0)
        e_cos = jnp.cos(e_GdotR).sum(axis=0)
        ee_recip = jnp.einsum("i,ijk->jk", e_sin**2 + e_cos**2, self.gweight_stress)
        coscos_sinsin = -self.ion_exp.real * e_cos - self.ion_exp.imag * e_sin
        ei_recip = 2 * jnp.einsum("i,ijk->jk", coscos_sinsin, self.gweight_stress)
        return ee_recip + ei_recip

    def stress_electron(self, configs):
        return self.stress_electron_real(configs) + self.stress_electron_recip(configs)

    def stress(self, configs):
        ne = self.nelec
        # Originally, `ei_const = -ne * i_sum * ijconst`, and
        #   `ee_const = ne * (ne - 1) / 2 * ijconst + ne * squareconst`
        # Considering the volume contribution from `ijconst` and `squareconst`:
        ei_const = ne * self.i_sum * self.ijconst * jnp.eye(3)
        ee_const = -(ne**2) / 2 * self.ijconst * jnp.eye(3)
        return self.stress_ion_ion + self.stress_electron(configs) + ee_const + ei_const


def select_big(gpts, cellvolume, recvec, alpha):
    gpoints = jnp.einsum("j...,jk->...k", gpts, recvec) * 2 * jnp.pi
    gsquared = jnp.einsum("...k,...k->...", gpoints, gpoints)
    gweight = 4 * jnp.pi * jnp.exp(-gsquared / (4 * alpha**2))
    gweight /= cellvolume * gsquared
    bigweight = gweight > 1e-12
    return gpoints[bigweight], gweight[bigweight]


DEFAULT = StressEstimator
