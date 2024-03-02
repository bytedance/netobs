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

# Original files: https://github.com/WagnerGroup/pyqmc/blob/master/pyqmc/distance.py
# Copyright (c) 2019-2024 The PyQMC Developers
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp

from netobs.systems import System


class SolidSystem(System):
    charges: jnp.ndarray
    "(natom,) Nuclear charges of the atoms."

    latvec: jnp.ndarray
    "(3, 3) Lattice vectors."

    spins: tuple[int, int]


def enforce_pbc(latvec: jnp.ndarray, x: jnp.ndarray):
    """Enforces periodic boundary conditions on a set of configs.

    Args:
        latvec: lattice vectors. Shape (3, 3)
        x: attempted new electron coordinates. Shape (nelec * 3)

    Returns:
        Final electron coordinates with PBCs imposed. Shape (nelec * 3)
    """
    # Writes epos in terms of (lattice vecs) fractional coordinates
    dim = latvec.shape[-1]
    electrons = x.reshape(-1, dim)
    recpvecs = jnp.linalg.inv(latvec)
    e_lvecs_coord = jnp.einsum("ij,jk->ik", electrons, recpvecs)

    _, mod = jnp.divmod(e_lvecs_coord, 1)
    return jnp.matmul(mod, latvec).ravel()


batch_enforce_pbc = jax.vmap(enforce_pbc, in_axes=(None, 0), out_axes=0)


def cell_vol(system: SolidSystem):
    """Volume of the cell."""
    return abs(jnp.linalg.det(system["latvec"]))


def recvec(system: SolidSystem, norm_to=2 * jnp.pi):
    """Reciprocal lattice of the cell."""
    return norm_to * jnp.linalg.inv(jnp.transpose(system["latvec"]))


class MinimalImageDistance:
    def __init__(self, latvec):
        latvec = jnp.asarray(latvec)
        ortho_tol = 1e-10
        diagonal = jnp.all(jnp.abs(latvec - jnp.diag(jnp.diagonal(latvec))) < ortho_tol)
        if diagonal:
            self.dist_i = self.diagonal_dist_i
        else:
            orthogonal = (
                jnp.dot(latvec[0], latvec[1]) < ortho_tol
                and jnp.dot(latvec[1], latvec[2]) < ortho_tol
                and jnp.dot(latvec[2], latvec[0]) < ortho_tol
            )
            if orthogonal:
                self.dist_i = self.orthogonal_dist_i
            else:
                self.dist_i = self.general_dist_i
        self._latvec = latvec
        self._invvec = jnp.linalg.inv(latvec)
        self.dim = self._latvec.shape[-1]
        # list of all 26 neighboring cells
        mesh_grid = jnp.meshgrid(*[jnp.array([0, 1, 2]) for _ in range(3)])
        self.point_list = jnp.stack([m.ravel() for m in mesh_grid], axis=0).T - 1
        self.shifts = self.point_list @ self._latvec

    def general_dist_i(self, atoms, electrons):
        atoms = atoms.reshape([1, -1, self.dim])
        electrons = electrons.reshape([-1, 1, self.dim])
        r_ae = electrons - atoms
        shifts = self.shifts.reshape((-1, *[1] * (len(r_ae.shape) - 1), 3))
        r_ae_all = r_ae[None] + shifts
        dists = jnp.linalg.norm(r_ae_all, axis=-1)
        mininds = jnp.argmin(dists, axis=0)
        inds = jnp.meshgrid(*[jnp.arange(n) for n in mininds.shape], indexing="ij")
        return r_ae_all[(mininds, *inds)]

    def orthogonal_dist_i(self, atoms, electrons):
        atoms = atoms.reshape([1, -1, self.dim]).real
        electrons = electrons.reshape([-1, 1, self.dim]).real
        r_ae = electrons - atoms
        frac_disps = jnp.einsum("...ij,jk->...ik", r_ae, self._invvec)
        replace_frac_disps = (frac_disps + 0.5) % 1 - 0.5
        return jnp.einsum("...ij,jk->...ik", replace_frac_disps, self._latvec)

    def diagonal_dist_i(self, atoms, electrons):
        atoms = atoms.reshape([1, -1, self.dim]).real
        electrons = electrons.reshape([-1, 1, self.dim]).real
        r_ae = electrons - atoms
        latvec_diag = jnp.diagonal(self._latvec)
        replace_r_ae = (r_ae + latvec_diag / 2) % latvec_diag - latvec_diag / 2
        return replace_r_ae

    def dist_matrix(self, configs):
        dist_mat = self.dist_i(configs, configs)
        # NOTE: This is patched to avoid NaN.
        # Diagonals are not used, so there is no need to put them to zero
        dist_mat += jnp.eye(dist_mat.shape[0])[..., None]
        return dist_mat

    def neighboring_r_ae(
        self, electrons: jnp.ndarray, atoms: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # shape (ncell, natom, 3)
        extended_atoms = atoms + self.shifts[:, None, :]
        # shape (ncell, nelec, natom, 3)
        ae = jnp.reshape(electrons, [-1, 1, 3]) - extended_atoms[:, None, ...]
        r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
        return ae, r_ae

    def closest_r_ae(
        self, electrons: jnp.ndarray, atoms: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ae = self.dist_i(atoms.ravel(), electrons)
        r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
        return ae, r_ae
