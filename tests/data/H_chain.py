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

import numpy as np
from DeepSolid import base_config, supercell
from pyscf.pbc import gto


def get_config():
    S = np.eye(3)
    cfg = base_config.default()

    cfg.network.detnet.hidden_dims = ((32, 4), (32, 4))
    cfg.network.detnet.distance_type = "tri"

    cell = gto.Cell()
    cell.atom = """
    H 2.0 0.0 0.0
    H 0.0 0.0 0.0
    """
    cell.basis = "ccpvdz"
    cell.a = np.array([[4, 0, 0], [0, 100, 0], [0, 0, 100]])
    cell.unit = "B"
    cell.spin = 0
    cell.verbose = 5
    cell.exp_to_discard = 0.1
    cell.build()
    simulation_cell = supercell.get_supercell(cell, S)
    simulation_cell.hf_type = "uhf"
    cfg.system.pyscf_cell = simulation_cell

    return cfg
