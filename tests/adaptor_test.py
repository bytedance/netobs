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

from pathlib import Path

import jax
import pytest
from jax import numpy as jnp

from netobs.adaptors.simple_hydrogen import SimpleHydrogen
from netobs.helpers.importer import import_module_or_file

try:
    from netobs.adaptors.ferminet_vmc import FERMINET_VERSION

    HAS_FERMINET_MAIN = FERMINET_VERSION == "main"
except ImportError:
    HAS_FERMINET_MAIN = False


try:
    from netobs.adaptors.deepsolid_vmc import DeepSolidVMCAdaptor

    HAS_DEEPSOLID = True
except ImportError:
    HAS_DEEPSOLID = False


@pytest.mark.skipif(
    not jax.__version__.startswith("0.4."),
    reason="Different JAX version has different behaviors. Please use JAX 0.4.x",
)
class TestHydrogen:
    adaptor = SimpleHydrogen(0.999, [])

    def test_wf(self, snapshot):
        self.adaptor.quality = 1
        params, data, system, _ = self.adaptor.restore("1")
        data = data[0, 0]  # 1 data point, remove leading axis
        assert self.adaptor.call_network(params, data, system) == snapshot

    def test_potential(self, snapshot):
        params, data, system, _ = self.adaptor.restore("1")
        data = data[0, 0]
        assert (
            self.adaptor.call_local_potential_energy(params, None, data, system)
            == snapshot
        )

    def test_kinetic(self, snapshot):
        params, data, system, _ = self.adaptor.restore("1")
        data = data[0, 0]
        assert (
            self.adaptor.call_local_kinetic_energy(params, None, data, system)
            == snapshot
        )

    def test_walk(self, snapshot):
        params, data, system, aux_data = self.adaptor.restore("5")
        key = jax.random.PRNGKey(42)
        key = jax.random.split(key, 1)
        mcmc_step = self.adaptor.make_walking_step(1, system)
        assert mcmc_step(key, params, data, aux_data)[0] == snapshot


@pytest.mark.skipif(
    not HAS_FERMINET_MAIN,
    reason="FermiNet main is required to test it",
)
class TestFermiNet:
    @pytest.fixture(autouse=True)
    def _setup_adaptor(self):
        from ferminet.base_config import default
        from ferminet.utils import system

        from netobs.adaptors.ferminet_vmc import FermiNetMainAdaptor

        cfg = default()
        cfg.system.molecule = [
            system.Atom(symbol="H", coords=(0, 0, 0)),
        ]
        cfg.system.electrons = (1, 0)
        cfg.network.ferminet.hidden_dims = ((32, 4), (32, 4))
        self.adaptor = FermiNetMainAdaptor(cfg, [])
        self.net_retore = str(Path(__file__).parent / "data" / "H_atom.npz")
        params, data, system, aux_data = self.adaptor.restore(self.net_retore)
        self.params = jax.tree_util.tree_map(lambda x: x[0], params)
        self.data = data[0, 0]
        self.system = system
        self.aux_data = aux_data

    def test_wf(self, snapshot):
        logpsi = self.adaptor.call_network(self.params, self.data, self.system)
        assert logpsi == snapshot

    def test_potential(self, snapshot):
        potential = self.adaptor.call_local_potential_energy(
            self.params, None, self.data, self.system
        )
        assert potential == snapshot

    def test_kinetic(self, snapshot):
        kinetic = self.adaptor.call_local_kinetic_energy(
            self.params, None, self.data, self.system
        )
        assert kinetic == snapshot

    def test_walk(self, snapshot):
        # This one is pmapped
        params, data, system, aux_data = self.adaptor.restore(self.net_retore)
        key = jax.random.PRNGKey(42)
        key = jax.random.split(key, 1)
        mcmc_step = self.adaptor.make_walking_step(1, system)
        assert mcmc_step(key, params, data, aux_data)[0] == snapshot


@pytest.mark.skipif(
    not HAS_DEEPSOLID,
    reason="DeepSolid is required to test it",
)
class TestDeepSolid:
    @pytest.fixture(autouse=True)
    def _setup_adaptor(self):
        data_dir = Path(__file__).parent / "data"
        cfg = import_module_or_file(str(data_dir / "H_chain.py")).get_config()
        self.adaptor = DeepSolidVMCAdaptor(cfg, [])
        self.net_retore = str(data_dir / "H_chain.npz")
        params, data, system, aux_data = self.adaptor.restore(self.net_retore)
        self.params = jax.tree_util.tree_map(lambda x: x[0], params)
        self.data = data[0, 0]
        self.system = system
        self.aux_data = aux_data

    def test_wf(self, snapshot):
        logpsi = self.adaptor.call_network(self.params, self.data, self.system)
        assert logpsi == snapshot

    def test_potential(self, snapshot):
        potential = self.adaptor.call_local_potential_energy(
            self.params, None, self.data, self.system
        )
        assert potential == snapshot

    def test_kinetic(self, snapshot):
        kinetic = self.adaptor.call_local_kinetic_energy(
            self.params, None, self.data, self.system
        )
        assert kinetic == snapshot

    def test_walk(self, snapshot):
        # This one is pmapped
        params, data, system, aux_data = self.adaptor.restore(self.net_retore)
        key = jax.random.PRNGKey(42)
        key = jax.random.split(key, 1)
        mcmc_step = self.adaptor.make_walking_step(1, system)
        data = mcmc_step(key, params, data, aux_data)[0]
        assert jnp.array(data) == snapshot  # ShardedDeviceArray -> DeviceArray
