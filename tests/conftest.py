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


@pytest.fixture(autouse=True)
def consistent_jax_print():
    def float_round(x):
        # Simply rounding will have "0.0" vs "-0.0" issue
        return "0.0" if abs(x) < 1e-7 else str(round(x, 7))

    jnp.set_printoptions(
        suppress=True,
        linewidth=100,
        formatter={"float_kind": float_round},
    )
