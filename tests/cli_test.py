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

from netobs.cli import make_cli

try:
    from netobs.adaptors.ferminet_vmc import FERMINET_VERSION

    HAS_FERMINET_MAIN = FERMINET_VERSION == "main"
except ImportError:
    HAS_FERMINET_MAIN = False


@pytest.mark.skipif(
    not HAS_FERMINET_MAIN,
    reason="FermiNet main is required to test it",
)
def test_cli():
    cli = make_cli()
    cli(
        [
            "@ferminet_vmc",
            "tests/data/H_atom.py",
            "@force:Antithetic",
            "--with",
            "steps=1",
            "--net-restore",
            "tests/data/H_atom.npz",
        ]
    )
