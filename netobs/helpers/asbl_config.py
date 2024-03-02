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

import sys
from typing import TYPE_CHECKING

from absl import flags
from ml_collections.config_flags import DEFINE_config_dict

from netobs.helpers.importer import import_module_or_file

if TYPE_CHECKING:
    from ml_collections import ConfigDict


def absl_config(config: str, args: list[str]) -> ConfigDict:
    """Resolve `ConfigDict` with abseil flags.

    Args:
        config: "module:options"-style config module spec.
        args: "--config.a.b=c"-style command line arguments.

    Returns:
        `ConfigDict` object.
    """
    config_name, *config_args = config.split(":", maxsplit=1)
    config_module = import_module_or_file(config_name)
    if config_args:
        cfg = config_module.get_config(config_args[0])
    else:
        cfg = config_module.get_config()

    flag_holder = DEFINE_config_dict("config", cfg)
    # Abseil won't be happy with an empty argv
    flags.FLAGS([sys.argv[0]] + args)
    return flag_holder.value
