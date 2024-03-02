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

import importlib.util
import sys
import uuid
from importlib import import_module
from typing import Any


def import_module_or_file(module_name: str) -> Any:
    """Import a python module or a python file.

    Args:
        module_name: the name of the module or file.
            If it ends with ".py", it will be considered as a file, otherwise module.

    Returns:
        Contents of the module.

    Raises:
        OSError: Python ifle not found.
    """
    if module_name.endswith(".py"):
        # generate unique module name
        module_id = "netobs_" + str(uuid.uuid4()).replace("-", "_")
        # `imp` is deprecated. Using `importlib` way
        spec = importlib.util.spec_from_file_location(module_id, module_name)
        if spec is None or spec.loader is None:
            raise OSError(f"Failed to load {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_id] = module
        spec.loader.exec_module(module)
    else:
        module = import_module(module_name)
    return module
