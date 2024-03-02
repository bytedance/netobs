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

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NetObsOptions:
    steps: int = 10
    "Steps to run inference."

    mcmc_steps: int = 10
    "Steps to run MCMC."

    mcmc_burn_in: int = 100
    "Burn-in steps for MCMC."

    random_seed: int = field(default_factory=lambda: int(1e6 * time.time()))
    """The random seed for the Monte Carlo simulation.
    Defaults to None, which means use current time as random seed.
    """

    log_interval: int = 10
    "Time interval in seconds between logs."

    save_interval: int = 600
    "Time interval in seconds between saves."

    network_restore: Any = None
    """The restore option for the network adaptor.
    Defaults to None, which means the network adaptor has a fixed and known way to
        restore, or it has other ways to specify how to restore.
    `network_restore` will be passed as is `NetworkAdaptor.restore`, so make sure the
        adaptor understands it.
    """

    estimator: dict = field(default_factory=dict)
    "Options for the estimators."

    observable: dict = field(default_factory=dict)
    "Options for the observable."
