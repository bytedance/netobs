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

r"""Example CLI application.

```bash
netobs @ferminet_vmc li2:5.051 @force:SWCT \
  --with steps=100 mcmc_burn_in=30 estimator.chunks=4 \
  --ckpt-mgr @:SavingCheckpointManager --ckpt Li2/swct \
  --net-restore Li2/qmcjax_ckpt_099999.npz \
  --config.network.hidden_dims '(256, 32)'
```
"""

from __future__ import annotations

import argparse
from typing import Any, TypedDict

from jax.config import config as jax_config
from omegaconf import OmegaConf
from typing_extensions import Self

from netobs.adaptors import NetworkAdaptor
from netobs.checkpoint import CheckpointManager, SavingCheckpointManager
from netobs.evaluate import evaluate_observable
from netobs.helpers.importer import import_module_or_file
from netobs.observables import Estimator
from netobs.options import NetObsOptions


class Expansion(TypedDict):
    estimator: dict[str, str]
    checkpoint: dict[str, str]
    adaptor: dict[str, str]


DEFAULT_EXPANSION: Expansion = {
    "estimator": {"@": "netobs.observables."},
    "checkpoint": {"@": "netobs.checkpoint"},
    "adaptor": {"@": "netobs.adaptors."},
}


class Arguments(argparse.Namespace):
    adaptor: str
    config: str
    estimator: str
    x64: bool
    ckpt_mgr: str
    ckpt: str
    ckpt_from: str
    ckpt_to: str
    net_restore: str
    opt: list[str]

    @property
    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("adaptor", type=str, help="Adaptor module and name")
        parser.add_argument("config", type=str, help="Config name")
        parser.add_argument("estimator", type=str, help="Estimator name and options")
        parser.add_argument("--x64", action="store_true", help="Enable x64 precision")
        parser.add_argument("--ckpt-mgr", default=None, help="Checkpoint manager")
        parser.add_argument("--ckpt", default=None, help="Checkpoint dir")
        parser.add_argument("--ckpt-from", default=None, help="Checkpoint to restore")
        parser.add_argument("--ckpt-to", default=None, help="Checkpoint to save")
        parser.add_argument("--net-restore", default=None, help="Network checkpoint")
        parser.add_argument("--with", dest="opt", nargs="*", help="NetOBS options")
        return parser

    def parse(self, argv: list[str] | None = None) -> tuple[Self, list[str]]:
        args, rest_args = self.parser.parse_known_args(argv, namespace=self)
        if args.ckpt_from is None and args.ckpt_to is None:
            args.ckpt_from = args.ckpt_to = args.ckpt
        return args, rest_args


def make_cli(
    arg_type: type[Arguments] = Arguments,
    # checkpoint_manager: type[CheckpointManager] | None = None,
    expansion: Expansion = DEFAULT_EXPANSION,
):
    def cli(argv: list[str] | None = None) -> None:
        args, rest_args = arg_type().parse(argv)
        if args.x64:
            jax_config.update("jax_enable_x64", True)
        adaptor_type: type[NetworkAdaptor] = resolve_object_option(
            args.adaptor, expansion["adaptor"]
        )
        estimator_type: type[Estimator] = resolve_object_option(
            args.estimator, expansion["estimator"]
        )
        checkpoint_mgr_type: type[CheckpointManager]
        if args.ckpt_mgr is None:
            checkpoint_mgr_type = SavingCheckpointManager
        else:
            checkpoint_mgr_type = resolve_object_option(
                args.ckpt_mgr, expansion["checkpoint"]
            )

        force_options = OmegaConf.structured(
            NetObsOptions(network_restore=args.net_restore)
        )
        force_options = OmegaConf.merge(force_options, OmegaConf.from_dotlist(args.opt))
        evaluate_observable(
            adaptor_type(args.config, rest_args),
            estimator_type,
            options=NetObsOptions(**force_options),
            checkpoint_mgr=checkpoint_mgr_type(args.ckpt_from, args.ckpt_to),
        )

    return cli


def resolve_object_option(name: str, expansion: dict[str, str]) -> Any:
    """Resolve object and option from "module:name:option" natation.

    Supported notations:
    - "module": resolve default object
    - "module:name": resolve `module.name`
    """
    for notation, replace in expansion.items():
        if name.startswith(notation):
            name = replace + name[len(notation) :]

    colon_count = name.count(":")
    if colon_count == 0:
        module, obj_name = name, "DEFAULT"
    elif colon_count == 1:
        module, obj_name = name.split(":")
    else:
        raise ValueError(f"Too many colons in '{name}'")

    obj = getattr(import_module_or_file(module), obj_name)
    if obj is None:
        raise ValueError("Estimator not found")
    return obj
