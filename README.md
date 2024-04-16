# NetObs

Calculate observables from neural network&ndash;based VMC (NN-VMC).

Currently, NetObs has built-in suport for [FermiNet](https://github.com/google-deepmind/ferminet) (molecular systems) and [DeepSolid](https://github.com/bytedance/DeepSolid) (solid systems). We are planning to add support for more neural network VMC frameworks, and it's also very easy to write your own [adaptor](#adaptor) to make NetObs support your network!

The currently supported observables include energy, electron density, interatomic force, stress, [etc.](netobs/observables) More observables are on the way, and it is also very easy to write your own!

## Installation

Clone this repository and run an editable installation:

```shell
pip install -e .
```

Please don't forget to install the NN-VMC code and the corresponding version of JAX beforehand.
NetObs is tested with [`google-deepmind/ferminet@e94435b`](https://github.com/google-deepmind/ferminet/commit/e94435b15ca73a6c5404cdd399916290990e340b)+`jax@0.4.24` and [`bytedance/DeepSolid`](https://github.com/bytedance/DeepSolid/)+`jax@0.2.26`.

## Command Line Example

NetObs comes with a friendly command-line app, making it easy for a quick try. An example command would be:

```shell
netobs @ferminet_vmc tests/data/H_atom.py @force:SWCT --with steps=20 --net-restore tests/data/H_atom.npz
```

A short explanation of the arguments:

- `@ferminet_vmc` specifies the [`Adaptor`](#adaptor). The `@` sign is a shortcut to access built-in `Adaptor`s, `Estimator`s, etc. `@ferminet_vmc` stands for the `DEFAULT` object in module `netobs.adaptors.ferminet_vmc`. Another example would be `my_module:MyAdaptor`.
- `tests/data/H_atoms.py` is a Python module or file that contains a `get_config` function. If you have a custom `Adaptor`, it can be anything your `Adaptor` recognizes.
- `@force:SWCT` specifies the [`Estimator`](#estimator). `@force:SWCT` means the `SWCT` class in the `netobs.observables.force` module.
- `--with steps=20` specifies the [options for NetObs](#options).
- `--net-restore ...` tells the `Adaptor` where to restore the network checkpoint.

<details>
<summary>Example output</summary>

```
2024-02-29 20:34:29,150 INFO netobs ferminet_vmc.py:54] Assuming running with FermiNet on main.
2024-02-29 20:34:29,333 INFO netobs force.py:319] Using molecular version of SWCT
2024-02-29 20:34:29,548 INFO netobs evaluate.py:102] Start burning in 100 steps
2024-02-29 20:34:30,522 INFO netobs evaluate.py:115] Starting 20 evaluation steps
2024-02-29 20:34:40,112 INFO netobs evaluate.py:148] Time per step: 0.01s
energy: -0.50042176 ± 0.00027392566
force
[[-7.4391162e-12 -1.3514609e-09 -1.4826655e-09]]
std force
[[4.9552114e-09 4.2379065e-09 3.2048899e-09]]
force_no_warp
[[-0.00097416 -0.00091072  0.00125167]]
std force_no_warp
[[0.00146565 0.00108437 0.00165038]]
```

> [!NOTE]
> The standard error (std) here are rough estimates which does NOT take autocorrelation into consideration due to performance conserns. You are STRONGLY encouraged to analyze the data using your own code.

</details>

Another example with DeepSolid:

```shell
netobs @deepsolid_vmc tests/data/H_chain.py @energy --with steps=20 --net-restore tests/data/H_chain.npz
```

> [!NOTE]
> If you want to use the SWCT force estimator with DeepSolid, it is highly recommended to use the [`tri` feature](https://github.com/bytedance/DeepSolid/pull/3).

## Core Concepts

### Adaptor

`Adaptor` is an abstract layer over the network that exposes a uniform API. For example, it tells NetObs how to restore from a checkpoint, evaluates the network and Hamiltonian, etc.

Currently, we have built-in support for FermiNet (molecules) and DeepSolid (solid systems). You can find their implementation in [netobs/adaptors](netobs/adaptors). It is also very easy to write your own adaptors!

### Estimator

You can regard estimators as different ways to estimate a physical quantity given a set of Monte Carlo samples. You can have multiple estimators for an observable, but usually a basic one is enough.

In the code, an `Estimator` tells us what type of observable it targets at, the expression of the estimator given a set of Monte Carlo samples, how to combine the results from different steps (simply averaging or more actions are required), etc.

An `Estimator` can work only in the molecular case, or it can be implemented to support molecules and solids at the same time. Check out the `SWCT` `Estimator` in [force.py](netobs/observables/force.py) for more.

You can find the implementation of built-in estimators in [netobs/observables](netobs/observables). And it is easy to create your own estimator!

## Options

- `steps` (int, default to `10`): Steps to run inference.
- `mcmc_steps` (int, default to `10`): Steps to run MCMC.
- `mcmc_burn_in` (int, default to `100`): Burn-in steps for MCMC.
- `random_seed` (int, default to current time): The random seed for the Monte Carlo simulation.
- `log_interval` (int, default to `10`): Time interval in seconds between logs.
- `save_interval` (int, default to `600`): Time interval in seconds between saves.
- `estimator` (dict): Options for the estimators.
- `observable` (dict): Options for the observable.

## Integrate in Your Code

You need to pass your `Adaptor`, `Estimator` class, and evaluate options to [`netobs.evaluate.evaluate_observable`](netobs/evaluate.py). For example,

```python
from netobs.helpers.importer import import_module_or_file
from netobs.observables.force import SWCT

cfg = import_module_or_file("H_chain.py").get_config()
adaptor = DeepSolidVMCAdaptor(cfg, [])
options = NetObsOptions(steps=20)
digest, all_values, state = evaluate_observable(adaptor, SWCT, options=options)
```

For more details, check out [this test](tests/numerical_test.py) for how to start an evaluation, and check out [this test](tests/adaptor_test.py) for how to use different adaptors in your code.

## Contributing

Contributions are welcomed and highly appreciated! We are open to new network `Adaptor`s and new `Estimator`s! For a detailed contribution guide, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you are using NetObs in your works, please consider [citing our papers](CITATIONS.bib).
