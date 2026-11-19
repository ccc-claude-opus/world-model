```sh
(cccuser) cccuser@cccimacdeiMac world-model % cd /Users/Shared/ccc/project/world-model/external/diamond && .venv/bin/python src/play.py --pretrained --size 448 --no-header
/Users/Shared/ccc/project/world-model/external/diamond/src/play.py:20: UserWarning: register_new_resolver() is deprecated and will be removed in a future release.
Use register_resolver() instead.
See https://github.com/hydra-ecosystem/omegaconf/issues/426 for migration instructions.

  OmegaConf.register_new_resolver("eval", eval)
/Users/Shared/ccc/project/world-model/external/diamond/.venv/lib/python3.10/site-packages/hydra/core/utils.py:212: UserWarning: register_new_resolver() is deprecated and will be removed in a future release.
Use register_resolver() instead.
See https://github.com/hydra-ecosystem/omegaconf/issues/426 for migration instructions.

  OmegaConf.register_new_resolver(
/Users/Shared/ccc/project/world-model/external/diamond/.venv/lib/python3.10/site-packages/hydra/core/utils.py:218: UserWarning: register_new_resolver() is deprecated and will be removed in a future release.
Use register_resolver() instead.
See https://github.com/hydra-ecosystem/omegaconf/issues/426 for migration instructions.

  OmegaConf.register_new_resolver(
/Users/Shared/ccc/project/world-model/external/diamond/.venv/lib/python3.10/site-packages/hydra/core/utils.py:230: UserWarning: register_new_resolver() is deprecated and will be removed in a future release.
Use register_resolver() instead.
See https://github.com/hydra-ecosystem/omegaconf/issues/426 for migration instructions.

  OmegaConf.register_new_resolver(
/Users/Shared/ccc/project/world-model/external/diamond/.venv/lib/python3.10/site-packages/hydra/_internal/config_loader_impl.py:291: FutureWarning: Implicit conversion from Version to str at 'hydra.runtime.version_base' during assignment is deprecated; pass the declared type or use OmegaConf.update() for an existing path.
  cfg.hydra.runtime.version_base = version.getbase()
 0: Alien
 1: Amidar
 2: Assault
 3: Asterix
 4: BankHeist
 5: BattleZone
 6: Boxing
 7: Breakout
 8: ChopperCommand
 9: CrazyClimber
10: DemonAttack
11: Freeway
12: Frostbite
13: Gopher
14: Hero
15: Jamesbond
16: Kangaroo
17: Krull
18: KungFuMaster
19: MsPacman
20: Pong
21: PrivateEye
22: Qbert
23: RoadRunner
24: Seaquest
25: UpNDown

Enter a number: 7
A.L.E: Arcade Learning Environment (version 0.9.0+750d7f9)
[Powered by Stella]
/Users/Shared/ccc/project/world-model/external/diamond/src/play.py:20: UserWarning: register_new_resolver() is deprecated and will be removed in a future release.
Use register_resolver() instead.
See https://github.com/hydra-ecosystem/omegaconf/issues/426 for migration instructions.

  OmegaConf.register_new_resolver("eval", eval)
A.L.E: Arcade Learning Environment (version 0.9.0+750d7f9)
[Powered by Stella]
/Users/Shared/ccc/project/world-model/external/diamond/src/play.py:20: UserWarning: register_new_resolver() is deprecated and will be removed in a future release.
Use register_resolver() instead.
See https://github.com/hydra-ecosystem/omegaconf/issues/426 for migration instructions.

  OmegaConf.register_new_resolver("eval", eval)
A.L.E: Arcade Learning Environment (version 0.9.0+750d7f9)
[Powered by Stella]
/Users/Shared/ccc/project/world-model/external/diamond/src/agent.py:55: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  sd = torch.load(Path(path_to_ckpt), map_location=self.device)
Collecting 1000 steps in real environment for world model initialization.
Collect Breakout_1000: 100%|████████████████████████| 1000/1000 [00:04<00:00, 206.95steps/s]
/Users/Shared/ccc/project/world-model/external/diamond/.venv/lib/python3.10/site-packages/torch/utils/data/sampler.py:65: UserWarning: `data_source` argument is not used and will be removed in 2.2.0.You may still have custom implementation that utilizes it.
  warnings.warn("`data_source` argument is not used and will be removed in 2.2.0."

Controls (general):

⏎ : reset env
. : pause/unpause
e : step-by-step (when paused)

Controls (play mode):

m : controller (policy/human)
↑ : imagination horizon (+1)
↓ : imagination horizon (-1)
→ : next environment (wm → test → train)
← : prev environment (wm ← test ← train)

Environment actions:

⎵ : fire
d : right
a : left


/Users/Shared/ccc/project/world-model/external/diamond/src/data/episode.py:41: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  for k, v in torch.load(Path(path), map_location=map_location).items()
2026-11-13 13:36:55.310 python[8660:13851437] TSM AdjustCapsLockLEDForKeyTransitionHandling - _ISSetPhysicalKeyboardCapsLockLED Inhibit
> Environment: train
```