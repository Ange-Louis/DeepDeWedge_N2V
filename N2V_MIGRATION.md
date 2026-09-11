# Migrating DeepDeWedge from Noise2Noise to Noise2Void / Noise2Void2

This document explains the motivation and design decisions behind DeepDeWedge_N2V's switch from Noise2Noise (N2N) training to Noise2Void (N2V) / Noise2Void2 (N2V2) training, using the [CAREamics](https://github.com/CAREamics/careamics) implementation. It is intended for anyone modifying this codebase, not for end users — see the main [README](README.md) for installation and usage instructions.

## 1. Motivation

The original DeepDeWedge trains its U-Net using **Noise2Noise** (Lehtinen et al., 2018): a pair of independently reconstructed tomograms of the same specimen (`tomo0`, `tomo1` — typically even/odd tilt series frames), sharing the same underlying signal but with independent noise realizations. One is used to construct the model input, the other serves as the training target.

This requires that two independent reconstructions be available. In some acquisition or reconstruction workflows, only a single usable tomogram exists, which makes N2N training inapplicable.

**Noise2Void** (Krull, Buchholz & Jug, *CVPR*, 2019) removes this requirement. Under the assumption that the signal is spatially correlated but the noise is voxel-wise independent given the signal, a network can be trained to predict a voxel's value from its surrounding context alone — without ever seeing that voxel's own value. In practice this is implemented via a *blind-spot masking* scheme: for each training patch, a small random subset of voxels is replaced with a value drawn from their local neighborhood ("manipulated"), and the training loss is computed only at those manipulated positions, comparing the network's prediction against the original (unmanipulated) value.

**Noise2Void2** (Höck, Buchholz, Brachmann, Jug & Freytag, *ECCV Workshops*, 2022) modifies the replacement strategy — using the **median** of the local neighborhood instead of a randomly sampled neighbor — to reduce checkerboard artifacts that can appear in vanilla N2V reconstructions. The N2V2 paper also proposes network architecture changes (BlurPool instead of MaxPool, non-residual U-Net, no uppermost skip connection); **this repository does not adopt those architecture changes** — only the median replacement strategy is used, via CAREamics's `strategy="median"` option. The 3D U-Net used for missing-wedge reconstruction is unchanged from the original DeepDeWedge.

## 2. Design decision: pure N2V (no more `tomo0`/`tomo1` pair)

The `tomo0`/`tomo1` pair is dropped entirely. A single tomogram is used both to construct the model input and, via blind-spot masking, the training target — removing the two-reconstruction requirement altogether, which was the primary motivation for this migration.

An important architectural note: the N2V/N2V2 blind-spot masking (real-space, per-voxel) is **orthogonal** to DeepDeWedge's existing missing-wedge simulation mechanism (Fourier-space, applied to whole frequency planes via `apply_fourier_mask_to_tomo`). The two are complementary, not substitutes — the missing-wedge mechanism, which trains the network to reconstruct the physically missing wedge under arbitrary orientations, is unchanged by this migration.

## 3. File-by-file changes

### `prepare_data.py` → `prepare_n2v_data.py`
- `tomo0_files`/`tomo1_files` → single `tomo_files`.
- Output structure simplified: `{fitting,val}_subtomos/subtomo/{i}.pt` (previously two subfolders, `subtomo0/` and `subtomo1/`).

### `refine_tomogram.py` → `refine_n2v_tomogram.py`
- `tomo0_files`/`tomo1_files` → single `tomo_files`; no more averaging of two independent refinements.
- **No blind-spot masking is applied at inference time.** This is intentional: blind-spot masking is a training-time trick to construct a supervision signal from a single volume; at inference, the goal is the best possible prediction on the real (unmodified) data.
- A missing-wedge Fourier mask (`mw_mask`, canonical orientation, unchanged from the original DeepDeWedge) is still applied to the loaded tomogram before inference. This is **not** a second, artificial wedge the way it is during training (§2): since the full tomogram is never rotated at inference time, `mw_mask`'s canonical orientation coincides with the tomogram's actual physical missing wedge. So this step doesn't remove any additional information — it simply re-imposes, as a clean zero in Fourier space, the one real missing wedge already present in the reconstruction, so its geometry matches exactly what the network was trained to fill in, rather than leaving whatever imperfect artifacts the raw reconstruction produced there.

### `subtomo_dataset.py`
- Loads a single `subtomo` per `__getitem__` instead of a `subtomo0`/`subtomo1` pair.
- `model_target` is now the sub-tomogram itself (rotated, still containing its real missing wedge) instead of `subtomo1`.
- Renamed key `"model_input"` → `"n2v_input"`: this is the volume with the artificial missing wedge added (unchanged mechanism), **before** any blind-spot masking — the blind-spot masking itself happens later, in `unet.py` (see below), because `N2VManipulate` operates on a full batch tensor `(B, C, (Z), Y, X)`, while `SubtomoDataset.__getitem__` produces one un-batched sample at a time.
- The double missing-wedge mechanism (`mw_mask`/`rot_mw_mask`) is unchanged — see §2.

### `normalization.py`
- Updated to call `prepare_data` with the new single-`tomo_files` signature.
- Renamed key `"model_input"` → `"n2v_input"`.
- **Design decision: normalization statistics are computed *before* blind-spot masking**, i.e. on `n2v_input` as produced by `SubtomoDataset`, not on the output of `N2VManipulate`. Rationale: `refine_n2v_tomogram.py` never applies blind-spot masking at inference (§ above), so normalization statistics should reflect the same data distribution the network sees at inference time, not a distribution slightly perturbed by masking. In practice this makes negligible numerical difference (`masked_pixel_percentage` is small, and replacement values are of the same order of magnitude as the values they replace), but it keeps train/inference statistics conceptually consistent.

### `unet.py` (`LitUnet3D`)
- New constructor parameter `n2v_manipulate_params` (a plain `dict`, converted internally to a `careamics.config.algorithms.n2v_manipulation.N2VManipulateConfig`) — kept as a raw dict rather than a Pydantic object so it serializes cleanly via `save_hyperparameters()` into `hparams.yaml`, consistent with how `unet_params`/`adam_params` are already handled.
- `N2VManipulate` (the CAREamics class that performs the actual blind-spot masking) is instantiated **lazily**, via a `n2v_manipulate` property, rather than eagerly in `__init__` or unconditionally in a Lightning hook. Two reasons:
  - It needs `self.device`, which is only reliable once Lightning has placed the module on its target device — not yet available in `__init__`.
  - It must be available not just during `trainer.fit(...)`, but also during a standalone `trainer.validate(...)` call (used once before fitting starts, to log an initial validation loss) — and Lightning hooks like `on_train_start` do **not** fire during a standalone `.validate()` call. A lazy property (instantiated on first access, cached via `hasattr`) works correctly in both contexts.
- `training_step`/`validation_step`: call `self.n2v_manipulate(batch["n2v_input"].unsqueeze(1))` to obtain `(masked, original, n2v_mask)` — the channel dimension is added/removed manually (`unsqueeze(1)`/`squeeze(1)`) since `N2VManipulate` expects `(B, C, (Z), Y, X)` while the rest of the pipeline works without an explicit channel dimension. `n2v_mask` is passed through to `masked_loss` (see below) — **this is essential**: without it, the loss is computed over the entire volume, including the ~95–99.8% of voxels that are identical between input and target (i.e., not manipulated), which would let a residual network satisfy most of the loss by trivially learning the identity.
- `update_subtomo_missing_wedges`: key renames only (`"model_input"` → `"n2v_input"`, `"subtomo0_file"` → `"subtomo_file"`), logic unchanged — this method operates on the missing-wedge mechanism, unrelated to N2V.

### `masked_loss.py`
- New required parameter `n2v_mask`. The Fourier-filtered residual (missing-wedge weighting logic unchanged) is now also multiplied, in real space, by `n2v_mask` before squaring and averaging, and normalized by the number of manipulated voxels rather than the total volume size — so the loss reflects only the blind-spot-manipulated voxels, as intended by the N2V/N2V2 training scheme.

### `fit_n2v_model.py`
- New CLI parameter `n2v_manipulate_params` (dict: `strategy` — `"uniform"` for N2V or `"median"` for N2V2 — plus `roi_size`, `masked_pixel_percentage`, `seed`), passed through to `LitUnet3D`.
- Output directory name fix (`val_subtomos/subtomo0` → `val_subtomos/subtomo`) to match `prepare_n2v_data.py`'s new structure.
- Seed reproducibility: the N2V manipulation seed is derived from the global `seed` argument rather than left to its own random default, so a single `--seed` controls the entire pipeline (weights initialization, rotations, *and* blind-spot masking) reproducibly:
  ```python
  used_seed = pl.seed_everything(seed, workers=True)
  n2v_manipulate_params["seed"] = used_seed + 1
  ```
  `pl.seed_everything` returns the effective seed even when `seed=None` was passed (in which case it generates one), which conveniently sidesteps `N2VManipulateConfig.seed`'s `Field(gt=0)` constraint. The `+1` offset is arbitrary; its only purpose is to avoid using the exact same seed value for two conceptually different sources of randomness.

## 4. Choosing between Noise2Void and Noise2Void2

Entirely controlled by `n2v_manipulate_params["strategy"]`:
- `"uniform"` (default) → Noise2Void: manipulated voxels are replaced with a randomly sampled neighbor.
- `"median"` → Noise2Void2: manipulated voxels are replaced with the median of their local neighborhood.

No other code path differs between the two — `N2VManipulate.__call__` dispatches internally based on this field.

## 5. Pitfalls encountered during the migration

These are worth documenting since they are easy to reintroduce if this code is refactored again, and because most of them are not specific to N2V — they were simply latent, pre-existing issues that surfaced once the environment (PyTorch Lightning version, CAREamics as a new dependency) changed.

- **`n2v_mask` computed but not passed to `masked_loss`.** The most functionally significant bug encountered: without it, training silently degrades into mostly learning the identity (see §3, `unet.py`). Always verify that the mask returned by `N2VManipulate` actually reaches the loss function.
- **`N2VManipulate` instantiated conditionally (`if self.current_epoch == 0`).** This breaks when resuming from a checkpoint (epoch > 0) or when `trainer.validate()` is called standalone before `trainer.fit()`. Resolved by using a lazy `@property` instead of relying on a specific Lightning hook (see §3).
- **PyTorch Lightning API drift**, unrelated to N2V but surfaced by upgrading the environment to resolve CAREamics's dependencies:
  - `Trainer(resume_from_checkpoint=...)` was removed in PL ≥ 2.0; use `trainer.fit(ckpt_path=...)` instead.
  - `Trainer(strategy=None)` is no longer accepted; use `strategy="auto"` for the same "let Lightning choose" behavior.
- **`torch.median(dim=...)` (used by N2V2's `median_manipulate`) has no deterministic CUDA implementation for its returned indices**, which conflicts with `Trainer(deterministic=True)`. Since only `.values` is used (not `.indices`), the actual median value is unaffected by this non-determinism — resolved by setting `deterministic="warn"` instead of `True`, which downgrades this specific class of violations to a warning instead of a hard failure, while keeping strict determinism everywhere else.
- **`careamics`'s local editable install failing to build** (`setuptools-scm was unable to detect version`): CAREamics uses `[tool.hatch.version] source = "vcs"`, which requires the `careamics/` directory itself to be a git repository with tags. Since the local checkout used here is not its own git repository, this was resolved by pinning a static `version = "0.1.0"` in `careamics/pyproject.toml` instead of relying on VCS-derived versioning.
- **Stray `IndentationError` on the first line of a CAREamics source file**, introduced by a copy-paste artifact while inspecting/editing the file manually — a reminder to check for invisible leading whitespace when editing vendored source files by hand.

## 6. Known open points / possible future work

- **Validation-time masking is currently stochastic**, not deterministic: `validation_step` reuses the same `N2VManipulate` instance (and thus the same shared random generator) as `training_step`, so a different set of voxels is masked at each validation epoch. This means `val_loss` is not computed on exactly the same masked positions across epochs, which could add some epoch-to-epoch noise to the validation metric. A deterministic alternative would require a second `N2VManipulate` instance with a fixed seed (similar to how `SubtomoDataset` already supports `deterministic_rotations` for validation).
- **`masked_pixel_percentage` was raised well above CAREamics's 2D-oriented default (0.2%) to 5%**, to compensate for the sparsity of supervision signal in large 3D sub-tomograms (a 96³ sub-tomogram has ~885k voxels; at 0.2%, fewer than 1800 voxels would be supervised per training step). This value has not been empirically validated for this specific use case and may need tuning based on observed convergence behavior.
- **DDP multi-GPU training**: each process instantiates its own `N2VManipulate` with the same configured seed. Since each replica processes different data (via `DistributedSampler`), this is not expected to cause correctness issues, but the exact interaction between per-replica random generators and DDP gradient synchronization has not been specifically audited.