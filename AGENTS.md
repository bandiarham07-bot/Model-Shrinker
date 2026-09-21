# Instructions for AI assistants

This repo is **Model Shrinker**: a PyTorch library that takes a trained MLP/CNN and returns a
smaller, faster copy via `shrinker.compress(model, "<method>", **options)`.

The library is finished. Contributors only **add compression methods**. Your task is almost
always: implement one method (from a paper or the user's own idea) as **one new file**.

## Rules

1. Create exactly one file: `shrinker/methods/<method_name>.py`. The file name is the method name.
   Optionally add `tests/test_<method_name>.py`.
2. **Do not edit any other file.** Everything outside `shrinker/methods/<your file>` and
   `tests/test_<your file>` is shared by all contributors. If the method truly needs a new
   dependency, stop and tell the user (it must be added to `pyproject.toml` by the maintainer).
3. The method is one function:
   ```python
   from shrinker.registry import register

   @register
   def apply(model, ratio=0.3):      # every option needs a default, except `loader`
       ...                           # `model` is already a copy: modify it freely
       return model
   ```
   Shared option names: `ratio` (fraction to remove), `loader` (DataLoader of `(x, y)` if the
   method needs data), `example_input` (sample batch if the method needs one).
4. **Pruning methods only compute a score per channel.** Use the library for the surgery; it also
   fixes the following BatchNorm and next layer and skips unsafe layers. See
   `shrinker/methods/l1_pruning.py`:
   ```python
   from shrinker.utils import keep_indices, prunable_layers, prune_channels

   for name, layer in prunable_layers(model):
       scores = ...                                # one score per output channel, higher = keep
       prune_channels(model, name, keep_indices(scores, ratio))
   ```
5. Other methods replace layers with `shrinker.utils.replace_module(model, name, new_layer)`.
6. Skip layers or models you can't handle (return them unchanged, log with `logging`); never crash.
7. Run `pytest` and make sure everything passes before finishing. `tests/test_contract.py`
   automatically tests every method on the six models in `examples/test_models.py`.

More detail: `CONTRIBUTING.md`. Project overview: `README.md`.
