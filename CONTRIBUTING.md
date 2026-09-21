# Contributing a compression method

Your job: read a paper, implement its method in **one file**, open a PR.
You don't need to touch anything else. The library already handles copying the model,
finding your method, benchmarking, finetuning, and (for pruning) all layer-shape bookkeeping.

## 1. Setup

```bash
git clone https://github.com/bandiarham07-bot/Model-Shrinker && cd Model-Shrinker
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # everything should pass before you start
```

## 2. Write the method

Work on a branch: `git checkout -b method/<name>`. Create `shrinker/methods/<name>.py`
(and, if you want, `tests/test_<name>.py`). The method file is discovered automatically.
**Don't edit `shrinker/methods/__init__.py`.**

### The contract (every method must follow this)

1. **One file** in `shrinker/methods/`, with one function decorated with `@register`.
   The **file name is the method name**: `my_method.py` → `compress(model, "my_method")`.
   ```python
   from shrinker.registry import register

   @register
   def apply(model: nn.Module, ratio: float = 0.3) -> nn.Module:
       ...
       return model
   ```
2. Optional extras, only if you want them: a different name `@register("other_name")`,
   and `kind="pruning"` (or `factorization`, `fusion`, `quantization`), which sets the order
   when methods are combined.
3. `model` is **already a copy**. Modify it freely, and `return` it.
4. **Every option has a default**, so `compress(model, "<name>")` works. Use the shared names:

   | Name | Meaning |
   |---|---|
   | `ratio` | fraction of channels/weights to remove, in [0, 1) |
   | `loader` | a DataLoader of `(inputs, labels)`, for methods that need data (calibration, Taylor scores). The only option allowed without a default. |
   | `example_input` | a sample input batch, for methods that need to trace or run the model |

5. **Skip what you don't support, never crash.** Unsupported layer types or model structures are
   left unchanged. Log them: `logging.getLogger(__name__).info("Skipping %s: <reason>", name)`.
6. **Read shapes from the layers** (`layer.out_channels`, `layer.in_features`), never assume them.
   Your method may run on a model another method has already shrunk.
7. **Prefer removing whole filters/neurons** (use the helpers in `shrinker.utils`): that shrinks the
   file and speeds the model up. Zeroing individual weights (unstructured pruning) is allowed, but
   on its own it doesn't make the file smaller or the model faster; `compare()` shows it as fewer
   non-zero params.
8. Optionally add tests for your method in `tests/` (see step 3).
9. Only edit your own files. Shared files (`core.py`, `registry.py`, `dependency.py`, `utils.py`,
   `benchmark.py`, `finetune.py`) are changed by the maintainers. If you need a new shared helper,
   ask the maintainer.

### Pruning methods: you only write the scoring

For pruning, the only part that's specific to your paper is **how to score channels**.
`prunable_layers()` gives you the layers that are safe to prune, and `prune_channels()` removes
channels from a layer **and** its BatchNorm **and** the next layer's inputs, all with the same
indices. It's tested to be exact. See [`l1_pruning.py`](shrinker/methods/l1_pruning.py):

```python
from shrinker.utils import keep_indices, prunable_layers, prune_channels

for name, layer in prunable_layers(model):
    scores = layer.weight.detach().abs().flatten(1).sum(dim=1)   # ← the paper's idea goes here
    keep = keep_indices(scores, ratio)                            # indices of the top (1 - ratio)
    prune_channels(model, name, keep)
```

- `prunable_layers` yields layers in execution order and always gives you the **current** layer,
  including input channels already removed by pruning the previous layer.
- The final classifier and layers feeding residual connections are skipped automatically.
- You may need other modules too, e.g. Network Slimming scores channels by the following
  BatchNorm's γ. Get any submodule with `shrinker.utils.get_module(model, name)`, or find the BN
  names with `shrinker.dependency.analyze(model)[0][name].batchnorms`.

### Other methods (quantization, factorization, fusion)

Replace layers with `shrinker.utils.replace_module(model, "features.3", new_layer)`. Useful helpers
in `shrinker.utils`: `get_module`, `replace_module`, `slice_conv_out/in`, `slice_linear_out/in`,
`slice_bn`.

Notes:
- **`torch.ao.quantization` is deprecated** in recent PyTorch releases (it still works but warns).
  The replacement is the separate [`torchao`](https://github.com/pytorch/ao) package. If your method
  needs it, mention it in your pull request so we can add it as an optional dependency.
- **Quantization speedups depend on the backend.** INT8 is fast on CPU (fbgemm/qnnpack), but eager
  INT8 on GPU often isn't. Quantized models are usually CPU-only.
- **Low-rank factorization** of an m×n weight into rank k only saves anything if k < mn / (m + n).
- **Always look at the measured latency** in `compare()`. Fewer FLOPs doesn't always mean faster.

## 3. Test it

`tests/test_contract.py` automatically checks **every** registered method on the six test models in
`examples/test_models.py` (plain MLP, tabular MLP with BatchNorm, VGG-style CNN, ResNet with
shortcuts, MobileNet with depthwise convs, and a CNN written with `F.relu`/`x.view`, run with a
batch of 1 and a non-square image). You get these checks for free:

- the file imports without errors
- every option has a default (except `loader`)
- the compressed model runs, has the same output shape, and gives finite outputs
- the original model is not modified
- the result is never bigger, and is smaller (or has more zero weights) on at least one test model
- it doesn't crash on models it doesn't fully support
- it can be saved and loaded (only a warning if not, since some PyTorch model types can't be)
- it can be combined with every other method (nothing is applied after quantization)

In `tests/test_<name>.py`, test what is **specific** to your method, for example that it removes
the channels your paper says are least important, and that options work.
See [`tests/test_l1_pruning.py`](tests/test_l1_pruning.py).

```bash
pytest                          # everything
pytest -k <name>                # just your method
python examples/demo.py --synthetic --method <name>   # end to end, with a before/after table
```

## 4. Open a pull request

- Push your branch and open a pull request.
- CI runs `pytest` on every PR. It must pass before merging.
- Add your method to the **Available methods** table in `README.md`.

## Code style

- Python 3.9+.
- Use `logging`, not `print`.
- No new dependencies beyond `torch` without asking the maintainer first (CI only installs `torch`).
