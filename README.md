# Model Shrinker

Take a PyTorch model you have **already trained** and get back a **smaller, faster copy** of it,
while keeping as much accuracy as possible.

Works with MLPs and CNNs (`nn.Linear`, `nn.Conv2d`, `nn.BatchNorm`). RNNs and Transformers are
not supported yet. Everything runs on your own machine.

## How it works

```
 your trained model
        │
        ▼
 1. compress()   pick a method, e.g. "l1_pruning"  →  you get a smaller COPY
        │                                              (your original is never changed)
        ▼
 2. compare()    see the difference: size, speed, accuracy
        │
        ▼
 3. finetune()   train the small model a little to win back lost accuracy
        │
        ▼
 4. save it and use the small model
```

A **method** is one way of making a model smaller, for example removing the least important
filters (pruning) or storing weights with fewer bits (quantization). Each method lives in its
own file in `shrinker/methods/`, and the team keeps adding new ones.

## Install

```bash
pip install git+https://github.com/bandiarham07-bot/Model-Shrinker
```

## Example

```python
import torch
import shrinker
from model import MyNet                      # your own model class

model = MyNet()
model.load_state_dict(torch.load("model.pth"))

example = torch.randn(1, 3, 32, 32)          # one input of the right shape

small = shrinker.compress(model, "l1_pruning", ratio=0.3, example_input=example)
shrinker.compare(model, small, example_input=example, val_loader=val_loader)

shrinker.finetune(small, train_loader, epochs=5)
torch.save(small, "model_small.pt")
```

`compare()` prints a table like this:

```
              original  compressed       change
-----------------------------------------------
params         666,858     167,802       -74.8%
size (MB)        2.681       0.683       -74.5%
latency (ms)    15.693       5.919  2.65x speed
MFLOPs         3804.56      979.53       -74.3%
accuracy       100.00%     100.00%    +0.00 pts
```

Speed ("latency") is measured by actually running the model, not estimated.

**Good to know**
- `ratio=0.3` means "remove about 30%". Each method has its own options.
- **Save the whole model** with `torch.save(small, path)` and load it with
  `torch.load(path, weights_only=False)`. A shrunk model no longer fits your original class,
  so `load_state_dict` into `MyNet()` won't work.
- `example_input` is optional. If you give it, the library checks the small model still gives
  the same output shape.
- Layers a method can't handle safely are left unchanged instead of being broken.
- `shrinker.list_methods()` shows all available methods.

## Available methods

| Method | What it does |
|---|---|
| `l1_pruning` | Removes the filters/neurons with the smallest weights ([Li et al. 2017](https://arxiv.org/abs/1608.08710)) |

## What's inside

```
shrinker/            the library
    core.py          compress(): copies your model and runs the chosen method on the copy
    registry.py      keeps the list of methods; finds new method files automatically
    dependency.py    works out which layers can be pruned safely, and prunes them without
                     breaking the layers connected to them
    utils.py         small tools for building smaller layers
    benchmark.py     compare(): measures size, speed, accuracy
    finetune.py      finetune(): retrains after compressing
    methods/         one file per method  ←  this is where the team adds new methods
examples/
    test_models.py   six small models every method is automatically tested on
    demo.py          the full flow from start to finish, with real numbers
tests/               automatic checks, run on every push by GitHub (CI)
```

Try the full flow yourself (no download needed, under a minute):

```bash
python examples/demo.py --synthetic
```

## Adding a method (for the team)

Setup, once:

```bash
git clone https://github.com/bandiarham07-bot/Model-Shrinker && cd Model-Shrinker
pip install -e ".[dev]"
pytest
```

Then:

1. Create **one file** in `shrinker/methods/`, e.g. `network_slimming.py`. The file name becomes
   the method name.
2. Write one function with `@register` above it. For pruning you only write **how important
   each channel is**; the library removes the channels and fixes the connected layers.
3. Run `pytest`. Your method is automatically tested on all six test models.
4. Push to a branch and open a pull request. CI runs the same tests.

You never need to change any other file. Full guide: [CONTRIBUTING.md](CONTRIBUTING.md).
Copy the pattern from [`l1_pruning.py`](shrinker/methods/l1_pruning.py).

## References

**Methods** (papers the team implements, one method file each)
1. Molchanov et al. 2019, [Importance Estimation for Neural Network Pruning](https://arxiv.org/abs/1906.10771): remove the filters whose removal hurts the loss least, estimated with gradients
2. He et al. 2019, [Filter Pruning via Geometric Median (FPGM)](https://arxiv.org/abs/1811.00250): remove redundant filters that duplicate others
3. Lin et al. 2020, [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179): remove filters whose outputs carry little information
4. Jacob et al. 2018, [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877): store weights and activations as 8-bit integers
5. Nagel et al. 2019, [Data-Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721): make 8-bit quantization lose less accuracy, without data
6. Kim et al. 2016, [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530): split big layers into smaller ones (Tucker decomposition / SVD)
7. Hinton et al. 2015, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531): train the small model to copy the big one

**Background reading** (ideas and studies behind the methods)
1. Blalock et al. 2020, [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033): how to judge pruning methods fairly
2. Liu et al. 2019, [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270): what really matters after structured pruning
3. Frankle & Carbin 2019, [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635): which weights in a network really matter
4. Fang et al. 2023, [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900): how removing a channel affects connected layers
5. Nagel et al. 2021, [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295): practical guide to quantization
6. Gou et al. 2021, [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525): overview of distillation
7. Ma et al. 2018, [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164): why fewer FLOPs doesn't always mean faster

## License

MIT
