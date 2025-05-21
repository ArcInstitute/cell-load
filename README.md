# vc-load

> Dataloaders for state-sets models

## Installation

Distributed via [`uv`](https://docs.astral.sh/uv).

```bash
# Clone repo
git clone github.com:arcinstitute/vc-load
cd vc-load

# Initialize venv
uv venv

# Install
uv pip install -e .
```

## Data Module Usage

This package provides a single Lightning‐compatible DataModule, `PerturbationDataModule`, to train perturbation models using
`h5py` instead of loading AnnData's into memory. You can instantiate a `PerturbationDataModule` as:

```python
from vc_load.utils.modules import get_datamodule
from vc_load.data_modules.tasks import TaskSpec, TaskType

# Define training and testing specifications
train_specs = [
    TaskSpec(dataset="replogle", task_type=TaskType.TRAINING),
    TaskSpec(dataset="tahoe", task_type=TaskType.TRAINING),
]

# remove jurkat cells from training by adding them to the test spec
test_specs = [
    TaskSpec(dataset="replogle", cell_type="jurkat", task_type=TaskType.ZEROSHOT),
]

# Create data module
dm = get_datamodule(
    "PerturbationDataModule",
    kwargs={
        "train_specs":       train_specs,
        "test_specs":        test_specs,
        "data_dir":          "/path/to/ctc/datasets",
        "few_shot_percent":  0.3,
        "embed_key":         "X_hvg",
        "output_space":      "gene",
        "basal_mapping_strategy": "nearest",
        "n_basal_samples":   1,
        "num_workers":       8,
        "batch_size":        128,
        "cell_sentence_len": 512,
        "should_yield_control_cells": True,
        "map_controls":      True,
        "pert_col":          "gene",
        "batch_col":         "gem_group",
        "cell_type_key":     "cell_type",
        "control_pert":      "DMSO_TF",
        "int_counts":        False,
        "normalize_counts":  False,
        "store_raw_basal":   False,
        "perturbation_features_file": None,
    },
    batch_size=128,
    cell_sentence_len=512,
)

# Setup and create data loaders
dm.setup()
train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()
```

## Folder Organization

The dataset should be a subfolder in the data_dir provided above.

## Parsing Dataset Specifications

The package provides a utility function to parse dataset specifications from strings. This is particularly useful when passing dataset specifications from command-line arguments or configuration files:

```python
from vc_load.data_modules.tasks import parse_dataset_specs
```

### Usage in Training Scripts

You can use this function in your training scripts to parse task specifications from string inputs.
The `parse_dataset_specs` function expects the user to provide colon separated entries on which cell types
to add to use for zeroshot, fewshot, or training. For example, to train on all cells in the replogle dataset
located in `data_dir/replogle`:

```python
train_task = [
    "replogle:jurkat:training",
    "replogle:hepg2:training",
    "replogle:k562:training",
]
train_specs = parse_dataset_specs(train_task)

test_task = ["replogle:rpe1:fewshot"] # or zeroshot
test_specs = parse_dataset_specs(test_task)

dm = get_datamodule(...) # see above snippet
```

or, more briefly, to use all cell types not in the test spec for a given dataset:


```python
train_task = ["replogle"]
train_specs = parse_dataset_specs(train_task)

test_task = ["replogle:rpe1:fewshot"] # or zeroshot
test_specs = parse_dataset_specs(test_task)

dm = get_datamodule(...) # see above snippet
```


## Configuration Parameters

When calling `get_datamodule("PerturbationDataModule", kwargs, batch_size, cell_sentence_len)`, note that the number of cells 
yielded will be `batch_size * cell_sentence_len`. The kwargs are:

```python
name: PerturbationDataModule
kwargs:
  # Required parameters
  train_specs: []                   # List[TaskSpec], required
  test_specs: []                    # List[TaskSpec], required
  data_dir: ""                      # str, path to root of per-dataset subfolders
  
  # Optional parameters with defaults
  few_shot_percent: 0.3             # float, only for FEWSHOT splits
  embed_key: "X_hvg"                # str or null, key in obsm/ for embeddings
  output_space: "gene"              # "gene" or "all"
  num_workers: 8                    # int, DataLoader workers
  pin_memory: true                  # bool
  batch_size: 128                   # int, meta-batch size = batch_size × cell_sentence_len
  cell_sentence_len: 512            # int, number of cells per sentence
  n_basal_samples: 1                # int, control samples per perturbed cell
  random_seed: 42                   # int, RNG for few-shot splits
  basal_mapping_strategy: "random"  # "batch", "random"
  should_yield_control_cells: true  # if false, will only yield perturbed cells and their mapped control cells
  batch_col: "gem_group"            # str, obs/ field name for batch
  pert_col: "gene"                  # str, obs/ field name for perturbation
  cell_type_key: "cell_type"        # str, obs/ field name for cell type
  control_pert: "non-targeting"     # str, label to treat as control
  map_controls: true                # if true, will map control cells to other random control cells
  perturbation_features_file: null  # torch file containing embeddings for perturbations
  store_raw_basal: false            # yield the basal cell's gene counts in each batch
  int_counts: false                 # if true, use raw counts, otherwise use log counts
  
output_dir: null                    # str or null
```

Any keys you omit will be taken from the defaults shown above.
