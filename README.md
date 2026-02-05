# Cell Load

A PyTorch-based data loading library for single-cell perturbation data.

## Features

- Load perturbation data from H5 files (AnnData format)
- Support for multiple cell types per dataset
- Configurable mapping strategies for control cell selection
- Zero-shot and few-shot learning support
- Cell barcode tracking (optional)
- **Preprocessing utilities for quality control and data filtering**

## Installation

```bash
uv pip install cell-load
```

## Quick Start

### 1. Create a TOML configuration file:

The TOML configuration file defines your datasets, training splits, and experimental setup. Here's the format:

```toml
# Dataset paths - maps dataset names to their directories
[datasets]
replogle = "/path/to/replogle_dataset/" # ADDS ALL h5 or h5ad files in this folder to training
jurkat = "/path/to/jurkat_dataset/"

# Training specifications
# All cell types in a dataset automatically go into training (excluding zeroshot/fewshot overrides)
[training]
replogle = "train"
jurkat = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]
"replogle.jurkat" = "test"
"jurkat.rpe1" = "val"

# Fewshot specifications - explicit perturbation lists
[fewshot]

[fewshot."replogle.rpe1"]
val = ["AARS"]
test = ["AARS", "NUP107", "RPUSD4"]  # can overlap with val
# train gets all other perturbations automatically

[fewshot."jurkat.k562"]
val = ["GENE1", "GENE2"]
test = ["GENE3", "GENE4"]
```

#### TOML Configuration Format

**`[datasets]`**: Maps dataset names to their directory paths
- Each dataset should contain H5 files (one per cell type)
- Files should be named like `cell_type.h5` or `cell_type.h5ad`

**`[training]`**: Specifies which datasets are used for training
- Set to `"train"` to include all cell types in training (except those in zeroshot/fewshot)

**`[zeroshot]`**: Holds out entire cell types for testing
- Format: `"dataset.cell_type" = "split"`
- Split can be `"val"` or `"test"`
- Example: `"replogle.jurkat" = "test"` holds out all Jurkat cells from training

**`[fewshot]`**: Holds out specific perturbations within cell types
- Format: `[fewshot."dataset.cell_type"]`
- `val = ["pert1", "pert2"]`: Perturbations for validation
- `test = ["pert3", "pert4"]`: Perturbations for testing
- Remaining perturbations go to training

It is worth noting that control cell mapping is only done withi the same file (e.g., 
a perturbed cell will not get mapped to a control cell from a different h5 file, even
if it has matched covariates).

### 2. Command Line Usage

The most common parameters for data loading are:

```bash
# Basic required parameters
data.kwargs.toml_config_path=/path/to/config.toml
data.kwargs.embed_key=X_hvg
data.kwargs.num_workers=24
data.kwargs.batch_col=gem_group
data.kwargs.pert_col=gene
data.kwargs.cell_type_key=cell_type
data.kwargs.control_pert=non-targeting

# Optional parameters
data.kwargs.barcode=true  # Enable cell barcode output
data.kwargs.perturbation_features_file=/path/to/gene_embeddings.pt
data.kwargs.output_space=gene
data.kwargs.basal_mapping_strategy=random
data.kwargs.n_basal_samples=1
data.kwargs.should_yield_control_cells=true
```

These plug in as hydra configurable settings in the [STATE](https://github.com/ArcInstitute/state) repository.

### 3. Standalone Programmatic Usage

```python
from cell_load.data_modules import PerturbationDataModule

dm = PerturbationDataModule(
    # Required parameters
    toml_config_path="/path/to/config.toml",
    embed_key="X_hvg",
    num_workers=24,
    batch_col="gem_group",
    pert_col="gene",
    cell_type_key="cell_type",
    control_pert="non-targeting",
    
    # Optional parameters
    barcode=True,  # Enable cell barcode output
    perturbation_features_file="/path/to/gene_embeddings.pt",
    output_space="gene",
    basal_mapping_strategy="random",
    n_basal_samples=1,
    should_yield_control_cells=True,
    batch_size=128,
)
dm.setup()

# Get training data
train_loader = dm.train_dataloader()
for batch in train_loader:
    # batch contains:
    # - pert_cell_emb: perturbed cell embeddings
    # - ctrl_cell_emb: control cell embeddings
    # - pert_emb: perturbation one-hot encodings or embeddings
    # - pert_name: perturbation names
    # - cell_type: cell types
    # - batch: batch information
    # - pert_cell_barcode: cell barcodes (if barcode=True)
    # - ctrl_cell_barcode: control cell barcodes (if barcode=True)
    pass
```

## Preprocessing

Cell Load provides several preprocessing utilities to help with data quality control and filtering before training.

### Quality Control: On-Target Knockdown Filtering

The `filter_on_target_knockdown` function filters perturbation data based on the effectiveness of gene knockdown. This is crucial for ensuring that your perturbation experiments actually worked as intended.

```python
import anndata
from cell_load.utils.data_utils import filter_on_target_knockdown

# Load your AnnData object
adata = anndata.read_h5ad("your_data.h5ad")

# Apply quality control filtering
filtered_adata = filter_on_target_knockdown(
    adata=adata,
    perturbation_column="gene",           # Column in obs containing perturbation info
    control_label="non-targeting",        # Label for control cells
    residual_expression=0.30,             # Perturbation-level threshold (30% residual = 70% knockdown)
    cell_residual_expression=0.50,        # Cell-level threshold (50% residual = 50% knockdown)
    min_cells=30,                         # Minimum cells per perturbation after filtering
    layer=None,                           # Use adata.X (or specify a layer)
    var_gene_name="gene_name"             # Column in var containing gene names
)

print(f"Original cells: {adata.n_obs}")
print(f"Filtered cells: {filtered_adata.n_obs}")
print(f"Removed {adata.n_obs - filtered_adata.n_obs} cells due to poor knockdown")
```

#### How the Filtering Works

The `filter_on_target_knockdown` function performs a three-stage filtering process:

1. **Perturbation-level filtering**: Keeps only perturbations where the average knockdown ≥ (1 - `residual_expression`)
2. **Cell-level filtering**: Within those perturbations, keeps only cells where knockdown ≥ (1 - `cell_residual_expression`)
3. **Minimum cell count**: Discards perturbations that have fewer than `min_cells` cells remaining after stages 1-2

Control cells are always preserved regardless of these criteria.

#### Parameters

- **`residual_expression`** (default: 0.30): Perturbation-level threshold. 0.30 means 70% knockdown required.
- **`cell_residual_expression`** (default: 0.50): Cell-level threshold. 0.50 means 50% knockdown required per cell.
- **`min_cells`** (default: 30): Minimum number of cells per perturbation after filtering.
- **`layer`**: Use a specific layer instead of `adata.X` (e.g., "counts", "log1p").
- **`var_gene_name`**: Column in `adata.var` containing gene names (default: "gene_name").

### Other Preprocessing Utilities

#### Check Individual Perturbation Effectiveness

```python
from cell_load.utils.data_utils import is_on_target_knockdown

# Check if a specific perturbation worked
is_effective = is_on_target_knockdown(
    adata=adata,
    target_gene="GENE1",
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30
)
print(f"GENE1 knockdown effective: {is_effective}")
```

#### Data Type Detection

```python
from cell_load.utils.data_utils import suspected_discrete_torch, suspected_log_torch

# Check if data appears to be raw counts
is_discrete = suspected_discrete_torch(torch_tensor_data)
print(f"Data appears to be discrete counts: {is_discrete}")

# Check if data is log-transformed
is_logged = suspected_log_torch(torch_tensor_data)
print(f"Data appears to be log-transformed: {is_logged}")
```

#### Gene Name Indexing

```python
from cell_load.utils.data_utils import set_var_index_to_col

# Set the var index to use gene names from a specific column
adata = set_var_index_to_col(adata, col="gene_name")
```

### Preprocessing Workflow Example

Here's a typical preprocessing workflow:

```python
import anndata
from cell_load.utils.data_utils import filter_on_target_knockdown, set_var_index_to_col

# 1. Load data
adata = anndata.read_h5ad("raw_data.h5ad")

# 2. Set up gene names as index (if needed)
adata = set_var_index_to_col(adata, col="gene_name")

# 3. Apply quality control filtering
filtered_adata = filter_on_target_knockdown(
    adata=adata,
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30,
    cell_residual_expression=0.50,
    min_cells=30
)

# 4. Save filtered data
filtered_adata.write_h5ad("filtered_data.h5ad")

# 5. Use in your TOML config
# [datasets]
# my_dataset = "/path/to/filtered_data.h5ad"
```

## Parameter Reference

### Required Parameters

- **`toml_config_path`**: Path to the TOML configuration file defining datasets and splits
- **`embed_key`**: Key in the H5 file's `obsm` section to use for cell embeddings (e.g., "X_hvg", "X_state")
- **`pert_col`**: Column name in `obs` for perturbation information (default: "gene")
- **`cell_type_key`**: Column name in `obs` for cell type information (default: "cell_type")
- **`batch_col`**: Column name in `obs` for batch/plate information (default: "gem_group")
- **`control_pert`**: Value in `pert_col` that represents control cells (default: "non-targeting")

### Optional Parameters

- **`barcode`**: If `true`, include cell barcodes in output (default: `false`)
- **`perturbation_features_file`**: Path to .pt file containing pre-computed gene embeddings
- **`output_space`**: Output space for model predictions ("gene" or "all", default: "gene")
- **`basal_mapping_strategy`**: Strategy for mapping perturbed cells to controls ("batch" or "random", default: "random")
- **`n_basal_samples`**: Number of control cells to sample per perturbed cell (default: 1)
- **`should_yield_control_cells`**: Include control cells in output (default: `true`)
- **`num_workers`**: Number of workers for data loading (default: 8)
- **`batch_size`**: Batch size for training (default: 128)
- **`val_subsample_fraction`**: Fraction of validation subsets to keep (e.g., `0.01` keeps ~1% of `val_datasets`)

### Usage

When creating the data module programmatically:

```python
from cell_load.data_modules import PerturbationDataModule

dm = PerturbationDataModule(
    toml_config_path="config.toml",
    # ... other parameters
)
```

## Advanced Configuration

### Zero-shot Learning

To set up zero-shot learning (entire cell types held out for testing):

```toml
[zeroshot]
"dataset.cell_type" = "test"
```

### Few-shot Learning

To set up few-shot learning (specific perturbations held out):

```toml
[fewshot."dataset.cell_type"]
val = ["pert1", "pert2"]
test = ["pert3", "pert4"]
```

### Custom Perturbation Features

To use pre-computed gene embeddings instead of one-hot encodings:

```bash
data.kwargs.perturbation_features_file=/path/to/gene_embeddings.pt
```

The .pt file should contain a dictionary mapping gene names to embedding vectors.

## Dataset Management

### Getting Datasets

Currently, Cell Load expects datasets to be in H5/AnnData format and stored locally. Users need to:

1. **Obtain datasets** from their original sources (e.g., published papers, repositories)
2. **Convert to AnnData format** if not already in that format
3. **Apply preprocessing** using the utilities described above
4. **Organize by cell type** with one H5 file per cell type
