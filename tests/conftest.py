
import pytest
import tempfile
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import toml 
from cell_load.utils.modules import get_datamodule

@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    """
    Create two dataset folders, each containing 3 H5 files (one per cell type)
    via AnnData.write. Each file has:
      - 10 perturbations (including control P0), 100 cells each
      - 1 batch category
      - obsm/X_hvg embedding of dimension 5
    Returns root path and list of all 6 cell types.
    """
    root = tmp_path_factory.mktemp("data_root")
    n_perts = 10
    n_cells_per_pert = 100
    emb_dim = 5
    batch_name = "batch1"
    cell_types = [f"CT{i}" for i in range(6)]
    perts = [f"P{i}" for i in range(n_perts)]

    for ds_idx, ds_name in enumerate(["dataset1", "dataset2"]):
        ds_dir = root / ds_name
        ds_dir.mkdir()

        # 3 cell‐types per dataset
        for ct in cell_types[3 * ds_idx : 3 * ds_idx + 3]:
            n_cells = n_perts * n_cells_per_pert

            # Build the obs DataFrame
            genes = np.repeat(perts, n_cells_per_pert)
            cell_type_arr = np.repeat(ct, n_cells)
            batch_arr = np.repeat(batch_name, n_cells)
            df = pd.DataFrame(
                {
                    "gene": genes,
                    "cell_type": cell_type_arr,
                    "gem_group": batch_arr,
                }
            )

            # Cast to categorical so AnnData writes categories & codes
            df["gene"] = pd.Categorical(df["gene"], categories=perts)
            df["cell_type"] = pd.Categorical(df["cell_type"], categories=[ct])
            df["gem_group"] = pd.Categorical(df["gem_group"], categories=[batch_name])

            # Create a dummy AnnData (no X needed for these tests)
            adata = ad.AnnData(obs=df)
            adata.obsm["X_hvg"] = np.random.rand(n_cells, emb_dim).astype(np.float32)
            adata.obsm["X_state"] = np.random.rand(n_cells, emb_dim).astype(np.float32)

            # Write directly to .h5 — AnnData will use the same HDF5 layout
            fpath = ds_dir / f"{ct}.h5"
            adata.write(fpath)

    return root, cell_types

def create_toml_config(root: Path, config_dict: dict) -> Path:
    """Create a temporary TOML configuration file."""
    # Make a deep copy to avoid modifying the original
    import copy

    config_copy = copy.deepcopy(config_dict)

    # Update dataset paths to use the test data root
    if "datasets" in config_copy:
        for dataset_name in config_copy["datasets"]:
            config_copy["datasets"][dataset_name] = str(root / dataset_name)

    # Create temporary file with proper closing
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as toml_file:
        toml.dump(config_copy, toml_file)
        toml_file.flush()
        toml_path = Path(toml_file.name)

    # Debug: Print the TOML content
    print(f"Created TOML config at {toml_path}")
    with open(toml_path, "r") as f:
        print("TOML content:")
        print(f.read())

    return toml_path


def make_datamodule(toml_config_path: Path, **dm_kwargs):
    """Create a data module using TOML configuration."""
    kwargs = {
        "toml_config_path": str(toml_config_path),
        **dm_kwargs,
    }
    return get_datamodule(
        "PerturbationDataModule", kwargs, batch_size=kwargs.get("batch_size", 16)
    )