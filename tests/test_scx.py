from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
import toml

from cell_load.data_modules.perturbation_dataloader import PerturbationDataModule
from cell_load.dataset import PerturbationDataset
from cell_load.mapping_strategies import RandomMappingStrategy
from cell_load.utils.data_utils import H5MetadataCache


pyscx = pytest.importorskip("pyscx")


@pytest.fixture
def scx_path(tmp_path: Path) -> Path:
    obs = pd.DataFrame(
        {
            "gene": pd.Categorical(["non-targeting", "non-targeting", "P1", "P1"]),
            "cell_type": pd.Categorical(["CT1"] * 4),
            "gem_group": pd.Categorical(["batch1", "batch1", "batch2", "batch2"]),
            "score": [0.1, 0.2, 0.3, 0.4],
        },
        index=["cell-0", "cell-1", "cell-2", "cell-3"],
    )
    var = pd.DataFrame(
        {"gene_name": ["G1", "G2", "G3"]},
        index=["ENSG1", "ENSG2", "ENSG3"],
    )
    adata = ad.AnnData(
        X=sp.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 7]], dtype=np.float32),
        obs=obs,
        var=var,
    )
    adata.obsm["X_hvg"] = np.asarray(adata.X.toarray()[:, :2], dtype=np.float32)
    adata.uns["log1p"] = {"base": None}

    path = tmp_path / "tiny.scx"
    pyscx.from_anndata(adata, str(path))
    return path


def test_scx_is_discovered(scx_path: Path):
    datamodule = object.__new__(PerturbationDataModule)

    assert datamodule._find_dataset_files(scx_path.parent) == {"tiny": scx_path}


def test_scx_backed_dataset_contract(scx_path: Path):
    cache = H5MetadataCache(
        str(scx_path),
        pert_col="gene",
        cell_type_key="cell_type",
        control_pert="non-targeting",
        batch_col="gem_group",
    )
    assert cache.pert_categories.tolist() == ["P1", "non-targeting"]
    assert cache.cell_type_categories.tolist() == ["CT1"]
    assert cache.n_cells == 4

    dataset = PerturbationDataset(
        name="tiny",
        h5_path=scx_path,
        mapping_strategy=RandomMappingStrategy(random_state=0),
        pert_col="gene",
        cell_type_key="cell_type",
        batch_col="gem_group",
        control_pert="non-targeting",
        embed_key="X_hvg",
        output_space="all",
        barcode=True,
        additional_obs=["score"],
    )

    assert dataset.n_cells == 4
    assert dataset.n_genes == 3
    assert dataset.get_gene_names() == ["G1", "G2", "G3"]
    assert dataset.cell_barcodes.tolist() == [
        "cell-0",
        "cell-1",
        "cell-2",
        "cell-3",
    ]
    assert dataset._fetch_obs_value(2, "score") == pytest.approx(0.3)
    torch.testing.assert_close(
        dataset.fetch_gene_expression(2), torch.tensor([4.0, 0.0, 5.0])
    )
    torch.testing.assert_close(
        dataset.fetch_obsm_expression(3, "X_hvg"), torch.tensor([0.0, 6.0])
    )
    torch.testing.assert_close(
        dataset._fetch_gene_expression_batch(np.array([3, 0])),
        torch.tensor([[0.0, 6.0, 7.0], [1.0, 0.0, 2.0]]),
    )
    torch.testing.assert_close(
        dataset._fetch_obsm_expression_batch(np.array([2, 1]), "X_hvg"),
        torch.tensor([[4.0, 0.0], [0.0, 3.0]]),
    )


def test_scx_datamodule_yields_training_batch(scx_path: Path, tmp_path: Path):
    config_path = tmp_path / "split.toml"
    with config_path.open("w") as config_file:
        toml.dump(
            {
                "datasets": {"tiny": str(scx_path)},
                "training": {"tiny": "train"},
            },
            config_file,
        )

    datamodule = PerturbationDataModule(
        toml_config_path=str(config_path),
        batch_size=2,
        num_workers=0,
        pert_col="gene",
        cell_type_key="cell_type",
        batch_col="gem_group",
        control_pert="non-targeting",
        embed_key="X_hvg",
        output_space="all",
        cell_sentence_len=2,
        use_consecutive_loading=True,
    )
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    assert batch["pert_cell_emb"].shape[-1] == 2
    assert batch["pert_cell_counts"].shape[-1] == 3
