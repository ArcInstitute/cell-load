import h5py
import numpy as np

from cell_load.utils.data_utils import read_h5_column, safe_decode_array


def test_read_h5_column_plain_dataset(tmp_path):
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("obs/_index", data=np.array([b"c0", b"c1"]))

    with h5py.File(path, "r") as f:
        values = safe_decode_array(read_h5_column(f["obs/_index"]))

    assert values.tolist() == ["c0", "c1"]


def test_read_h5_column_nullable_string_array(tmp_path):
    # Layout anndata >= 0.11 writes for pandas' nullable string dtype
    # (the default string dtype in pandas >= 3).
    path = tmp_path / "nullable.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("obs/_index")
        group.attrs["encoding-type"] = "nullable-string-array"
        group.attrs["encoding-version"] = "0.1.0"
        group.create_dataset("values", data=np.array([b"c0", b"c1"]))
        group.create_dataset("mask", data=np.array([False, False]))

    with h5py.File(path, "r") as f:
        values = safe_decode_array(read_h5_column(f["obs/_index"]))

    assert values.tolist() == ["c0", "c1"]


def test_dataset_reads_string_index_and_gene_names(tmp_path):
    """Gene names and barcodes load from a file written by the installed anndata."""
    import anndata as ad
    import pandas as pd
    import pytest

    if not hasattr(ad.settings, "allow_write_nullable_strings"):
        pytest.skip("anndata < 0.11 cannot write nullable string arrays")

    from cell_load.dataset._perturbation import PerturbationDataset
    from cell_load.mapping_strategies import RandomMappingStrategy

    obs = pd.DataFrame(
        {
            "gene": pd.Categorical(["ctrl"] * 4 + ["g0"] * 4),
            "cell_type": pd.Categorical(["A"] * 8),
            "batch": pd.Categorical(["b1"] * 8),
        },
        index=pd.Index([f"c{i}" for i in range(8)], dtype="string"),
    )
    var = pd.DataFrame(
        {"gene_name": pd.array(["g0", "g1", "g2"], dtype="string")},
        index=["E0", "E1", "E2"],
    )
    path = tmp_path / "strings.h5ad"
    adata = ad.AnnData(np.ones((8, 3), dtype=np.float32), obs=obs, var=var)
    # Default on pandas >= 3; opt-in on older anndata/pandas.
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(path)
    with h5py.File(path, "r") as f:
        assert isinstance(f["obs/_index"], h5py.Group)
        assert isinstance(f["var/gene_name"], h5py.Group)

    dataset = PerturbationDataset(
        name="strings",
        h5_path=path,
        mapping_strategy=RandomMappingStrategy(),
        pert_onehot_map=None,
        batch_onehot_map=None,
        cell_type_onehot_map=None,
        pert_col="gene",
        cell_type_key="cell_type",
        batch_col="batch",
        control_pert="ctrl",
        embed_key=None,
    )

    assert dataset.get_gene_names() == ["g0", "g1", "g2"]
    assert dataset._load_cell_barcodes().tolist() == [f"c{i}" for i in range(8)]
