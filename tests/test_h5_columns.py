import h5py
import numpy as np
import pytest

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


def _write_nullable_strings(adata, path, **kwargs):
    import anndata as ad
    import pytest

    if not hasattr(ad.settings, "allow_write_nullable_strings"):
        pytest.skip("anndata < 0.11 cannot write nullable string arrays")
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(path, **kwargs)


def test_metadata_cache_reads_string_batch_column(tmp_path):
    """A non-categorical string batch column loads instead of raising TypeError."""
    import anndata as ad
    import pandas as pd

    from cell_load.utils.data_utils import H5MetadataCache

    obs = pd.DataFrame(
        {
            "gene": pd.Categorical(["ctrl"] * 2 + ["g0"] * 2),
            "cell_type": pd.Categorical(["A"] * 4),
            "batch": pd.array(["b1", "b2", "b1", "b2"], dtype="string"),
        },
        index=[f"c{i}" for i in range(4)],
    )
    path = tmp_path / "batch.h5ad"
    _write_nullable_strings(
        ad.AnnData(np.ones((4, 2), dtype=np.float32), obs=obs),
        path,
        convert_strings_to_categoricals=False,
    )
    with h5py.File(path, "r") as f:
        assert isinstance(f["obs/batch"], h5py.Group)
        assert "categories" not in f["obs/batch"]

    cache = H5MetadataCache(
        str(path),
        pert_col="gene",
        cell_type_key="cell_type",
        control_pert="ctrl",
        batch_col="batch",
    )

    assert cache.batch_categories.tolist() == ["b1", "b2"]
    assert cache.batch_codes.tolist() == [0, 1, 0, 1]


def test_read_h5_column_categorical(tmp_path):
    path = tmp_path / "categorical.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("var/_index")
        group.attrs["encoding-type"] = "categorical"
        group.create_dataset("categories", data=np.array([b"g0", b"g1"]))
        group.create_dataset("codes", data=np.array([1, 0], dtype=np.int8))
        bad = f.create_group("var/with_na")
        bad.attrs["encoding-type"] = "categorical"
        bad.create_dataset("categories", data=np.array([b"g0"]))
        bad.create_dataset("codes", data=np.array([0, -1], dtype=np.int8))

    with h5py.File(path, "r") as f:
        assert safe_decode_array(read_h5_column(f["var/_index"])).tolist() == [
            "g1",
            "g0",
        ]
        with pytest.raises(ValueError, match="missing values"):
            read_h5_column(f["var/with_na"])


def test_categorical_obs_index_loads_barcodes(tmp_path):
    """A categorical obs/_index is decoded instead of raising."""
    from types import SimpleNamespace

    from cell_load.dataset._perturbation import PerturbationDataset

    path = tmp_path / "categorical.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("obs/_index")
        group.attrs["encoding-type"] = "categorical"
        group.create_dataset("categories", data=np.array([b"c0", b"c1"]))
        group.create_dataset("codes", data=np.array([1, 0, 1], dtype=np.int8))

    with h5py.File(path, "r") as f:
        stub = SimpleNamespace(h5_file=f, h5_path=path)
        barcodes = PerturbationDataset._load_cell_barcodes(stub)

    assert barcodes.tolist() == ["c1", "c0", "c1"]
