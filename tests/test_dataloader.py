import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from vc_load.data_modules.samplers import PerturbationBatchSampler
from vc_load.data_modules.tasks import TaskSpec, TaskType
from vc_load.utils.data_utils import GlobalH5MetadataCache, H5MetadataCache
from vc_load.utils.modules import get_datamodule


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

            # Write directly to .h5 — AnnData will use the same HDF5 layout
            fpath = ds_dir / f"{ct}.h5"
            adata.write(fpath)

    return root, cell_types


def make_datamodule(root, train_specs, test_specs, **dm_kwargs):
    kwargs = {
        "train_specs": train_specs,
        "test_specs": test_specs,
        "data_dir": str(root),
        **dm_kwargs,
    }
    # use get_datamodule to instantiate
    return get_datamodule(
        "PerturbationDataModule", kwargs, batch_size=kwargs.get("batch_size", 16)
    )


def test_zero_shot_excludes_celltype(synthetic_data):
    root, cell_types = synthetic_data
    zs_ct = cell_types[0]

    train_specs = [TaskSpec(dataset="dataset1", task_type=TaskType.TRAINING)]
    test_specs = [
        TaskSpec(dataset="dataset1", cell_type=zs_ct, task_type=TaskType.ZEROSHOT)
    ]

    dm = make_datamodule(
        root,
        train_specs,
        test_specs,
        few_shot_percent=0.3,
        embed_key="X_hvg",
        batch_size=16,
        control_pert="P0",
    )
    dm.setup()
    for subset in dm.train_datasets:
        ds = subset.dataset
        for idx in subset.indices:
            assert ds.get_cell_type(idx) != zs_ct


def test_few_shot_includes_celltype_both(synthetic_data):
    root, cell_types = synthetic_data
    fs_ct = cell_types[1]

    train_specs = [TaskSpec(dataset="dataset1", task_type=TaskType.TRAINING)]
    test_specs = [
        TaskSpec(dataset="dataset1", cell_type=fs_ct, task_type=TaskType.FEWSHOT)
    ]

    dm = make_datamodule(
        root,
        train_specs,
        test_specs,
        few_shot_percent=0.4,
        embed_key="X_hvg",
        batch_size=16,
        control_pert="P0",
    )
    dm.setup()

    seen_in_train = any(
        subset.dataset.get_cell_type(idx) == fs_ct
        for subset in dm.train_datasets
        for idx in subset.indices
    )
    seen_in_test = any(
        subset.dataset.get_cell_type(idx) == fs_ct
        for subset in dm.test_datasets
        for idx in subset.indices
    )
    assert seen_in_train and seen_in_test


def test_sampler_groups(synthetic_data):
    root, _ = synthetic_data
    train_specs = [TaskSpec(dataset="dataset1", task_type=TaskType.TRAINING)]

    dm = make_datamodule(
        root,
        train_specs,
        [],
        few_shot_percent=0.3,
        embed_key="X_hvg",
        batch_size=3,
        control_pert="P0",
    )
    dm.setup()
    loader = dm.train_dataloader()
    ds = loader.batch_sampler.dataset

    cell_sentence_len = 20
    sampler = PerturbationBatchSampler(
        dataset=ds,
        batch_size=3,
        cell_sentence_len=cell_sentence_len,
        test=False,
        use_batch=False,
    )
    batches = list(sampler)

    for batch in batches:
        # break your batch into sentences of length 20
        for i in range(0, len(batch), cell_sentence_len):
            chunk = batch[i : i + cell_sentence_len]
            ct_codes = set()
            pert_codes = set()

            for gidx in chunk:
                offset = 0
                for subset in ds.datasets:
                    if gidx < offset + len(subset):
                        local = subset.indices[gidx - offset]
                        cache = subset.dataset.metadata_cache
                        ct_codes.add(cache.cell_type_codes[local])
                        pert_codes.add(cache.pert_codes[local])
                        break
                    offset += len(subset)

            assert len(ct_codes) == 1, f"Mixed cell types in chunk {chunk}: {ct_codes}"
            assert len(pert_codes) == 1, f"Mixed perts in chunk {chunk}: {pert_codes}"


def test_collate_fn_shapes_and_keys(synthetic_data):
    root, _ = synthetic_data
    train_specs = [TaskSpec(dataset="dataset2", task_type=TaskType.TRAINING)]

    dm = make_datamodule(
        root, train_specs, [], embed_key="X_hvg", batch_size=4, control_pert="P0"
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    for key in (
        "X",
        "basal",
        "pert",
        "pert_name",
        "cell_type",
        "gem_group",
        "gem_group_name",
    ):
        assert key in batch
    assert isinstance(batch["X"], torch.Tensor)
    assert batch["X"].shape[0] == 4
    assert batch["X"].ndim == 2


def test_getitem_basal_matches_control(synthetic_data):
    root, _ = synthetic_data
    train_specs = [TaskSpec(dataset="dataset1", task_type=TaskType.TRAINING)]

    dm = make_datamodule(
        root, train_specs, [], embed_key="X_hvg", batch_size=4, control_pert="P0"
    )
    dm.setup()
    subset = dm.train_datasets[0]
    ds = subset.dataset
    ds.to_subset_dataset(
        "train",
        perturbed_indices=np.array(subset.indices)[
            ds.metadata_cache.pert_codes[subset.indices]
            != ds.metadata_cache.control_pert_code
        ],
        control_indices=np.array(subset.indices)[
            ds.metadata_cache.pert_codes[subset.indices]
            == ds.metadata_cache.control_pert_code
        ],
    )
    sample = ds[subset.indices[0]]
    # basal must be same shape as X and non-negative
    assert sample["basal"].shape == sample["X"].shape
    assert torch.all(sample["basal"] >= 0)


def test_to_subset_dataset_control_flag(synthetic_data):
    root, _ = synthetic_data
    train_specs = [TaskSpec(dataset="dataset1", task_type=TaskType.TRAINING)]

    dm = make_datamodule(root, train_specs, [], embed_key="X_hvg", control_pert="P0")
    dm.setup()
    subset = dm.train_datasets[0]
    ds = subset.dataset

    all_idxs = np.array(subset.indices)
    ctrl_mask = (
        ds.metadata_cache.pert_codes[all_idxs] == ds.metadata_cache.control_pert_code
    )
    pert_idxs = all_idxs[~ctrl_mask]
    ctrl_idxs = all_idxs[ctrl_mask]

    # by default we yield controls
    full = ds.to_subset_dataset("val", pert_idxs, ctrl_idxs)
    assert len(full.indices) == len(pert_idxs) + len(ctrl_idxs)

    # turn off yielding controls
    ds.should_yield_control_cells = False
    no_ctrl = ds.to_subset_dataset("val", pert_idxs, ctrl_idxs)
    assert len(no_ctrl.indices) == len(pert_idxs)


def test_pickle_and_unpickle_dataset(synthetic_data):
    root, _ = synthetic_data
    train_specs = [TaskSpec(dataset="dataset1", task_type=TaskType.TRAINING)]

    dm = make_datamodule(root, train_specs, [], embed_key="X_hvg", control_pert="P0")
    dm.setup()
    ds = dm.train_datasets[0].dataset

    data = pickle.dumps(ds)
    ds2 = pickle.loads(data)
    # after unpickle, handle must re-open
    assert hasattr(ds2, "h5_file")
    assert ds2.n_cells == ds.n_cells


def test_invalid_split_name_raises(synthetic_data):
    root, _ = synthetic_data
    train_specs = [TaskSpec(dataset="dataset1", task_type=TaskType.TRAINING)]

    dm = make_datamodule(root, train_specs, [], embed_key="X_hvg", control_pert="P0")
    dm.setup()
    ds = dm.train_datasets[0].dataset

    with pytest.raises(ValueError):
        ds.to_subset_dataset("invalid_split", np.array([0]), np.array([]))


def test_H5MetadataCache_parses_categories_and_codes(synthetic_data):
    root, cell_types = synthetic_data
    # pick one file
    fpath = next((root / "dataset1").glob("*.h5"))
    cache = H5MetadataCache(
        str(fpath),
        pert_col="gene",
        cell_type_key="cell_type",
        control_pert="P0",
        batch_col="gem_group",
    )

    # perturbation categories should be P0..P9
    assert list(cache.pert_categories) == [f"P{i}" for i in range(10)]
    # only one cell_type category
    assert list(cache.cell_type_categories) == [fpath.stem]
    # only one batch category
    assert list(cache.batch_categories) == ["batch1"]
    # codes length matches total cells
    assert cache.n_cells == 10 * 100
    # control mask sums to 100 (cells with P0)
    assert int(cache.control_mask.sum()) == 100


def test_GlobalH5MetadataCache_singleton_behavior(synthetic_data):
    root, _ = synthetic_data
    f1 = next((root / "dataset1").glob("*.h5"))
    f2 = next((root / "dataset2").glob("*.h5"))

    gcache = GlobalH5MetadataCache()
    c1 = gcache.get_cache(str(f1), "gene", "cell_type", "P0", "gem_group")
    c1b = gcache.get_cache(str(f1), "gene", "cell_type", "P0", "gem_group")
    c2 = gcache.get_cache(str(f2), "gene", "cell_type", "P0", "gem_group")

    # Same path returns same instance
    assert c1 is c1b
    # Different path returns different instance
    assert c1 is not c2
