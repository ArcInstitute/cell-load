import pickle
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import toml
import torch

from cell_load.config import ExperimentConfig
from cell_load.data_modules.samplers import PerturbationBatchSampler
from cell_load.utils.data_utils import GlobalH5MetadataCache, H5MetadataCache
from conftest import create_toml_config, make_datamodule


def test_zeroshot_excludes_celltype(synthetic_data):
    """Test that zeroshot cell types are excluded from training."""
    root, cell_types = synthetic_data
    zs_ct = cell_types[0]  # CT0

    config = {
        "datasets": {
            "dataset1": "placeholder"
        },  # Will be updated by create_toml_config
        "training": {"dataset1": "train"},
        "zeroshot": {f"dataset1.{zs_ct}": "test"},
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm.setup()

        # Debug: Check if datasets were created
        print(f"Train datasets: {len(dm.train_datasets)}")
        print(f"Val datasets: {len(dm.val_datasets)}")
        print(f"Test datasets: {len(dm.test_datasets)}")

        # Verify we have some data
        assert len(dm.train_datasets) > 0, "No training datasets created"
        assert len(dm.test_datasets) > 0, "No test datasets created"

        # Verify zeroshot cell type is not in training data
        train_celltypes = set()
        for subset in dm.train_datasets:
            ds = subset.dataset
            for idx in subset.indices:
                train_celltypes.add(ds.get_cell_type(idx))

        print(f"Training cell types: {train_celltypes}")
        assert zs_ct not in train_celltypes, (
            f"Zeroshot cell type {zs_ct} found in training data"
        )

        # Verify zeroshot cell type is in test data
        test_celltypes = set()
        for subset in dm.test_datasets:
            ds = subset.dataset
            for idx in subset.indices:
                test_celltypes.add(ds.get_cell_type(idx))

        print(f"Test cell types: {test_celltypes}")
        assert zs_ct in test_celltypes, (
            f"Zeroshot cell type {zs_ct} not found in test data"
        )

    finally:
        toml_path.unlink()  # Clean up temp file


def test_fewshot_perturbation_splitting(synthetic_data):
    """Test that fewshot perturbations are correctly split between train/val/test."""
    root, cell_types = synthetic_data
    fs_ct = cell_types[1]  # CT1

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
        "fewshot": {
            f"dataset1.{fs_ct}": {
                "val": ["P1", "P2"],
                "test": ["P3", "P4", "P1"],  # P1 appears in both val and test
            }
        },
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm.setup()

        # Debug: Check if datasets were created
        print(f"Train datasets: {len(dm.train_datasets)}")
        print(f"Val datasets: {len(dm.val_datasets)}")
        print(f"Test datasets: {len(dm.test_datasets)}")

        # Verify we have data in all splits
        assert len(dm.train_datasets) > 0, "No training datasets created"
        assert len(dm.val_datasets) > 0, "No validation datasets created"
        assert len(dm.test_datasets) > 0, "No test datasets created"

        # Check that fewshot cell type appears in all splits
        def get_celltypes_in_split(datasets):
            return {
                subset.dataset.get_cell_type(idx)
                for subset in datasets
                for idx in subset.indices
            }

        train_cts = get_celltypes_in_split(dm.train_datasets)
        val_cts = get_celltypes_in_split(dm.val_datasets)
        test_cts = get_celltypes_in_split(dm.test_datasets)

        print(f"Train celltypes: {train_cts}")
        print(f"Val celltypes: {val_cts}")
        print(f"Test celltypes: {test_cts}")

        assert fs_ct in train_cts, f"Fewshot celltype {fs_ct} not in training"
        assert fs_ct in val_cts, f"Fewshot celltype {fs_ct} not in validation"
        assert fs_ct in test_cts, f"Fewshot celltype {fs_ct} not in test"

        # Verify specific perturbations are in correct splits
        def get_perturbations_for_celltype(datasets, target_ct):
            perts = set()
            for subset in datasets:
                ds = subset.dataset
                for idx in subset.indices:
                    if ds.get_cell_type(idx) == target_ct:
                        pert_name = ds.get_perturbation_name(idx)
                        if pert_name != "P0":  # Skip control
                            perts.add(pert_name)
            return perts

        train_perts = get_perturbations_for_celltype(dm.train_datasets, fs_ct)
        val_perts = get_perturbations_for_celltype(dm.val_datasets, fs_ct)
        test_perts = get_perturbations_for_celltype(dm.test_datasets, fs_ct)

        print(f"Train perts for {fs_ct}: {train_perts}")
        print(f"Val perts for {fs_ct}: {val_perts}")
        print(f"Test perts for {fs_ct}: {test_perts}")

        # Val should contain P1, P2
        assert "P1" in val_perts, "P1 not found in validation"
        assert "P2" in val_perts, "P2 not found in validation"

        # Test should contain P1, P3, P4
        assert "P1" in test_perts, "P1 not found in test"
        assert "P3" in test_perts, "P3 not found in test"
        assert "P4" in test_perts, "P4 not found in test"

        # Train should contain remaining perturbations (P5-P9)
        expected_train = {"P5", "P6", "P7", "P8", "P9"}
        assert expected_train.issubset(train_perts), (
            f"Expected train perts {expected_train} not found in {train_perts}"
        )

    finally:
        toml_path.unlink()


def test_training_dataset_behavior(synthetic_data):
    """Test that training datasets include all non-overridden cell types."""
    root, cell_types = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder", "dataset2": "placeholder"},
        "training": {"dataset1": "train", "dataset2": "train"},
        "zeroshot": {"dataset1.CT0": "test"},  # Exclude CT0 from training
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm.setup()

        # Debug info
        print(f"Train datasets: {len(dm.train_datasets)}")
        print(f"Test datasets: {len(dm.test_datasets)}")

        assert len(dm.train_datasets) > 0, "No training datasets created"

        # Get all cell types in training data
        train_celltypes = set()
        for subset in dm.train_datasets:
            ds = subset.dataset
            for idx in subset.indices:
                train_celltypes.add(ds.get_cell_type(idx))

        print(f"Training cell types found: {train_celltypes}")

        # Should include CT1, CT2 from dataset1 and CT3, CT4, CT5 from dataset2
        # Should NOT include CT0 (zeroshot)
        expected_celltypes = {"CT1", "CT2", "CT3", "CT4", "CT5"}

        assert "CT0" not in train_celltypes, "CT0 should not be in training (zeroshot)"
        assert train_celltypes == expected_celltypes, (
            f"Expected {expected_celltypes}, got {train_celltypes}"
        )

    finally:
        toml_path.unlink()


def test_config_validation():
    """Test that configuration validation works correctly."""
    # Test missing dataset paths
    config = {"training": {"nonexistent_dataset": "train"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config, f)
        toml_path = Path(f.name)

    try:
        experiment_config = ExperimentConfig.from_toml(str(toml_path))
        with pytest.raises(ValueError, match="Missing dataset paths"):
            experiment_config.validate()
    finally:
        toml_path.unlink()

    # Test invalid splits
    config = {
        "datasets": {"dataset1": "/fake/path"},
        "zeroshot": {"dataset1.CT0": "invalid_split"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config, f)
        toml_path = Path(f.name)

    try:
        experiment_config = ExperimentConfig.from_toml(str(toml_path))
        with pytest.raises(ValueError, match="Invalid split"):
            experiment_config.validate()
    finally:
        toml_path.unlink()


def test_sampler_groups(synthetic_data):
    """Test that the sampler correctly groups cells by cell type and perturbation."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=3,
            control_pert="P0",
        )
        dm.setup()

        # Check that we have training data
        assert len(dm.train_datasets) > 0, "No training datasets created"

        loader = dm.train_dataloader()
        assert loader is not None, "Train dataloader is None"

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
            # Break batch into sentences of length 20
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

                assert len(ct_codes) == 1, (
                    f"Mixed cell types in chunk {chunk}: {ct_codes}"
                )
                assert len(pert_codes) == 1, (
                    f"Mixed perts in chunk {chunk}: {pert_codes}"
                )

    finally:
        toml_path.unlink()


def test_collate_fn_shapes_and_keys(synthetic_data):
    """Test that the collate function produces correctly shaped outputs."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset2": "placeholder"},
        "training": {"dataset2": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path, embed_key="X_hvg", batch_size=4, control_pert="P0"
        )
        dm.setup()

        assert len(dm.train_datasets) > 0, "No training datasets created"

        loader = dm.train_dataloader()
        assert loader is not None, "Train dataloader is None"

        batch = next(iter(loader))

        for key in (
            "pert_cell_emb",
            "ctrl_cell_emb",
            "pert_emb",
            "pert_name",
            "cell_type",
            "cell_type_onehot",
            "batch",
            "batch_name",
        ):
            assert key in batch
        assert isinstance(batch["pert_cell_emb"], torch.Tensor)
        assert batch["pert_cell_emb"].shape[0] == 4
        assert batch["pert_cell_emb"].ndim == 2

        # Test with X_state embedding
        dm = make_datamodule(
            toml_path, embed_key="X_state", batch_size=4, control_pert="P0"
        )
        dm.setup()

        assert len(dm.train_datasets) > 0, (
            "No training datasets created for X_state test"
        )

        loader = dm.train_dataloader()
        assert loader is not None, "Train dataloader is None for X_state test"

        batch = next(iter(loader))

        for key in (
            "pert_cell_emb",
            "ctrl_cell_emb",
            "pert_cell_counts",
            "pert_emb",
            "pert_name",
            "cell_type",
            "cell_type_onehot",
            "batch",
            "batch_name",
        ):
            assert key in batch
        assert isinstance(batch["pert_cell_emb"], torch.Tensor)
        assert batch["pert_cell_emb"].shape[0] == 4
        assert batch["pert_cell_emb"].ndim == 2

    finally:
        toml_path.unlink()


def test_getitem_basal_matches_control(synthetic_data):
    """Test that control cell mappings work correctly."""
    root, _ = synthetic_data

    config = {"datasets": {"dataset1": ""}, "training": {"dataset1": "train"}}

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path, embed_key="X_hvg", batch_size=4, control_pert="P0"
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
        # Control embedding must be same shape as perturbation embedding and non-negative
        assert sample["ctrl_cell_emb"].shape == sample["pert_cell_emb"].shape
        assert torch.all(sample["ctrl_cell_emb"] >= 0)

    finally:
        toml_path.unlink()


def test_to_subset_dataset_control_flag(synthetic_data):
    """Test the should_yield_control_cells flag."""
    root, _ = synthetic_data

    config = {"datasets": {"dataset1": ""}, "training": {"dataset1": "train"}}

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(toml_path, embed_key="X_hvg", control_pert="P0")
        dm.setup()
        subset = dm.train_datasets[0]
        ds = subset.dataset

        all_idxs = np.array(subset.indices)
        ctrl_mask = (
            ds.metadata_cache.pert_codes[all_idxs]
            == ds.metadata_cache.control_pert_code
        )
        pert_idxs = all_idxs[~ctrl_mask]
        ctrl_idxs = all_idxs[ctrl_mask]

        # By default we yield controls
        full = ds.to_subset_dataset("val", pert_idxs, ctrl_idxs)
        assert len(full.indices) == len(pert_idxs) + len(ctrl_idxs)

        # Turn off yielding controls
        ds.should_yield_control_cells = False
        no_ctrl = ds.to_subset_dataset("val", pert_idxs, ctrl_idxs)
        assert len(no_ctrl.indices) == len(pert_idxs)

    finally:
        toml_path.unlink()


def test_pickle_and_unpickle_dataset(synthetic_data):
    """Test that datasets can be pickled and unpickled."""
    root, _ = synthetic_data

    config = {"datasets": {"dataset1": ""}, "training": {"dataset1": "train"}}

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(toml_path, embed_key="X_hvg", control_pert="P0")
        dm.setup()
        ds = dm.train_datasets[0].dataset

        data = pickle.dumps(ds)
        ds2 = pickle.loads(data)
        # After unpickle, handle must re-open
        assert hasattr(ds2, "h5_file")
        assert ds2.n_cells == ds.n_cells

    finally:
        toml_path.unlink()


def test_invalid_split_name_raises(synthetic_data):
    """Test that invalid split names raise appropriate errors."""
    root, _ = synthetic_data

    config = {"datasets": {"dataset1": ""}, "training": {"dataset1": "train"}}

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(toml_path, embed_key="X_hvg", control_pert="P0")
        dm.setup()
        ds = dm.train_datasets[0].dataset

        with pytest.raises(ValueError):
            ds.to_subset_dataset("invalid_split", np.array([0]), np.array([]))

    finally:
        toml_path.unlink()


def test_H5MetadataCache_parses_categories_and_codes(synthetic_data):
    """Test that H5 metadata caching works correctly."""
    root, cell_types = synthetic_data
    # Pick one file
    fpath = next((root / "dataset1").glob("*.h5"))
    cache = H5MetadataCache(
        str(fpath),
        pert_col="gene",
        cell_type_key="cell_type",
        control_pert="P0",
        batch_col="gem_group",
    )

    # Perturbation categories should be P0..P9
    assert list(cache.pert_categories) == [f"P{i}" for i in range(10)]
    # Only one cell_type category
    assert list(cache.cell_type_categories) == [fpath.stem]
    # Only one batch category
    assert list(cache.batch_categories) == ["batch1"]
    # Codes length matches total cells
    assert cache.n_cells == 10 * 100
    # Control mask sums to 100 (cells with P0)
    assert int(cache.control_mask.sum()) == 100


def test_GlobalH5MetadataCache_singleton_behavior(synthetic_data):
    """Test that global metadata cache behaves as a singleton."""
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


def test_complex_fewshot_scenario(synthetic_data):
    """Test a complex scenario with multiple fewshot configurations."""
    root, cell_types = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder", "dataset2": "placeholder"},
        "training": {"dataset2": "train"},  # Only dataset2 is for training
        "zeroshot": {
            "dataset1.CT0": "test",  # CT0 entirely goes to test
            "dataset1.CT1": "val",  # CT1 entirely goes to val
        },
        "fewshot": {
            "dataset1.CT2": {  # CT2 is split by perturbations
                "val": ["P1", "P2"],
                "test": ["P8", "P9"],
                # P3-P7 will go to train automatically
            }
        },
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm.setup()

        # Debug info
        print(f"Train datasets: {len(dm.train_datasets)}")
        print(f"Val datasets: {len(dm.val_datasets)}")
        print(f"Test datasets: {len(dm.test_datasets)}")

        # Verify datasets exist
        assert len(dm.train_datasets) > 0, "No training datasets created"
        assert len(dm.val_datasets) > 0, "No validation datasets created"
        assert len(dm.test_datasets) > 0, "No test datasets created"

        # Get cell types in each split
        def get_celltypes_in_split(datasets):
            return {
                subset.dataset.get_cell_type(idx)
                for subset in datasets
                for idx in subset.indices
            }

        train_cts = get_celltypes_in_split(dm.train_datasets)
        val_cts = get_celltypes_in_split(dm.val_datasets)
        test_cts = get_celltypes_in_split(dm.test_datasets)

        print(f"Train celltypes: {train_cts}")
        print(f"Val celltypes: {val_cts}")
        print(f"Test celltypes: {test_cts}")

        # Verify expected distributions
        assert "CT0" not in train_cts, (
            "CT0 should not be in training (zeroshot to test)"
        )
        assert "CT1" not in train_cts, "CT1 should not be in training (zeroshot to val)"
        assert "CT2" in train_cts, "CT2 should be in training (fewshot partial)"
        assert {"CT3", "CT4", "CT5"}.issubset(train_cts), (
            "Training dataset cell types should be in training"
        )

        assert "CT0" not in val_cts, "CT0 should not be in val (goes to test)"
        assert "CT1" in val_cts, "CT1 should be in val (zeroshot to val)"
        assert "CT2" in val_cts, "CT2 should be in val (fewshot partial)"

        assert "CT0" in test_cts, "CT0 should be in test (zeroshot to test)"
        assert "CT1" not in test_cts, "CT1 should not be in test (goes to val)"
        assert "CT2" in test_cts, "CT2 should be in test (fewshot partial)"

        # Verify perturbation splitting for CT2
        def get_perturbations_for_celltype(datasets, target_ct):
            perts = set()
            for subset in datasets:
                ds = subset.dataset
                for idx in subset.indices:
                    if ds.get_cell_type(idx) == target_ct:
                        pert_name = ds.get_perturbation_name(idx)
                        if pert_name != "P0":  # Skip control
                            perts.add(pert_name)
            return perts

        ct2_train_perts = get_perturbations_for_celltype(dm.train_datasets, "CT2")
        ct2_val_perts = get_perturbations_for_celltype(dm.val_datasets, "CT2")
        ct2_test_perts = get_perturbations_for_celltype(dm.test_datasets, "CT2")

        print(f"CT2 train perts: {ct2_train_perts}")
        print(f"CT2 val perts: {ct2_val_perts}")
        print(f"CT2 test perts: {ct2_test_perts}")

        # Check perturbation assignments
        assert {"P1", "P2"}.issubset(ct2_val_perts), (
            f"P1,P2 should be in val, got {ct2_val_perts}"
        )
        assert {"P8", "P9"}.issubset(ct2_test_perts), (
            f"P8,P9 should be in test, got {ct2_test_perts}"
        )
        assert {"P3", "P4", "P5", "P6", "P7"}.issubset(ct2_train_perts), (
            f"P3-P7 should be in train, got {ct2_train_perts}"
        )

    finally:
        toml_path.unlink()


def test_barcode_functionality(synthetic_data):
    """Test that cell barcodes are included when barcode=True."""
    root, cell_types = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        # Test with barcode=True
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
            barcode=True,
        )
        dm.setup()

        # Get a batch from the dataloader
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Check that barcodes are present
        assert "pert_cell_barcode" in batch, "pert_cell_barcode not found in batch"
        assert "ctrl_cell_barcode" in batch, "ctrl_cell_barcode not found in batch"

        # Check that barcodes are lists of strings
        assert isinstance(batch["pert_cell_barcode"], list), (
            "pert_cell_barcode should be a list"
        )
        assert isinstance(batch["ctrl_cell_barcode"], list), (
            "ctrl_cell_barcode should be a list"
        )
        assert len(batch["pert_cell_barcode"]) == len(batch["ctrl_cell_barcode"]), (
            "barcode lists should have same length"
        )

        # Check that barcodes are strings
        for barcode in batch["pert_cell_barcode"]:
            assert isinstance(barcode, str), "barcodes should be strings"
        for barcode in batch["ctrl_cell_barcode"]:
            assert isinstance(barcode, str), "barcodes should be strings"

        # Test with barcode=False (default)
        dm_no_barcode = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm_no_barcode.setup()

        # Get a batch from the dataloader
        train_loader_no_barcode = dm_no_barcode.train_dataloader()
        batch_no_barcode = next(iter(train_loader_no_barcode))

        # Check that barcodes are not present
        assert "pert_cell_barcode" not in batch_no_barcode, (
            "pert_cell_barcode should not be present when barcode=False"
        )
        assert "ctrl_cell_barcode" not in batch_no_barcode, (
            "ctrl_cell_barcode should not be present when barcode=False"
        )

    finally:
        toml_path.unlink()  # Clean up temp file
