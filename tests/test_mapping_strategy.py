import numpy as np
import pytest

from cell_load.mapping_strategies import RandomMappingStrategy, BatchMappingStrategy
from conftest import create_toml_config, make_datamodule


class TestRandomMappingStrategy:
    """Test class for RandomMappingStrategy functionality."""

    @pytest.fixture(scope="function")
    def mapping_strategy_cached(self):
        """Create a mapping strategy with caching enabled."""
        return RandomMappingStrategy(
            name="test_random_cached",
            random_state=42,
            n_basal_samples=1,
            cache_perturbation_control_pairs=True,
        )

    @pytest.fixture(scope="function")
    def mapping_strategy_uncached(self):
        """Create a mapping strategy with caching disabled."""
        return RandomMappingStrategy(
            name="test_random_uncached",
            random_state=42,
            n_basal_samples=1,
            cache_perturbation_control_pairs=False,
        )

    @pytest.fixture(scope="class")
    def dataset_with_strategy(self, synthetic_data):
        """Create a dataset with mapping strategy for testing."""
        root, cell_types = synthetic_data

        # Create simple TOML config
        config = {
            "datasets": {"dataset1": "placeholder"},
            "training": {"dataset1": "train"},
        }
        toml_path = create_toml_config(root, config)

        # Create datamodule with random strategy
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            basal_mapping_strategy="random",
            cache_perturbation_control_pairs=True,
            control_pert="P0",
        )
        dm.setup()

        # Get the first dataset
        dataset = dm.train_datasets[0].dataset  # type: ignore
        return dataset, dm, toml_path

    @pytest.fixture(scope="class")
    def indices_data(self, dataset_with_strategy):
        """Extract control and perturbed indices from the dataset."""
        dataset, dm, toml_path = dataset_with_strategy

        # Get control and perturbed indices
        cache = dataset.metadata_cache
        control_mask = cache.control_mask
        control_indices = np.where(control_mask)[0]
        perturbed_indices = np.where(~control_mask)[0]

        return {
            "dataset": dataset,
            "dm": dm,
            "toml_path": toml_path,
            "cache": cache,
            "control_indices": control_indices,
            "perturbed_indices": perturbed_indices,
            "cell_type": dataset.get_cell_type(perturbed_indices[0])
            if len(perturbed_indices) > 0
            else None,
        }

    @pytest.fixture(scope="class")
    def dataloader_cached(self, synthetic_data):
        """Create a DataLoader with caching enabled."""
        root, cell_types = synthetic_data

        # Create simple TOML config
        config = {
            "datasets": {"dataset1": "placeholder"},
            "training": {"dataset1": "train"},
        }
        toml_path = create_toml_config(root, config)

        # Create datamodule with random strategy and caching enabled
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            basal_mapping_strategy="random",
            cache_perturbation_control_pairs=True,
            control_pert="P0",
            batch_size=4,
        )
        dm.setup()

        return dm.train_dataloader(), dm, toml_path

    @pytest.fixture(scope="class")
    def dataloader_uncached(self, synthetic_data):
        """Create a DataLoader with caching disabled."""
        root, cell_types = synthetic_data

        # Create simple TOML config
        config = {
            "datasets": {"dataset1": "placeholder"},
            "training": {"dataset1": "train"},
        }
        toml_path = create_toml_config(root, config)

        # Create datamodule with random strategy and caching disabled
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            basal_mapping_strategy="random",
            cache_perturbation_control_pairs=False,
            control_pert="P0",
            batch_size=4,
        )
        dm.setup()

        return dm.train_dataloader(), dm, toml_path

    def test_initialization_defaults(self):
        """Test RandomMappingStrategy initialization with default parameters."""
        strategy = RandomMappingStrategy()

        assert not strategy.cache_perturbation_control_pairs

    def test_initialization_custom_parameters(self):
        """Test RandomMappingStrategy initialization with custom parameters."""
        strategy = RandomMappingStrategy(
            name="custom_random",
            random_state=123,
            n_basal_samples=3,
            cache_perturbation_control_pairs=True,
        )

        assert strategy.name == "custom_random"  # type: ignore
        assert strategy.random_state == 123
        assert strategy.n_basal_samples == 3
        assert strategy.cache_perturbation_control_pairs

    def test_mapping_creation_at_start_cached(
        self, mapping_strategy_cached, indices_data
    ):
        """Check to see if the mapping is created at the start when caching is enabled."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        # Before registration, mapping should be empty
        assert len(mapping_strategy_cached.split_control_mapping["train"]) == 0

        # Register split indices
        mapping_strategy_cached.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # After registration, mapping should be populated
        assert len(mapping_strategy_cached.split_control_mapping["train"]) == len(
            perturbed_indices
        ) + len(control_indices)

    def test_no_mapping_regeneration_cached(self, dataloader_cached):
        """Check mappings are not regenerated every epoch when caching is enabled."""
        loader_cached, dm_cached, toml_path_cached = dataloader_cached

        # Get the mapping strategies from the datasets
        cached_strategy = dm_cached.train_datasets[0].dataset.mapping_strategy

        # Store initial mappings for cached strategy
        initial_mappings = cached_strategy.split_control_mapping["train"].copy()

        # Simulate multiple epochs by iterating over the DataLoader multiple times
        num_epochs = 3

        # Test cached strategy - mappings should remain the same across epochs
        for epoch in range(num_epochs):
            # Iterate through all batches in the epoch
            for batch_idx, batch in enumerate(loader_cached):
                # After each batch, check that mappings haven't changed
                assert (
                    cached_strategy.split_control_mapping["train"] == initial_mappings
                )

                # Only test first few batches to avoid long test times
                if batch_idx >= 2:
                    break

    def test_one_to_one_mapping_cached(self, mapping_strategy_cached, indices_data):
        """Make sure a control cell is mapping to only one perturb cell for a given perturbation."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        mapping_strategy_cached.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # Check that each control cell maps to exactly one perturbed cell for each perturbation
        # Create a nested dictionary of cell type -> perturbation -> control cell set
        cell_type_to_pert_to_control_set = {}
        for pert_idx, control_indices in mapping_strategy_cached.split_control_mapping[
            "train"
        ].items():
            if pert_idx in control_indices:
                continue
            # Get cell type and perturbation name
            cell_type = dataset.get_cell_type(pert_idx)
            pert_name = dataset.get_perturbation_name(pert_idx)

            if cell_type not in cell_type_to_pert_to_control_set:
                cell_type_to_pert_to_control_set[cell_type] = {}
            if pert_name not in cell_type_to_pert_to_control_set[cell_type]:
                cell_type_to_pert_to_control_set[cell_type][pert_name] = set()

            for control_idx in control_indices:
                assert (
                    control_idx
                    not in cell_type_to_pert_to_control_set[cell_type][pert_name]
                )
                cell_type_to_pert_to_control_set[cell_type][pert_name].add(control_idx)

    def test_split_control_mapping_keys_contain_both_cell_types(
        self, mapping_strategy_cached, indices_data
    ):
        """Make sure split_control_mapping keys contain both perturb and control cells."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        control_indices_set = set(control_indices)
        perturbed_indices_set = set(perturbed_indices)

        # Register split indices
        mapping_strategy_cached.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )
        contains_perturbed_cells = False
        contains_control_cells = False
        for cell_idx in mapping_strategy_cached.split_control_mapping["train"]:
            if cell_idx in control_indices_set:
                contains_control_cells = True

            if cell_idx in perturbed_indices_set:
                contains_perturbed_cells = True

        assert contains_control_cells and contains_perturbed_cells

    def test_random_shuffling_on_initialization(self, indices_data):
        """Make sure initialization shuffling differs across different random states."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        # Create two strategies with different random states
        strategy1 = RandomMappingStrategy(
            random_state=42, n_basal_samples=1, cache_perturbation_control_pairs=True
        )
        strategy2 = RandomMappingStrategy(
            random_state=123, n_basal_samples=1, cache_perturbation_control_pairs=True
        )

        # Register split indices for both strategies
        strategy1.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )
        strategy2.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # Mappings should be different due to different random states
        assert (
            strategy1.split_control_mapping["train"]
            != strategy2.split_control_mapping["train"]
        )

    def test_get_control_index_consistency_cached(
        self, mapping_strategy_cached, indices_data
    ):
        """Check that get_control_index always returns the same value once the class has been initialized."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        # Register split indices
        mapping_strategy_cached.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # Test that get_control_index returns the same value multiple times
        pert_idx = perturbed_indices[0]
        result1 = mapping_strategy_cached.get_control_index(dataset, "train", pert_idx)
        result2 = mapping_strategy_cached.get_control_index(dataset, "train", pert_idx)
        result3 = mapping_strategy_cached.get_control_index(dataset, "train", pert_idx)

        assert result1 == result2 == result3
        # Test whether result1 is a single number, either int or np.int64 and not an array
        assert isinstance(int(result1), int)
        assert result1 in control_indices

    def test_get_control_indices_consistency_cached(
        self, mapping_strategy_cached, indices_data
    ):
        """Test that get_control_indices always returns the same values when caching is enabled."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        # Register split indices
        mapping_strategy_cached.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # Test that get_control_indices returns the same values multiple times
        pert_idx = perturbed_indices[0]
        result1 = mapping_strategy_cached.get_control_indices(
            dataset, "train", pert_idx
        )
        result2 = mapping_strategy_cached.get_control_indices(
            dataset, "train", pert_idx
        )
        result3 = mapping_strategy_cached.get_control_indices(
            dataset, "train", pert_idx
        )

        assert np.array_equal(result1, result2)
        assert np.array_equal(result2, result3)
        assert len(result1) == mapping_strategy_cached.n_basal_samples

    def test_backwards_compatibility_uncached(
        self, mapping_strategy_uncached, indices_data
    ):
        """Test backwards compatibility when cache_perturbation_control_pairs == False."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        # Register split indices
        mapping_strategy_uncached.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # Check that no mappings are created (original behavior)
        assert len(mapping_strategy_uncached.split_control_mapping["train"]) == 0

        # Check that control pool is populated
        cell_types = dataset.get_all_cell_types(control_indices)
        for cell_type in cell_types:
            assert (
                len(mapping_strategy_uncached.split_control_pool["train"][cell_type])
                >= 0
            )

        # Test that get_control_indices works
        pert_idx = perturbed_indices[0]
        results = [
            mapping_strategy_uncached.get_control_index(dataset, "train", pert_idx)
            for i in range(5)
        ]

        # Check that all results are valid control indices
        for result in results:
            assert result in control_indices

        # Check that all results are not the same
        assert len(set(results)) > 1

    def test_multiple_splits_independence(self, mapping_strategy_cached, indices_data):
        """Test that different splits maintain independent mappings."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        # Split perturbed indices for different splits
        mid_point = len(perturbed_indices) // 2
        train_pert_indices = perturbed_indices[:mid_point]
        val_pert_indices = perturbed_indices[mid_point:]

        # Register split indices for different splits
        mapping_strategy_cached.register_split_indices(
            dataset, "train", train_pert_indices, control_indices
        )
        mapping_strategy_cached.register_split_indices(
            dataset, "val", val_pert_indices, control_indices
        )

        # Check that splits have separate mappings
        assert len(mapping_strategy_cached.split_control_mapping["train"]) == len(
            train_pert_indices
        ) + len(control_indices)
        assert len(mapping_strategy_cached.split_control_mapping["val"]) == len(
            val_pert_indices
        ) + len(control_indices)

        # Check that train and val mappings don't overlap
        train_keys = set(mapping_strategy_cached.split_control_mapping["train"].keys())
        val_keys = set(mapping_strategy_cached.split_control_mapping["val"].keys())
        assert len(train_keys.intersection(val_keys)) == len(control_indices)

    def test_random_state_reproducibility(self, indices_data):
        """Test that random_state ensures reproducible results."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        # Create two strategies with same random state
        strategy1 = RandomMappingStrategy(
            random_state=42, n_basal_samples=1, cache_perturbation_control_pairs=True
        )
        strategy2 = RandomMappingStrategy(
            random_state=42, n_basal_samples=1, cache_perturbation_control_pairs=True
        )

        # Register split indices for both strategies
        strategy1.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )
        strategy2.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # Test that mappings are identical
        assert (
            strategy1.split_control_mapping["train"]
            == strategy2.split_control_mapping["train"]
        )

    def test_cache_parameter_effectiveness(
        self, mapping_strategy_cached, mapping_strategy_uncached, indices_data
    ):
        """Test that cache_perturbation_control_pairs parameter works correctly."""
        dataset = indices_data["dataset"]
        control_indices = indices_data["control_indices"]
        perturbed_indices = indices_data["perturbed_indices"]

        # Register split indices for both strategies
        mapping_strategy_cached.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )
        mapping_strategy_uncached.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # Check that cached strategy has mappings
        assert len(mapping_strategy_cached.split_control_mapping["train"]) > 0

        # Check that uncached strategy has no mappings
        assert len(mapping_strategy_uncached.split_control_mapping["train"]) == 0

        # Check that both strategies have control pools
        cell_types = dataset.get_all_cell_types(control_indices)
        for cell_type in cell_types:
            assert (
                len(mapping_strategy_cached.split_control_pool["train"][cell_type]) > 0
            )
            assert (
                len(mapping_strategy_uncached.split_control_pool["train"][cell_type])
                > 0
            )


class TestBatchMappingStrategy:
    """Tests for BatchMappingStrategy functionality."""

    @pytest.fixture(scope="class")
    def dataset_with_batch_strategy(self, synthetic_data):
        """Create a dataset configured to use the batch mapping strategy."""
        root, cell_types = synthetic_data

        config = {
            "datasets": {"dataset1": "placeholder"},
            "training": {"dataset1": "train"},
        }
        toml_path = create_toml_config(root, config)

        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            basal_mapping_strategy="batch",
            cache_perturbation_control_pairs=False,
            control_pert="P0",
        )
        dm.setup()

        dataset = dm.train_datasets[0].dataset  # type: ignore
        return dataset, dm, toml_path

    @pytest.fixture(scope="class")
    def indices_data_batch(self, dataset_with_batch_strategy):
        """Extract control and perturbed indices for the batch mapping dataset."""
        dataset, dm, toml_path = dataset_with_batch_strategy
        cache = dataset.metadata_cache
        control_mask = cache.control_mask
        control_indices = np.where(control_mask)[0]
        perturbed_indices = np.where(~control_mask)[0]

        return {
            "dataset": dataset,
            "dm": dm,
            "toml_path": toml_path,
            "cache": cache,
            "control_indices": control_indices,
            "perturbed_indices": perturbed_indices,
        }

    def test_register_split_indices_groups_by_batch_and_cell_type(
        self, indices_data_batch
    ):
        """Ensure control pools are keyed by (batch, cell_type)."""
        dataset = indices_data_batch["dataset"]
        control_indices = indices_data_batch["control_indices"]
        perturbed_indices = indices_data_batch["perturbed_indices"]

        strategy = dataset.mapping_strategy
        assert isinstance(strategy, BatchMappingStrategy)

        strategy.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        # Keys should be tuples of (batch, cell_type)
        for key, pool in strategy.split_control_maps["train"].items():
            assert isinstance(key, tuple) and len(key) == 2
            batch_key, cell_type_key = key
            # Check pool membership aligns with the key
            for idx in pool:
                assert dataset.get_batch(idx) == batch_key
                assert dataset.get_cell_type(idx) == cell_type_key

    def test_get_control_indices_same_batch_and_cell_type(self, indices_data_batch):
        """Selected controls should share both batch and cell type with the perturbed cell."""
        dataset = indices_data_batch["dataset"]
        control_indices = indices_data_batch["control_indices"]
        perturbed_indices = indices_data_batch["perturbed_indices"]

        strategy = dataset.mapping_strategy
        assert isinstance(strategy, BatchMappingStrategy)

        strategy.register_split_indices(
            dataset, "train", perturbed_indices, control_indices
        )

        pert_idx = int(perturbed_indices[0])
        pert_batch = dataset.get_batch(pert_idx)
        pert_cell_type = dataset.get_cell_type(pert_idx)

        ctrl_idxs = strategy.get_control_indices(dataset, "train", pert_idx)
        assert isinstance(ctrl_idxs, np.ndarray)
        assert len(ctrl_idxs) == strategy.n_basal_samples

        for ci in ctrl_idxs:
            assert dataset.get_batch(int(ci)) == pert_batch
            assert dataset.get_cell_type(int(ci)) == pert_cell_type

    def test_random_state_reproducibility_batch(self, indices_data_batch):
        """Same seed should yield identical control selections."""
        dataset = indices_data_batch["dataset"]
        control_indices = indices_data_batch["control_indices"]
        perturbed_indices = indices_data_batch["perturbed_indices"]

        s1 = BatchMappingStrategy(random_state=7, n_basal_samples=3)
        s2 = BatchMappingStrategy(random_state=7, n_basal_samples=3)

        s1.register_split_indices(dataset, "train", perturbed_indices, control_indices)
        s2.register_split_indices(dataset, "train", perturbed_indices, control_indices)

        pert_idx = int(perturbed_indices[0])
        out1 = s1.get_control_indices(dataset, "train", pert_idx)
        out2 = s2.get_control_indices(dataset, "train", pert_idx)
        assert np.array_equal(out1, out2)
