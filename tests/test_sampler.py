import torch.multiprocessing as mp
import os
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch.distributed as dist

from cell_load.data_modules.samplers import PerturbationBatchSampler
from cell_load.dataset import MetadataConcatDataset
from conftest import create_toml_config, make_datamodule


def distributed_worker(rank, world_size, test_data, results_list):
    """Worker function that runs in each distributed process"""
    try:
        # Initialize distributed
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(12355 + world_size)  # Unique port per test
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Unpack test data
        root, toml_config, batch_size, cell_sentence_len = test_data

        # Create datamodule
        dm = make_datamodule(
            toml_config,
            embed_key="X_hvg",
            batch_size=batch_size,
            control_pert="P0",
        )
        dm.setup()

        # Create dataset
        ds = MetadataConcatDataset(dm.train_datasets)

        # Create sampler
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=batch_size,
            drop_last=False,
            cell_sentence_len=cell_sentence_len,
            test=False,
            use_batch=False,
            seed=42,
        )

        # Collect results
        results = {
            "batch_size": batch_size,
            "cell_sentence_len": cell_sentence_len,
            "rank": rank,
            "distributed": sampler.distributed,
            "num_replicas": sampler.num_replicas,
            "sampler_rank": sampler.rank,
            "total_batches": len(sampler),
            "total_sentences": len(sampler.sentences),
            "batch_data": sampler.batches.copy(),
        }

        sampler.set_epoch(1)
        results["ep1_batch_data"] = sampler.batches.copy()

        # Put results in queue
        results_list.append(results)

        # Clean up
        dist.destroy_process_group()

    except Exception as e:
        error_result = {
            "rank": rank,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        results_list.append(error_result)


def run_distributed_test(
    synthetic_data, world_size=2, batch_size=16, cell_sentence_len=6
):
    """Helper function to run distributed test and collect results"""
    root, cell_types = synthetic_data

    # Create TOML config
    config = {
        "datasets": {"dataset1": "placeholder", "dataset2": "placeholder"},
        "training": {"dataset1": "train", "dataset2": "train"},
    }
    toml_path = create_toml_config(root, config)

    # Prepare test data
    test_data = (root, toml_path, batch_size, cell_sentence_len)

    with mp.Manager() as manager:
        results_list = manager.list()
        mp.spawn(
            distributed_worker,
            args=(world_size, test_data, results_list),
            nprocs=world_size,
        )  # type: ignore
        results = list(results_list)

    toml_path.unlink()
    return results


@pytest.fixture(scope="class")
def distributed_test_results(synthetic_data):
    """Fixture that runs distributed test once per test class."""
    return run_distributed_test(synthetic_data)


class TestDistributedPerturbationBatchSampler:
    """Test class for distributed functionality"""

    def test_distributed_initialization(self, distributed_test_results):
        """Test that both processes initialize correctly in distributed mode"""
        results = distributed_test_results

        # Check that we got results from both processes
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # Check for errors
        for result in results:
            assert "error" not in result, (
                f"Process {result.get('rank', '?')} failed: {result.get('error', 'Unknown error')}"
            )

        # Sort results by rank
        results.sort(key=lambda x: x["rank"])

        # Check distributed initialization
        for i, result in enumerate(results):
            assert result["rank"] == i, f"Expected rank {i}, got {result['rank']}"
            assert result["distributed"], f"Process {i} not in distributed mode"
            assert result["num_replicas"] == 2, f"Process {i} incorrect num_replicas"
            assert result["sampler_rank"] == i, f"Process {i} incorrect sampler rank"

    def test_correct_batch_count_generation(self, distributed_test_results):
        """Test that correct number of batches are generated across processes"""
        results = distributed_test_results

        # Check for errors
        for result in results:
            assert "error" not in result, f"Process failed: {result.get('error')}"

        # Check that both processes generate batches
        for result in results:
            assert result["total_batches"] > 0, (
                f"Process {result['rank']} generated no batches"
            )

        # Check that actual batch data was generated
        for result in results:
            assert len(result["batch_data"]) > 0, (
                f"Process {result['rank']} has no batch data"
            )
            assert len(result["batch_data"]) == result["total_batches"], (
                f"Process {result['rank']} batch count mismatch"
            )

    def test_all_processes_same_sentences(self, distributed_test_results):
        """Test that all processes see the same sentences"""
        results = distributed_test_results

        # Check for errors
        for result in results:
            assert "error" not in result, f"Process failed: {result.get('error')}"

        # Check that total sentences are same across processes (they see the same global sentences)
        sentence_counts = [result["total_sentences"] for result in results]
        assert len(set(sentence_counts)) == 1, (
            f"Processes have different sentence counts: {sentence_counts}"
        )

    def test_processes_generate_different_batches(self, distributed_test_results):
        """Test that different batches are generated with no overlap between processes"""
        results = distributed_test_results

        # Check for errors
        for result in results:
            assert "error" not in result, f"Process failed: {result.get('error')}"

        first_batches = []
        for result in results:
            first_batches.append(np.array(result["batch_data"][0]))

        assert not np.array_equal(first_batches[0], first_batches[1]), (
            "Processes generated the same first batch"
        )

    def test_epoch_shuffling_across_processes(self, distributed_test_results):
        """Test that per-epoch shuffling works correctly across processes"""
        results = distributed_test_results

        # Check for errors
        for result in results:
            assert "error" not in result, f"Process failed: {result.get('error')}"

        # Check that epoch 1 batches are different from epoch 0 for every process
        for result in results:
            ep0_batch = result["batch_data"][0]
            ep1_batch = result["ep1_batch_data"][0]
            assert not np.array_equal(ep0_batch, ep1_batch), (
                f"Epoch 1 batches are the same as epoch 0 for process {result['rank']}"
            )

    def test_last_batch_not_same_every_epoch(self, distributed_test_results):
        """Test that last batch is not the same every epoch"""
        results = distributed_test_results

        # Check for errors
        for result in results:
            assert "error" not in result, f"Process failed: {result.get('error')}"

        # Check that last batch is not the same every epoch
        for result in results:
            ep0_batch = result["batch_data"][-1]
            ep1_batch = result["ep1_batch_data"][-1]
            assert not np.array_equal(ep0_batch, ep1_batch), (
                f"Last batch is the same every epoch for process {result['rank']}"
            )


class TestSingleProcessCompatibility:
    """Test backwards compatibility for single process (non-distributed)"""

    def test_single_process_backwards_compatibility(self, synthetic_data):
        """Test that sampler works correctly in single process mode"""
        root, cell_types = synthetic_data

        # Create TOML config
        config = {
            "datasets": {"dataset1": "placeholder"},
            "training": {"dataset1": "train"},
        }
        toml_path = create_toml_config(root, config)

        try:
            # Create datamodule
            dm = make_datamodule(
                toml_path,
                embed_key="X_hvg",
                batch_size=16,
                control_pert="P0",
            )
            dm.setup()

            # Create dataset
            ds = MetadataConcatDataset(dm.train_datasets)

            # Create sampler (should be non-distributed)
            sampler = PerturbationBatchSampler(
                dataset=ds,
                batch_size=16,
                drop_last=False,
                cell_sentence_len=8,
                test=False,
                use_batch=False,
                seed=42,
            )

            sampler._create_batches()

            # Check non-distributed mode
            assert not sampler.distributed, (
                "Sampler should be in non-distributed mode"
            )
            assert sampler.num_replicas == 1, (
                "Should have 1 replica in non-distributed mode"
            )
            assert sampler.rank == 0, "Should have rank 0 in non-distributed mode"

            # Check that sampler generates batches
            total_batches = len(sampler)
            assert total_batches > 0, "Sampler should generate batches"

            # Test iteration
            batch_count = 0
            all_indices = set()
            for batch in sampler:
                batch_count += 1
                assert len(batch) > 0, f"Batch {batch_count} is empty"
                all_indices.update(batch)

            assert batch_count == total_batches, (
                "Iteration count should match len(sampler)"
            )
            assert len(all_indices) > 0, "Should process some data"

            print(
                f"Single process test passed: {total_batches} batches, {len(all_indices)} indices"
            )

        finally:
            toml_path.unlink()


def _write_single_celltype_dataset(root: Path, perturbation_sequence: list[str]) -> None:
    import anndata as ad
    import pandas as pd

    dataset_dir = root / "dataset1"
    dataset_dir.mkdir()

    n_cells = len(perturbation_sequence)
    obs = pd.DataFrame(
        {
            "gene": perturbation_sequence,
            "cell_type": ["CT0"] * n_cells,
            "gem_group": ["B0"] * n_cells,
        }
    )
    obs["gene"] = pd.Categorical(obs["gene"], categories=sorted(set(obs["gene"])))
    obs["cell_type"] = pd.Categorical(obs["cell_type"], categories=["CT0"])
    obs["gem_group"] = pd.Categorical(obs["gem_group"], categories=["B0"])

    adata = ad.AnnData(obs=obs)
    adata.obsm["X_hvg"] = np.random.rand(n_cells, 4).astype(np.float32)
    adata.write(dataset_dir / "CT0.h5")


def _write_multicelltype_dataset(
    root: Path, rows: list[tuple[str, str]], fname: str = "multi.h5"
) -> None:
    import anndata as ad
    import pandas as pd

    dataset_dir = root / "dataset1"
    dataset_dir.mkdir()

    obs = pd.DataFrame(
        {
            "gene": [gene for _, gene in rows],
            "cell_type": [ct for ct, _ in rows],
            "gem_group": ["B0"] * len(rows),
        }
    )
    obs["gene"] = pd.Categorical(obs["gene"], categories=sorted(set(obs["gene"])))
    obs["cell_type"] = pd.Categorical(
        obs["cell_type"], categories=sorted(set(obs["cell_type"]))
    )
    obs["gem_group"] = pd.Categorical(obs["gem_group"], categories=["B0"])

    adata = ad.AnnData(obs=obs)
    adata.obsm["X_hvg"] = np.random.rand(len(rows), 4).astype(np.float32)
    adata.write(dataset_dir / fname)


def test_consecutive_loading_raises_for_noncontiguous_groups(tmp_path):
    _write_single_celltype_dataset(
        tmp_path, ["P1", "P2", "P1", "P0", "P0"]
    )  # (CT0,P1) appears twice with (CT0,P2) in between

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }
    toml_path = create_toml_config(tmp_path, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=2,
            cell_sentence_len=2,
            num_workers=0,
            control_pert="P0",
            use_consecutive_loading=True,
        )
        dm.setup()

        with pytest.raises(ValueError, match="non-consecutive group"):
            dm.train_dataloader()
    finally:
        toml_path.unlink()


def test_consecutive_loading_raises_for_cross_celltype_interleaving(tmp_path):
    _write_multicelltype_dataset(
        tmp_path,
        [
            ("CT0", "P1"),
            ("CT1", "P9"),
            ("CT0", "P1"),
            ("CT1", "P9"),
            ("CT0", "P0"),
            ("CT1", "P0"),
        ],
    )

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }
    toml_path = create_toml_config(tmp_path, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=2,
            cell_sentence_len=2,
            num_workers=0,
            control_pert="P0",
            use_consecutive_loading=True,
        )
        dm.setup()

        with pytest.raises(ValueError, match="non-consecutive group"):
            dm.train_dataloader()
    finally:
        toml_path.unlink()


def test_consecutive_loading_allows_contiguous_groups(tmp_path):
    _write_single_celltype_dataset(tmp_path, ["P1", "P1", "P2", "P2", "P0", "P0"])

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }
    toml_path = create_toml_config(tmp_path, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=2,
            cell_sentence_len=2,
            num_workers=0,
            control_pert="P0",
            use_consecutive_loading=True,
        )
        dm.setup()

        train_loader = dm.train_dataloader()
        first_batch = next(iter(train_loader))
        assert "pert_cell_emb" in first_batch
    finally:
        toml_path.unlink()
