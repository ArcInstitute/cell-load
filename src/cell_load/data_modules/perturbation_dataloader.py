import logging
import glob
import re

from functools import partial
from pathlib import Path
from typing import Literal, Set, Dict

import h5py
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config import ExperimentConfig
from ..dataset import MetadataConcatDataset, PerturbationDataset
from ..mapping_strategies import BatchMappingStrategy, RandomMappingStrategy
from ..utils.data_utils import (
    GlobalH5MetadataCache,
    generate_onehot_map,
    safe_decode_array,
)
from .samplers import PerturbationBatchSampler

logger = logging.getLogger(__name__)


class PerturbationDataModule(LightningDataModule):
    """
    A unified data module that sets up train/val/test splits for multiple dataset/celltype
    combos. Allows zero-shot, few-shot tasks, and uses a pluggable mapping strategy
    (batch, random, nearest) to match perturbed cells with control cells.
    """

    def __init__(
        self,
        toml_config_path: str,
        batch_size: int = 128,
        num_workers: int = 8,
        random_seed: int = 42,  # this should be removed by seed everything
        pert_col: str = "gene",
        batch_col: str = "gem_group",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        embed_key: Literal["X_hvg", "X_state"] | None = None,
        output_space: Literal["gene", "all"] = "gene",
        basal_mapping_strategy: Literal["batch", "random"] = "random",
        n_basal_samples: int = 1,
        should_yield_control_cells: bool = True,
        cell_sentence_len: int = 512,
        cache_perturbation_control_pairs: bool = False,
        drop_last: bool = False,
        **kwargs,  # missing perturbation_features_file  and store_raw_basal for backwards compatibility
    ):
        """
        This class is responsible for serving multiple PerturbationDataset's each of which is specific
        to a dataset/cell type combo. It sets up training, validation, and test splits for each dataset
        and cell type, and uses a pluggable mapping strategy to match perturbed cells with control cells.

        Args:
            toml_config_path: Path to TOML configuration file
            batch_size: Batch size for PyTorch DataLoader
            num_workers: Num workers for PyTorch DataLoader
            few_shot_percent: Fraction of data to use for few-shot tasks
            random_seed: For reproducible splits & sampling
            embed_key: Embedding key or matrix in the H5 file to use for feauturizing cells
            output_space: The output space for model predictions (gene or latent, which uses embed_key)
            basal_mapping_strategy: One of {"batch","random","nearest","ot"}
            n_basal_samples: Number of control cells to sample per perturbed cell
            cache_perturbation_control_pairs: If True cache perturbation-control pairs at the start of training and reuse them.
            drop_last: Whether to drop the last sentence set if it is smaller than cell_sentence_len
        """
        super().__init__()

        # Load and validate configuration
        self.toml_config_path = toml_config_path
        self.config = ExperimentConfig.from_toml(toml_config_path)
        self.config.validate()

        # Experiment level params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.drop_last = drop_last

        # H5 field names
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.output_space = output_space

        # Sampling and mapping
        self.n_basal_samples = n_basal_samples
        self.cell_sentence_len = cell_sentence_len
        self.should_yield_control_cells = should_yield_control_cells
        self.cache_perturbation_control_pairs = cache_perturbation_control_pairs

        # Optional behaviors
        self.map_controls = kwargs.get("map_controls", True)
        self.perturbation_features_file = kwargs.get("perturbation_features_file")
        self.int_counts = kwargs.get("int_counts", False)
        self.normalize_counts = kwargs.get("normalize_counts", False)
        self.store_raw_basal = kwargs.get("store_raw_basal", False)
        self.barcode = kwargs.get("barcode", False)

        logger.info(
            f"Initializing DataModule: batch_size={batch_size}, workers={num_workers}, "
            f"random_seed={random_seed}"
        )

        # Mapping strategy
        self.basal_mapping_strategy = basal_mapping_strategy
        self.mapping_strategy_cls = {
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
        }[basal_mapping_strategy]

        # Determine if raw expression is needed
        self.store_raw_expression = bool(
            self.embed_key
            and (
                (self.embed_key != "X_hvg" and self.output_space == "gene")
                or self.output_space == "all"
            )
        )

        # Prepare dataset lists and maps
        self.train_datasets: list[Dataset] = []
        self.val_datasets: list[Dataset] = []
        self.test_datasets: list[Dataset] = []

        self.all_perts: Set[str] = set()
        self.pert_onehot_map: dict[str, torch.Tensor] | None = None
        self.batch_onehot_map: dict[str, torch.Tensor] | None = None
        self.cell_type_onehot_map: dict[str, torch.Tensor] | None = None

        # Initialize global maps
        self._setup_global_maps()

    def get_var_names(self):
        """
        Get the variable names (gene names) from the first dataset.
        This assumes all datasets have the same gene names.
        """
        if len(self.test_datasets) == 0:
            raise ValueError("No test datasets available to extract variable names.")
        underlying_ds: PerturbationDataset = self.test_datasets[0].dataset
        return underlying_ds.get_gene_names(output_space=self.output_space)

    def setup(self, stage: str | None = None):
        """
        Set up training and test datasets.
        """
        if len(self.train_datasets) == 0:
            self._setup_datasets()
            logger.info(
                "Done! Train / Val / Test splits: %d / %d / %d",
                len(self.train_datasets),
                len(self.val_datasets),
                len(self.test_datasets),
            )

    def save_state(self, filepath: str):
        """
        Save the data module configuration to a torch file.
        This saves only the initialization parameters, not the computed splits for the datasets.

        Args:
            filepath: Path where to save the configuration (should end with .torch)
        """
        save_dict = {
            "toml_config_path": self.toml_config_path,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "random_seed": self.random_seed,
            "pert_col": self.pert_col,
            "batch_col": self.batch_col,
            "cell_type_key": self.cell_type_key,
            "control_pert": self.control_pert,
            "embed_key": self.embed_key,
            "output_space": self.output_space,
            "basal_mapping_strategy": self.basal_mapping_strategy,
            "n_basal_samples": self.n_basal_samples,
            "should_yield_control_cells": self.should_yield_control_cells,
            "cell_sentence_len": self.cell_sentence_len,
            "cache_perturbation_control_pairs": self.cache_perturbation_control_pairs,
            # Include the optional behaviors
            "map_controls": self.map_controls,
            "perturbation_features_file": self.perturbation_features_file,
            "int_counts": self.int_counts,
            "normalize_counts": self.normalize_counts,
            "store_raw_basal": self.store_raw_basal,
            "barcode": self.barcode,
        }

        torch.save(save_dict, filepath)
        logger.info(f"Saved data module configuration to {filepath}")

    @classmethod
    def load_state(cls, filepath: str):
        """
        Load a data module from a saved torch file.
        This reconstructs the data module with the original initialization parameters.
        You will need to call setup() after loading to recreate the datasets.

        Args:
            filepath: Path to the saved configuration file

        Returns:
            PerturbationDataModule: A new instance with the saved configuration
        """
        save_dict = torch.load(filepath, map_location="cpu")
        logger.info(f"Loaded data module configuration from {filepath}")

        # Validate that the toml config file still exists
        toml_path = Path(save_dict["toml_config_path"])
        if not toml_path.exists():
            logger.warning(
                f"TOML config file not found at {toml_path}. "
                "Make sure the file exists or the path is correct."
            )

        # Extract the kwargs that were passed to __init__
        kwargs = {
            "map_controls": save_dict.pop("map_controls", True),
            "cache_perturbation_control_pairs": save_dict.pop("cache_perturbation_control_pairs", False),
            "perturbation_features_file": save_dict.pop(
                "perturbation_features_file", None
            ),
            "int_counts": save_dict.pop("int_counts", False),
            "normalize_counts": save_dict.pop("normalize_counts", False),
            "store_raw_basal": save_dict.pop("store_raw_basal", False),
            "barcode": save_dict.pop("barcode", True),
        }

        # Create new instance with all the saved parameters
        return cls(**save_dict, **kwargs)

    def get_var_dims(self):
        underlying_ds: PerturbationDataset = self.test_datasets[0].dataset
        if self.embed_key:
            input_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
        else:
            input_dim = underlying_ds.n_genes

        gene_dim = underlying_ds.n_genes
        try:
            hvg_dim = underlying_ds.get_num_hvgs()
        except AttributeError:
            assert self.embed_key is None, "No X_hvg detected, using raw .X"
            hvg_dim = gene_dim

        if self.embed_key is not None:
            output_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
        else:
            output_dim = input_dim  # training on raw gene expression

        gene_names = underlying_ds.get_gene_names(output_space=self.output_space)

        # get the shape of the first value in pert_onehot_map
        pert_dim = next(iter(self.pert_onehot_map.values())).shape[0]
        batch_dim = next(iter(self.batch_onehot_map.values())).shape[0]

        pert_names = list(self.pert_onehot_map.keys())

        return {
            "input_dim": input_dim,
            "gene_dim": gene_dim,
            "hvg_dim": hvg_dim,
            "output_dim": output_dim,
            "pert_dim": pert_dim,
            "gene_names": gene_names,
            "batch_dim": batch_dim,
            "pert_names": pert_names,
        }

    def get_shared_perturbations(self) -> Set[str]:
        """
        Compute shared perturbations between train and test sets by inspecting
        only the actual subset indices in self.train_datasets and self.test_datasets.

        This ensures we don't accidentally include all perturbations from the entire h5 file.
        """

        def _extract_perts_from_subset(subset) -> Set[str]:
            """
            Helper that returns the set of perturbation names for the
            exact subset indices in 'subset'.
            """
            ds = subset.dataset  # The underlying PerturbationDataset
            idxs = subset.indices  # The subset of row indices relevant to this Subset

            # ds.pert_col typically is 'gene' or similar
            pert_codes = ds.metadata_cache.pert_codes[idxs]
            # Convert each code to its corresponding string label
            pert_names = ds.pert_categories[pert_codes]

            return set(pert_names)

        # 1) Gather all perturbations found across the *actual training subsets*
        train_perts = set()
        for subset in self.train_datasets:
            train_perts.update(_extract_perts_from_subset(subset))

        # 2) Gather all perturbations found across the *actual testing subsets*
        test_perts = set()
        for subset in self.test_datasets:
            test_perts.update(_extract_perts_from_subset(subset))

        # 3) Intersection = shared across both train and test
        shared_perts = train_perts & test_perts

        logger.info(f"Found {len(train_perts)} distinct perts in the train subsets.")
        logger.info(f"Found {len(test_perts)} distinct perts in the test subsets.")
        logger.info(f"Found {len(shared_perts)} shared perturbations (train ∩ test).")

        return shared_perts

    def get_control_pert(self):
        # Return the control perturbation name
        return self.train_datasets[0].dataset.control_pert

    def train_dataloader(self):
        if len(self.train_datasets) == 0:
            raise ValueError(
                "No training datasets available. Please call setup() first."
            )
        return self._create_dataloader(self.train_datasets, test=False)

    def val_dataloader(self):
        if len(self.val_datasets) == 0:
            return self._create_dataloader(self.test_datasets, test=False)
        return self._create_dataloader(self.val_datasets, test=False)

    def test_dataloader(self):
        if len(self.test_datasets) == 0:
            return None
        return self._create_dataloader(self.test_datasets, test=True, batch_size=1)

    def predict_dataloader(self):
        if len(self.test_datasets) == 0:
            return None
        return self._create_dataloader(self.test_datasets, test=True)

    # Helper functions to set up global maps and datasets

    def _create_dataloader(
        self,
        datasets: list[Dataset],
        test: bool = False,
        batch_size: int | None = None,
    ):
        """Create a DataLoader with appropriate configuration."""
        use_int_counts = "int_counts" in self.__dict__ and self.int_counts
        collate_fn = partial(PerturbationDataset.collate_fn, int_counts=use_int_counts)

        ds = MetadataConcatDataset(datasets)
        use_batch = self.basal_mapping_strategy == "batch"

        batch_size = batch_size or (1 if test else self.batch_size)

        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=batch_size,
            drop_last=self.drop_last,
            cell_sentence_len=self.cell_sentence_len,
            test=test,
            use_batch=use_batch,
        )

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=4 if not test and self.num_workers > 0 else None,
        )

    def _setup_global_maps(self):
        """
        Set up global one-hot maps for perturbations and batches.
        For perturbations, we scan through all files in all train_specs and test_specs.
        """
        all_perts = set()
        all_batches = set()
        all_celltypes = set()

        for dataset_name in self.config.get_all_datasets():
            dataset_path = Path(self.config.datasets[dataset_name])
            files = self._find_dataset_files(dataset_path)

            for _fname, fpath in files.items():
                with h5py.File(fpath, "r") as f:
                    pert_arr = f[f"obs/{self.pert_col}/categories"][:]
                    perts = set(safe_decode_array(pert_arr))
                    all_perts.update(perts)

                    try:
                        batch_arr = f[f"obs/{self.batch_col}/categories"][:]
                    except KeyError:
                        batch_arr = f[f"obs/{self.batch_col}"][:]
                    batches = set(safe_decode_array(batch_arr))
                    all_batches.update(batches)

                    try:
                        celltype_arr = f[f"obs/{self.cell_type_key}/categories"][:]
                    except KeyError:
                        celltype_arr = f[f"obs/{self.cell_type_key}"][:]
                    celltypes = set(safe_decode_array(celltype_arr))
                    all_celltypes.update(celltypes)

        # Create one-hot maps
        if self.perturbation_features_file:
            # Load the custom featurizations from a torch file
            featurization_dict = torch.load(self.perturbation_features_file)
            # Validate that every perturbation in all_perts is in the featurization dict.
            missing = all_perts - set(featurization_dict.keys())
            if len(missing) > 0:
                feature_dim = next(iter(featurization_dict.values())).shape[-1]
                for pert in missing:
                    featurization_dict[pert] = torch.zeros(feature_dim)

                logger.info(
                    "Set %d missing perturbations to zero vectors.", len(missing)
                )

            logger.info(
                "Loaded custom perturbation featurizations for %d perturbations.",
                len(featurization_dict),
            )
            self.pert_onehot_map = featurization_dict  # use the custom featurizations
        else:
            # Fall back to default: generate one-hot mapping
            self.pert_onehot_map = generate_onehot_map(all_perts)

        self.batch_onehot_map = generate_onehot_map(all_batches)
        self.cell_type_onehot_map = generate_onehot_map(all_celltypes)

    def _create_base_dataset(
        self, dataset_name: str, fpath: Path
    ) -> PerturbationDataset:
        """Create a base PerturbationDataset instance."""
        mapping_kwargs = {"map_controls": self.map_controls}
        
        # Add cache_perturbation_control_pairs to mapping strategy kwargs
        mapping_kwargs["cache_perturbation_control_pairs"] = self.cache_perturbation_control_pairs

        return PerturbationDataset(
            name=dataset_name,
            h5_path=fpath,
            mapping_strategy=self.mapping_strategy_cls(
                random_state=self.random_seed,
                n_basal_samples=self.n_basal_samples,
                **mapping_kwargs,
            ),
            embed_key=self.embed_key,
            pert_onehot_map=self.pert_onehot_map,
            batch_onehot_map=self.batch_onehot_map,
            cell_type_onehot_map=self.cell_type_onehot_map,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            batch_col=self.batch_col,
            control_pert=self.control_pert,
            random_state=self.random_seed,
            should_yield_control_cells=self.should_yield_control_cells,
            store_raw_expression=self.store_raw_expression,
            output_space=self.output_space,
            store_raw_basal=self.store_raw_basal,
            barcode=self.barcode,
        )

    def _setup_datasets(self):
        """
        Set up training datasets with proper handling of zeroshot/fewshot splits w/ TOML.
        Uses H5MetadataCache for faster metadata access.
        """

        for dataset_name in self.config.get_all_datasets():
            dataset_path = Path(self.config.datasets[dataset_name])
            files = self._find_dataset_files(dataset_path)

            # Get configuration for this dataset
            zeroshot_celltypes = self.config.get_zeroshot_celltypes(dataset_name)
            fewshot_celltypes = self.config.get_fewshot_celltypes(dataset_name)
            is_training_dataset = self.config.training.get(dataset_name) == "train"

            logger.info(f"Processing dataset {dataset_name}:")
            logger.info(f"  - Training dataset: {is_training_dataset}")
            logger.info(f"  - Zeroshot cell types: {list(zeroshot_celltypes.keys())}")
            logger.info(f"  - Fewshot cell types: {list(fewshot_celltypes.keys())}")

            # Process each file in the dataset
            for fname, fpath in tqdm(
                list(files.items()), desc=f"Processing {dataset_name}"
            ):
                # Create metadata cache
                cache = GlobalH5MetadataCache().get_cache(
                    str(fpath),
                    self.pert_col,
                    self.cell_type_key,
                    self.control_pert,
                    self.batch_col,
                )

                # Create base dataset
                ds = self._create_base_dataset(dataset_name, fpath)
                train_sum = val_sum = test_sum = 0

                # Process each cell type in this file
                for ct_idx, ct in enumerate(cache.cell_type_categories):
                    ct_mask = cache.cell_type_codes == ct_idx
                    n_cells = np.sum(ct_mask)

                    if n_cells == 0:
                        continue

                    ct_indices = np.where(ct_mask)[0]

                    # Split into control and perturbed indices
                    ctrl_mask = cache.pert_codes[ct_indices] == cache.control_pert_code
                    ctrl_indices = ct_indices[ctrl_mask]
                    pert_indices = ct_indices[~ctrl_mask]

                    # Determine how to handle this cell type
                    counts = self._process_celltype(
                        ds,
                        ct,
                        ct_indices,
                        ctrl_indices,
                        pert_indices,
                        cache,
                        dataset_name,
                        zeroshot_celltypes,
                        fewshot_celltypes,
                        is_training_dataset,
                    )

                    train_sum += counts["train"]
                    val_sum += counts["val"]
                    test_sum += counts["test"]

                tqdm.write(
                    f"Processed {fname}: {train_sum} train, {val_sum} val, {test_sum} test"
                )

            logger.info("\n")

    def _split_fewshot_celltype(
        self,
        ds: PerturbationDataset,
        pert_indices: np.ndarray,
        ctrl_indices: np.ndarray,
        cache,
        pert_config: dict[str, list[str]],
    ) -> dict[str, int]:
        """Split a fewshot cell type according to perturbation assignments."""
        counts = {"train": 0, "val": 0, "test": 0}

        # Get perturbation codes for this cell type
        pert_codes = cache.pert_codes[pert_indices]

        # Create sets of perturbation codes for each split
        val_pert_names = set(pert_config.get("val", []))
        test_pert_names = set(pert_config.get("test", []))

        val_pert_codes = set()
        test_pert_codes = set()

        for i, pert_name in enumerate(cache.pert_categories):
            if pert_name in val_pert_names:
                val_pert_codes.add(i)
            if pert_name in test_pert_names:
                test_pert_codes.add(i)

        # Split perturbation indices by their codes
        val_mask = np.isin(pert_codes, list(val_pert_codes))
        test_mask = np.isin(pert_codes, list(test_pert_codes))
        train_mask = ~(val_mask | test_mask)

        val_pert_indices = pert_indices[val_mask]
        test_pert_indices = pert_indices[test_mask]
        train_pert_indices = pert_indices[train_mask]

        # Split controls proportionally
        rng = np.random.default_rng(self.random_seed)
        ctrl_indices_shuffled = rng.permutation(ctrl_indices)

        n_val = len(val_pert_indices)
        n_test = len(test_pert_indices)
        n_train = len(train_pert_indices)
        total_pert = n_val + n_test + n_train

        if total_pert > 0:
            n_ctrl_val = int(len(ctrl_indices) * n_val / total_pert)
            n_ctrl_test = int(len(ctrl_indices) * n_test / total_pert)

            val_ctrl_indices = ctrl_indices_shuffled[:n_ctrl_val]
            test_ctrl_indices = ctrl_indices_shuffled[
                n_ctrl_val : n_ctrl_val + n_ctrl_test
            ]
            train_ctrl_indices = ctrl_indices_shuffled[n_ctrl_val + n_ctrl_test :]

            # Create subsets
            if len(val_pert_indices) > 0:
                subset = ds.to_subset_dataset("val", val_pert_indices, val_ctrl_indices)
                self.val_datasets.append(subset)
                counts["val"] = len(subset)

            if len(test_pert_indices) > 0:
                subset = ds.to_subset_dataset(
                    "test", test_pert_indices, test_ctrl_indices
                )
                self.test_datasets.append(subset)
                counts["test"] = len(subset)

            if len(train_pert_indices) > 0:
                subset = ds.to_subset_dataset(
                    "train", train_pert_indices, train_ctrl_indices
                )
                self.train_datasets.append(subset)
                counts["train"] = len(subset)

        return counts

    def _find_dataset_files(self, dataset_path: Path) -> dict[str, Path]:
        files: Dict[str, Path] = {}
        path_str = str(dataset_path)

        # Check if path contains glob patterns
        if any(char in path_str for char in "*?[]{}"):
            # Handle brace expansion manually since Python glob doesn't support it
            expanded_patterns = self._expand_braces(path_str)

            for pattern in expanded_patterns:
                if pattern.startswith("/"):
                    # Absolute path - use glob.glob()
                    if pattern.endswith((".h5", ".h5ad")):
                        # Pattern already specifies file extension
                        for fpath_str in sorted(glob.glob(pattern)):
                            fpath = Path(fpath_str)
                            files[fpath.stem] = fpath
                    else:
                        # Pattern doesn't specify extension, add file patterns
                        for ext in ("*.h5", "*.h5ad"):
                            full_pattern = f"{pattern.rstrip('/')}/{ext}"
                            for fpath_str in sorted(glob.glob(full_pattern)):
                                fpath = Path(fpath_str)
                                files[fpath.stem] = fpath
                else:
                    # Relative path - use Path.glob()
                    if pattern.endswith((".h5", ".h5ad")):
                        for fpath in sorted(Path().glob(pattern)):
                            files[fpath.stem] = fpath
                    else:
                        for ext in ("*.h5", "*.h5ad"):
                            full_pattern = f"{pattern.rstrip('/')}/{ext}"
                            for fpath in sorted(Path().glob(full_pattern)):
                                files[fpath.stem] = fpath
        else:
            # No glob patterns - treat as regular path
            if dataset_path.is_file():
                # Single file
                files[dataset_path.stem] = dataset_path
            else:
                # Directory - search for files
                for ext in ("*.h5", "*.h5ad"):
                    for fpath in sorted(dataset_path.glob(ext)):
                        files[fpath.stem] = fpath

        return files

    def _expand_braces(self, pattern: str) -> list[str]:
        """Expand brace patterns like {a,b,c} into multiple patterns."""

        def expand_single_brace(text: str) -> list[str]:
            # Find the first brace group
            match = re.search(r"\{([^}]+)\}", text)
            if not match:
                return [text]

            # Extract the options and expand them
            before = text[: match.start()]
            after = text[match.end() :]
            options = match.group(1).split(",")

            results = []
            for option in options:
                new_text = before + option.strip() + after
                # Recursively expand any remaining braces
                results.extend(expand_single_brace(new_text))

            return results

        return expand_single_brace(pattern)

    def _process_celltype(
        self,
        ds: PerturbationDataset,
        celltype: str,
        ct_indices: np.ndarray,
        ctrl_indices: np.ndarray,
        pert_indices: np.ndarray,
        cache,
        dataset_name: str,
        zeroshot_celltypes: dict[str, str],
        fewshot_celltypes: dict[str, dict[str, list[str]]],
        is_training_dataset: bool,
    ) -> dict[str, int]:
        """Process a single cell type and return counts for each split."""
        counts = {"train": 0, "val": 0, "test": 0}

        if celltype in zeroshot_celltypes:
            # Zeroshot: all cells go to specified split
            split = zeroshot_celltypes[celltype]
            subset = ds.to_subset_dataset(split, pert_indices, ctrl_indices)

            if split == "train":
                self.train_datasets.append(subset)
            elif split == "val":
                self.val_datasets.append(subset)
            elif split == "test":
                self.test_datasets.append(subset)

            counts[split] = len(subset)

        elif celltype in fewshot_celltypes:
            # Fewshot: split perturbations according to config
            pert_config = fewshot_celltypes[celltype]
            split_counts = self._split_fewshot_celltype(
                ds, pert_indices, ctrl_indices, cache, pert_config
            )
            for split, count in split_counts.items():
                counts[split] += count

        elif is_training_dataset:
            # Regular training cell type
            subset = ds.to_subset_dataset("train", pert_indices, ctrl_indices)
            self.train_datasets.append(subset)
            counts["train"] = len(subset)

        return counts
