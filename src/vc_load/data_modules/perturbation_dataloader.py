import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import h5py
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from ..dataset.perturbation_dataset import PerturbationDataset
from ..mapping_strategies import BatchMappingStrategy, RandomMappingStrategy
from ..utils.data_utils import (
    GlobalH5MetadataCache,
    generate_onehot_map,
    safe_decode_array,
)
from .samplers import PerturbationBatchSampler
from .tasks import TaskSpec, TaskType

logger = logging.getLogger(__name__)


class MetadataConcatDataset(ConcatDataset):
    """
    ConcatDataset that enforces consistent metadata across all constituent datasets.
    """

    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)
        base = datasets[0].dataset
        self.embed_key = base.embed_key
        self.control_pert = base.control_pert
        self.pert_col = base.pert_col
        self.batch_col = base.batch_col

        for ds in datasets:
            md = ds.dataset
            if (
                md.embed_key != self.embed_key
                or md.control_pert != self.control_pert
                or md.pert_col != self.pert_col
                or md.batch_col != self.batch_col
            ):
                raise ValueError(
                    "All datasets must share the same embed_key, control_pert, pert_col, and batch_col"
                )


class PerturbationDataModule(LightningDataModule):
    """
    A unified data module that sets up train/val/test splits for multiple dataset/celltype
    combos. Allows zero-shot, few-shot tasks, and uses a pluggable mapping strategy
    (batch, random, nearest) to match perturbed cells with control cells.
    """

    def __init__(
        self,
        train_specs: List[TaskSpec],
        test_specs: List[TaskSpec],
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 8,
        few_shot_percent: float = 0.3,
        random_seed: int = 42,  # this should be removed by seed everything
        pert_col: str = "gene",
        batch_col: str = "gem_group",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        embed_key: Optional[Literal["X_hvg", "X_state"]] = None,
        output_space: Literal["gene", "all"] = "gene",
        basal_mapping_strategy: Literal["batch", "random"] = "random",
        n_basal_samples: int = 1,
        should_yield_control_cells: bool = True,
        cell_sentence_len: int = 512,
        **kwargs,  # missing perturbation_features_file  and store_raw_basal for backwards compatibility
    ):
        """
        This class is responsible for serving multiple PerturbationDataset's each of which is specific
        to a dataset/cell type combo. It sets up training, validation, and test splits for each dataset
        and cell type, and uses a pluggable mapping strategy to match perturbed cells with control cells.

        Args:
            train_specs: A list of TaskSpec for training tasks
            test_specs: A list of TaskSpec for testing tasks
            data_dir: Path to the root directory containing the H5 files
            batch_size: Batch size for PyTorch DataLoader
            num_workers: Num workers for PyTorch DataLoader
            few_shot_percent: Fraction of data to use for few-shot tasks
            random_seed: For reproducible splits & sampling
            embed_key: Embedding key or matrix in the H5 file to use for feauturizing cells
            output_space: The output space for model predictions (gene or latent, which uses embed_key)
            basal_mapping_strategy: One of {"batch","random","nearest","ot"}
            n_basal_samples: Number of control cells to sample per perturbed cell
        """
        super().__init__()
        # Experiment level params
        self.train_specs = train_specs
        self.test_specs = test_specs
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.few_shot_percent = few_shot_percent
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

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

        # Optional behaviors
        self.map_controls = kwargs.get("map_controls", True)
        self.perturbation_features_file = kwargs.get("perturbation_features_file")
        self.int_counts = kwargs.get("int_counts", False)
        self.normalize_counts = kwargs.get("normalize_counts", False)
        self.store_raw_basal = kwargs.get("store_raw_basal", False)

        logger.info(
            f"Initializing DataModule: batch_size={batch_size}, workers={num_workers}, "
            f"few_shot_percent={few_shot_percent}, random_seed={random_seed}"
        )

        # Mapping strategy
        self.basal_mapping_strategy = basal_mapping_strategy
        self.mapping_strategy_cls = {
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
            "nearest": BatchMappingStrategy,  # replace with actual nearest strategy
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
        self.train_datasets: List[Dataset] = []
        self.val_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []
        self.fewshot_splits: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

        self.all_perts: Set[str] = set()
        self.pert_onehot_map: Optional[Dict[str, torch.Tensor]] = None
        self.batch_onehot_map: Optional[Dict[str, torch.Tensor]] = None
        self.celltype_onehot_map: Optional[Dict[str, torch.Tensor]] = None

        # Initialize global maps
        self._setup_global_maps()

    def _setup_global_maps(self):
        """
        Set up global one-hot maps for perturbations and batches.
        For perturbations, we scan through all files in all train_specs and test_specs.
        """
        all_perts = set()
        all_batches = set()
        dataset_names = {spec.dataset for spec in (self.train_specs + self.test_specs)}
        for ds_name in dataset_names:
            files = self._find_dataset_files(ds_name)
            for _fname, fpath in files.items():
                with h5py.File(fpath, "r") as f:
                    pert_arr = f["obs/{}/categories".format(self.pert_col)][:]
                    perts = set(safe_decode_array(pert_arr))
                    all_perts.update(perts)

                    try:
                        batch_arr = f["obs/{}/categories".format(self.batch_col)][:]
                    except KeyError:
                        batch_arr = f["obs/{}".format(self.batch_col)][:]
                    batches = set(safe_decode_array(batch_arr))
                    all_batches.update(batches)

        # Create one-hot maps
        if self.perturbation_features_file:
            # Load the custom featurizations from a torch file
            featurization_dict = torch.load(self.perturbation_features_file)
            # Validate that every perturbation in all_perts is in the featurization dict.
            missing = all_perts - set(featurization_dict.keys())
            if missing:
                logger.warning(
                    f"The following perturbations are missing from the featurization file: {missing}"
                )

                # set the missing perturbations to 1D zero vectors of the correct size
                feat_size = next(iter(featurization_dict.values())).shape[0]
                for pert in missing:
                    featurization_dict[pert] = torch.zeros(feat_size)

            logger.info(
                f"Loaded custom perturbation featurizations for {len(featurization_dict)} perturbations."
            )
            self.pert_onehot_map = featurization_dict  # use the custom featurizations
        else:
            # Fall back to default: generate one-hot mapping
            self.pert_onehot_map = generate_onehot_map(all_perts)
        self.batch_onehot_map = generate_onehot_map(all_batches)

    def _group_specs_by_dataset(
        self, specs: List[TaskSpec]
    ) -> Dict[str, List[TaskSpec]]:
        """
        Group TaskSpec objects by dataset name.
        """
        dmap = {}
        for spec in specs:
            dmap.setdefault(spec.dataset, []).append(spec)
        return dmap

    def _setup_global_fewshot_splits(self):
        """Set up fewshot splits using cached metadata."""
        start_time = time.time()
        self.fewshot_splits = {}

        # Create metadata caches for all files
        metadata_caches = {}
        for ds_name in {spec.dataset for spec in self.test_specs}:
            files = self._find_dataset_files(ds_name)
            for fname, fpath in files.items():
                metadata_caches[fpath] = GlobalH5MetadataCache().get_cache(
                    str(fpath),
                    self.pert_col,
                    self.cell_type_key,
                    self.control_pert,
                    self.batch_col,
                )

        # Process each fewshot spec
        fewshot_specs = [s for s in self.test_specs if s.task_type == TaskType.FEWSHOT]
        for spec in fewshot_specs:
            ct = spec.cell_type
            pert_counts = defaultdict(int)

            # Aggregate counts across files
            files = self._find_dataset_files(spec.dataset)
            for fname, fpath in files.items():
                cache = metadata_caches[fpath]
                try:
                    ct_code = np.where(cache.cell_type_categories == ct)[0][0]
                    mask = cache.cell_type_codes == ct_code

                # TODO: Identify exact exception (probably IndexError)
                except:
                    # Skip cell type if not found in this file
                    continue

                pert_codes = cache.pert_codes[mask]
                control_pert_code = cache.control_pert_code
                for pert_code in range(len(cache.pert_categories)):
                    if pert_code != control_pert_code:
                        # Skip control perturbations, count perturbations for this cell type
                        # over all files
                        pert_name = cache.pert_categories[pert_code]
                        pert_counts[pert_name] += np.sum(pert_codes == pert_code)

            # Split perturbations using the same logic as before
            total_cells = sum(pert_counts.values())
            target_train = total_cells * self.few_shot_percent

            train_perts, test_perts = [], []
            current_train = 0

            for pert_name, count in sorted(
                pert_counts.items(), key=lambda x: x[1], reverse=True
            ):
                if abs((current_train + count) - target_train) < abs(
                    current_train - target_train
                ):
                    train_perts.append(pert_name)
                    current_train += count
                else:
                    test_perts.append(pert_name)

            self.fewshot_splits[ct] = {
                "train_perts": set(train_perts),
                "test_perts": set(test_perts),
            }

            end_time = time.time()
            logger.info(
                f"Cell type {ct}: {len(train_perts)} train perts, {len(test_perts)} test perts in {end_time - start_time:.2f} seconds."
            )

    def _setup_datasets(self):
        """
        Set up training datasets with proper handling of zeroshot/fewshot splits.
        Uses H5MetadataCache for faster metadata access.

        TODO: This needs to be changed to explicitly accept a list of perturbations
        """

        # First compute global fewshot splits
        self._setup_global_fewshot_splits()

        # Get zeroshot cell types and store as a set
        zeroshot_cts = {
            spec.cell_type
            for spec in self.test_specs
            if spec.task_type == TaskType.ZEROSHOT
        }

        # Ordered list for reproducible splicing into val / test
        val_test_cts = [
            spec.cell_type
            for spec in self.test_specs
            if spec.task_type == TaskType.ZEROSHOT or spec.task_type == TaskType.FEWSHOT
        ]

        # get a list of all training cell types to iterate over
        train_cts = []
        for spec in self.train_specs:
            if spec.task_type == TaskType.TRAINING:
                if spec.cell_type is None:
                    # add all the cell types from this dataset
                    files = self._find_dataset_files(spec.dataset)
                    for fname, fpath in files.items():
                        with h5py.File(fpath, "r") as f:
                            cats = [
                                x.decode("utf-8")
                                for x in f[f"obs/{self.cell_type_key}/categories"][:]
                            ]
                            train_cts.extend(cats)
                else:
                    # add the specific cell type
                    train_cts.append(spec.cell_type)

        train_cts = set(train_cts) - set(val_test_cts)

        # Choose 40% of the fewshot cell types for validation
        num_val_cts = int(len(val_test_cts) * 0.4)

        # if we don't have enough cell types for validation,
        # just use some of the test data as validation too
        # this should never be an issue if you just stop the model based on number of steps
        if num_val_cts > 0:
            val_cts = set(val_test_cts[:num_val_cts])
        else:
            val_cts = set()

        # keep track of what we add to test
        final_test_cts = set()
        final_val_cts = set()

        # Process each training spec by grouping them by dataset
        train_map = self._group_specs_by_dataset(self.train_specs)
        test_map = self._group_specs_by_dataset(self.test_specs)
        # get the union of the keys in train_map and test_map
        all_datasets = set(train_map.keys()).union(set(test_map.keys()))
        for ds_name in all_datasets:
            files = self._find_dataset_files(ds_name)
            # Outer progress bar: iterate over plates (files) in the dataset
            for fname, fpath in tqdm(
                list(files.items()),
                desc=f"Processing plates in dataset {ds_name}",
                position=0,
                leave=True,
            ):
                logger.info(f"Processing file: {fpath}")

                # Create metadata cache for this file
                cache = GlobalH5MetadataCache().get_cache(
                    str(fpath),
                    self.pert_col,
                    self.cell_type_key,
                    self.control_pert,
                    self.batch_col,
                )

                mapping_kwargs = {
                    "map_controls": self.map_controls,
                }

                # Create base dataset
                ds = PerturbationDataset(
                    name=ds_name,
                    h5_path=fpath,
                    mapping_strategy=self.mapping_strategy_cls(
                        random_state=self.random_seed,
                        n_basal_samples=self.n_basal_samples,
                        **mapping_kwargs,
                    ),
                    embed_key=self.embed_key,
                    pert_onehot_map=self.pert_onehot_map,
                    batch_onehot_map=self.batch_onehot_map,
                    pert_col=self.pert_col,
                    cell_type_key=self.cell_type_key,
                    batch_col=self.batch_col,
                    control_pert=self.control_pert,
                    random_state=self.random_seed,
                    should_yield_control_cells=self.should_yield_control_cells,
                    store_raw_expression=self.store_raw_expression,
                    output_space=self.output_space,
                    store_raw_basal=self.store_raw_basal,
                )

                train_sum = 0
                val_sum = 0
                test_sum = 0

                # Inner progress bar: iterate over cell types within the current plate
                for ct_idx, ct in tqdm(
                    enumerate(cache.cell_type_categories),
                    total=len(cache.cell_type_categories),
                    desc="Processing cell types",
                    position=1,
                    leave=False,
                ):
                    # Create mask for this cell type using cached codes
                    ct_mask = cache.cell_type_codes == ct_idx
                    n_cells = np.sum(ct_mask)

                    if n_cells == 0:
                        continue

                    # Get indices for this cell type
                    ct_indices = np.where(ct_mask)[0]

                    if ct in zeroshot_cts:
                        # First, split controls
                        ctrl_mask = (
                            cache.pert_codes[ct_indices] == cache.control_pert_code
                        )
                        ctrl_indices = ct_indices[ctrl_mask]
                        pert_indices = ct_indices[~ctrl_mask]

                        # For zeroshot, all cells go to val / test, and none go to train
                        if (
                            len(val_cts) == 0
                        ):  # if there are no cell types in validation, let's just put into both val and test
                            val_subset = ds.to_subset_dataset(
                                "val", pert_indices, ctrl_indices
                            )
                            self.val_datasets.append(val_subset)
                            val_sum += len(val_subset)
                            final_val_cts.add(ct)

                            test_subset = ds.to_subset_dataset(
                                "test", pert_indices, ctrl_indices
                            )
                            self.test_datasets.append(test_subset)
                            test_sum += len(test_subset)
                            final_test_cts.add(ct)
                        else:  # otherwise we can split
                            if ct in val_cts:
                                # If this cell type is in the val set, create a val subset
                                val_subset = ds.to_subset_dataset(
                                    "val", pert_indices, ctrl_indices
                                )
                                self.val_datasets.append(val_subset)
                                val_sum += len(val_subset)
                                final_val_cts.add(ct)
                            else:
                                test_subset = ds.to_subset_dataset(
                                    "test", pert_indices, ctrl_indices
                                )
                                self.test_datasets.append(test_subset)
                                test_sum += len(test_subset)
                                final_test_cts.add(ct)

                    elif ct in self.fewshot_splits:
                        # For fewshot, use pre-computed splits
                        train_perts = self.fewshot_splits[ct]["train_perts"]
                        train_pert_codes = np.where(
                            np.isin(cache.pert_categories, list(train_perts))
                        )[0]

                        # Use cached pert names for this cell type
                        ct_pert_codes = cache.pert_codes[ct_indices]
                        control_pert_code = cache.control_pert_code

                        # Split controls
                        ctrl_mask = ct_pert_codes == control_pert_code
                        ctrl_indices = ct_indices[ctrl_mask]

                        # Shuffle controls
                        rng = np.random.default_rng(self.random_seed)
                        ctrl_indices = rng.permutation(ctrl_indices)
                        n_train_controls = int(
                            len(ctrl_indices) * self.few_shot_percent
                        )
                        train_controls = ctrl_indices[:n_train_controls]
                        test_controls = ctrl_indices[n_train_controls:]

                        # Split perturbed cells
                        pert_indices = ct_indices[~ctrl_mask]
                        pert_codes = ct_pert_codes[~ctrl_mask]

                        # Create masks for train/test split using pre-computed pert lists
                        train_pert_mask = np.isin(pert_codes, list(train_pert_codes))
                        train_pert_indices = pert_indices[train_pert_mask]
                        test_pert_indices = pert_indices[~train_pert_mask]

                        # Split train into train/val if we have any training data
                        if len(train_pert_indices) > 0:
                            train_pert_indices = rng.permutation(train_pert_indices)

                            # Create train subset
                            train_subset = ds.to_subset_dataset(
                                "train", train_pert_indices, train_controls
                            )
                            self.train_datasets.append(train_subset)
                            train_sum += len(train_subset)

                        # Create test subset. 40% of the specified fewshot cell types (and a minimum of at least 1)
                        # is used as validation data
                        if len(test_pert_indices) > 0:
                            if (
                                len(val_cts) == 0
                            ):  # if there are no cell types in validation, let's just put into both val and test
                                val_subset = ds.to_subset_dataset(
                                    "val", test_pert_indices, test_controls
                                )
                                self.val_datasets.append(val_subset)
                                val_sum += len(val_subset)
                                final_val_cts.add(ct)

                                test_subset = ds.to_subset_dataset(
                                    "test", test_pert_indices, test_controls
                                )
                                self.test_datasets.append(test_subset)
                                test_sum += len(test_subset)
                                final_test_cts.add(ct)
                            else:  # otherwise we can split
                                if ct in val_cts:
                                    # If this cell type is in the val set, create a val subset
                                    val_subset = ds.to_subset_dataset(
                                        "val", test_pert_indices, test_controls
                                    )
                                    self.val_datasets.append(val_subset)
                                    val_sum += len(val_subset)
                                    final_val_cts.add(ct)
                                else:
                                    test_subset = ds.to_subset_dataset(
                                        "test", test_pert_indices, test_controls
                                    )
                                    self.test_datasets.append(test_subset)
                                    test_sum += len(test_subset)
                                    final_test_cts.add(ct)

                    elif ct in train_cts:
                        # Regular training cell type - no perturbation-based splitting needed
                        # Get all cells for this cell type
                        ct_pert_codes = cache.pert_codes[ct_indices]

                        # Split into control and perturbed
                        control_pert_code = cache.control_pert_code
                        ctrl_mask = ct_pert_codes == control_pert_code
                        ctrl_indices = ct_indices[ctrl_mask]
                        pert_indices = ct_indices[~ctrl_mask]

                        # Randomly shuffle indices
                        rng = np.random.default_rng(self.random_seed)
                        train_ctrl_indices = rng.permutation(ctrl_indices)
                        train_pert_indices = rng.permutation(pert_indices)

                        # Create train subset if we have data
                        if len(train_pert_indices) > 0:
                            train_subset = ds.to_subset_dataset(
                                "train", train_pert_indices, train_ctrl_indices
                            )
                            self.train_datasets.append(train_subset)
                            train_sum += len(train_subset)

                # Write progress information (using tqdm.write to avoid breaking the progress bars)
                tqdm.write(
                    f"Processed {fname} with {cache.n_cells} cells, "
                    f"{train_sum} train, {val_sum} val, {test_sum} test."
                )

                # Clean up file handles
                del cache

            logger.info(f"Finished processing dataset {ds_name}")
            logger.info(f"Using {final_test_cts} as test cell types")
            logger.info(f"Using {final_val_cts} as val cell types")

    def _setup_global_celltype_map(self):
        """
        Create a global cell type map across all H5 files in train+test specs,
        so that each dataset can use a consistent one-hot scheme.
        """
        dataset_files = {spec.dataset for spec in (self.train_specs + self.test_specs)}
        # For each dataset, gather all .h5 files and read out the pert categories
        all_celltypes = set()
        for ds_name in dataset_files:
            files_dict = self._find_dataset_files(ds_name)
            for file_path in files_dict.values():
                with h5py.File(file_path, "r") as f:
                    cats = [
                        x.decode("utf-8")
                        for x in f[f"obs/{self.cell_type_key}/categories"][:]
                    ]
                    all_celltypes.update(cats)

        if len(all_celltypes) == 0:
            raise ValueError("No cell types found across datasets?")

        self.celltype_onehot_map = generate_onehot_map(all_celltypes)
        self.num_celltypes = len(self.celltype_onehot_map)

    def _setup_global_pert_map(self):
        """
        Create a global perturbation map across all H5 files in train+test specs,
        so that each dataset can use a consistent one-hot scheme.
        """
        dataset_files = {spec.dataset for spec in (self.train_specs + self.test_specs)}
        # For each dataset, gather all .h5 files and read out the pert categories
        all_perts = set()
        for ds_name in dataset_files:
            files_dict = self._find_dataset_files(ds_name)
            for file_path in files_dict.values():
                with h5py.File(file_path, "r") as f:
                    cats = [
                        x.decode("utf-8")
                        for x in f[f"obs/{self.pert_col}/categories"][:]
                    ]
                    all_perts.update(cats)

        if len(all_perts) == 0:
            raise ValueError("No perturbations found across datasets?")

        self.pert_onehot_map = generate_onehot_map(all_perts)
        self.num_perts = len(self.pert_onehot_map)

    def _setup_global_batch_map(self):
        """
        Create a global gem_group / batch map across all H5 files in train+test specs,
        so that each dataset can use a consistent one-hot scheme.
        """
        dataset_files = {spec.dataset for spec in (self.train_specs + self.test_specs)}
        # For each dataset, gather all .h5 files and read out the pert categories
        all_batches = set()
        for ds_name in dataset_files:
            files_dict = self._find_dataset_files(ds_name)
            for file_path in files_dict.values():
                with h5py.File(file_path, "r") as f:
                    try:
                        cats = [
                            x.decode("utf-8")
                            for x in f[f"obs/{self.pert_col}/categories"][:]
                        ]
                    except KeyError:
                        cats = f[f"obs/{self.pert_col}"][:].astype(str)
                    all_batches.update(cats)

        if len(all_batches) == 0:
            raise ValueError("No perturbations found across datasets?")

        self.batch_onehot_map = generate_onehot_map(all_batches)
        self.num_batches = len(self.batch_onehot_map)

    def _find_dataset_files(self, dataset_name: str) -> Dict[str, Path]:
        """
        Locate *.h5 files for a given dataset_name inside self.data_dir.
        Return a dict: {cell_type -> file_path}.

        """
        pattern = "*.h5"

        # TODO-Abhi: currently using a debug prefix to use unmerged replogle
        # datasets for testing, change this.
        working_folder = self.data_dir / f"{dataset_name}"
        if not working_folder.exists():
            raise FileNotFoundError(f"No directory named {working_folder}")

        files = sorted(working_folder.glob(pattern))
        cell_types = {}
        for fpath in files:
            # Typically naming: <cell_type>.h5
            cell_type = fpath.stem.split(".")[0]
            cell_types[cell_type] = fpath
        return cell_types

    def _group_specs_by_dataset(
        self, specs: List[TaskSpec]
    ) -> Dict[str, List[TaskSpec]]:
        """
        Group a list of TaskSpecs by dataset name.
        Return dict: {dataset_name -> [TaskSpec, TaskSpec, ...]}
        """
        dmap = {}
        for spec in specs:
            dmap.setdefault(spec.dataset, []).append(spec)
        return dmap

    def _get_test_cell_types(
        self, dataset: str, test_map: Dict[str, List[TaskSpec]]
    ) -> Set[str]:
        """
        Identify cell types used for testing in the given dataset.
        So we can skip them for training, in the zero-shot approach.
        """
        test_cell_types = set()
        if dataset in test_map:
            for spec in test_map[dataset]:
                if spec.task_type in (TaskType.ZEROSHOT, TaskType.FEWSHOT):
                    test_cell_types.add(spec.cell_type)
        return test_cell_types

    def _get_training_cell_types(
        self, specs: List[TaskSpec], files_dict: Dict[str, Path], test_cts: Set[str]
    ) -> Set[str]:
        """
        Determine which cell types are used for training. If a spec has a cell_type
        of None, we use all available (except those in 'test_cts').
        """
        all_cts = set(files_dict.keys())
        train_cts = set()
        for s in specs:
            if s.cell_type is None:
                # means "train on all cts except the test ones"
                train_cts.update(all_cts - test_cts)
            else:
                if s.cell_type not in files_dict:
                    raise ValueError(
                        f"Cell type {s.cell_type} not found in dataset files."
                    )
                # skip if it's in test cts
                if s.cell_type in test_cts:
                    logger.warning(
                        f"Spec says train on cell_type {s.cell_type}, but it is also a test cell type. Skipping."
                    )
                else:
                    train_cts.add(s.cell_type)
        return train_cts

    def setup(self):
        """
        Set up training and test datasets.
        """
        if len(self.train_datasets) == 0:
            logger.info("Setting up training and test datasets...")
            self._setup_datasets()
            logger.info(
                "Done! Train / Val / Test splits: %d / %d / %d",
                len(self.train_datasets),
                len(self.val_datasets),
                len(self.test_datasets),
            )

    def train_dataloader(self):
        if len(self.train_datasets) == 0:
            return None
        use_int_counts = "int_counts" in self.__dict__ and self.int_counts
        collate_fn = lambda batch: PerturbationDataset.collate_fn(
            batch,
            int_counts=use_int_counts,
        )
        ds = MetadataConcatDataset(self.train_datasets)
        use_batch = self.basal_mapping_strategy == "batch"
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=self.batch_size,
            drop_last=False,
            cell_sentence_len=self.cell_sentence_len,
            test=False,
            use_batch=use_batch,
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        if len(self.val_datasets) == 0:
            return None
        use_int_counts = "int_counts" in self.__dict__ and self.int_counts
        collate_fn = lambda batch: PerturbationDataset.collate_fn(
            batch,
            int_counts=use_int_counts,
        )
        ds = MetadataConcatDataset(self.val_datasets)
        use_batch = self.basal_mapping_strategy == "batch"
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=self.batch_size,
            drop_last=False,
            cell_sentence_len=self.cell_sentence_len,
            test=False,
            use_batch=use_batch,
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        if len(self.test_datasets) == 0:
            return None
        use_int_counts = "int_counts" in self.__dict__ and self.int_counts
        collate_fn = lambda batch: PerturbationDataset.collate_fn(
            batch,
            int_counts=use_int_counts,
        )
        ds = MetadataConcatDataset(self.test_datasets)
        use_batch = self.basal_mapping_strategy == "batch"
        # batch size 1 for test - since we don't want to oversample. This logic should probably be cleaned up
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=1,
            drop_last=False,
            cell_sentence_len=self.cell_sentence_len,
            test=True,
            use_batch=use_batch,
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        if len(self.test_datasets) == 0:
            return None
        use_int_counts = "int_counts" in self.__dict__ and self.int_counts
        collate_fn = lambda batch: PerturbationDataset.collate_fn(
            batch,
            int_counts=use_int_counts,
        )
        ds = MetadataConcatDataset(self.test_datasets)
        use_batch = self.basal_mapping_strategy == "batch"
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=self.batch_size,
            drop_last=False,
            cell_sentence_len=self.cell_sentence_len,
            test=True,
            use_batch=use_batch,
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def set_inference_mapping_strategy(self, strategy_cls, **strategy_kwargs):
        """
        Then we create an instance of that strategy and call each test dataset's
        reset_mapping_strategy.
        """
        # normal usage for e.g. NearestNeighborMappingStrategy, etc.
        self.basal_mapping_strategy = strategy_cls.name()
        self.mapping_strategy_cls = strategy_cls
        self.mapping_strategy_kwargs = strategy_kwargs

        for ds_subset in self.test_datasets:
            ds: PerturbationDataset = ds_subset.dataset
            ds.reset_mapping_strategy(
                strategy_cls, stage="inference", **strategy_kwargs
            )

        logger.info(
            "Mapping strategy set to %s for test datasets.", strategy_cls.__name__
        )

    def get_var_dims(self):
        underlying_ds: PerturbationDataset = self.test_datasets[0].dataset
        if self.embed_key:
            input_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
        else:
            input_dim = underlying_ds.n_genes

        gene_dim = underlying_ds.n_genes
        # gene_dim = underlying_ds.get_num_hvgs()
        try:
            hvg_dim = underlying_ds.get_num_hvgs()
        except:
            assert self.embed_key is None, "No X_hvg detected, using raw .X"
            hvg_dim = gene_dim

        if self.embed_key is not None:
            output_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
        else:
            output_dim = input_dim  # training on raw gene expression

        gene_names = underlying_ds.get_gene_names()

        # get the shape of the first value in pert_onehot_map
        pert_dim = next(iter(self.pert_onehot_map.values())).shape[0]
        batch_dim = next(iter(self.batch_onehot_map.values())).shape[0]

        return {
            "input_dim": input_dim,
            "gene_dim": gene_dim,
            "hvg_dim": hvg_dim,
            "output_dim": output_dim,
            "pert_dim": pert_dim,
            "gene_names": gene_names,
            "batch_dim": batch_dim,
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
