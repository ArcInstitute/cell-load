import random
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..dataset import PerturbationDataset

from .mapping_strategies import BaseMappingStrategy

# Set up logger
logger = logging.getLogger(__name__)


class RandomMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control cell(s) drawn from the same plate.
    We ensure that only control cells with the same cell type
    as the perturbed cell are considered.

    Args:
        cache_perturbation_control_pairs (bool): If True, cache perturbation-control pairs
            at the start of training and reuse them. If False, sample new control cells
            for each perturbed cell every time (original behavior). Default is False.
    """

    def __init__(
        self,
        name="random",
        random_state=42,
        n_basal_samples=1,
        cache_perturbation_control_pairs=False,
        use_consecutive_loading=False,
        **kwargs,
    ):
        super().__init__(name, random_state, n_basal_samples, **kwargs)

        self.cache_perturbation_control_pairs = cache_perturbation_control_pairs
        self.use_consecutive_loading = use_consecutive_loading

        if self.cache_perturbation_control_pairs:
            logger.info(
                f"RandomMappingStrategy initialized with cache_perturbation_control_pairs=True (random_state={random_state}, n_basal_samples={n_basal_samples})"
            )
            logger.info(
                f"Warning: If using n_basal_samples > 1, use the original behavior by setting cache_perturbation_control_pairs=False"
            )
        if self.use_consecutive_loading:
            logger.info(
                "RandomMappingStrategy initialized with use_consecutive_loading=True; "
                "control mappings will be assigned in file order."
            )

        # Map cell type -> list of control indices.
        self.split_control_pool = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }

        # Fixed mapping from perturbed_idx -> list of control indices.
        # Only used when cache_perturbation_control_pairs=True
        self.split_control_mapping: dict[str, dict[int, list[int]]] = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }

        # Initialize Python's random module with the same seed
        self.rng = random.Random(random_state)

    def name():
        return "random"

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        For the given split, group all control indices by their cell type.
        We assume that if a filter is provided in the dataset then all indices belong to the same cell type;
        but if no filter was applied, then this grouping is necessary.

        If cache_perturbation_control_pairs is True, also create a fixed mapping from
        perturbed_idx -> list of control indices.
        """

        all_indices = np.concatenate([perturbed_indices, control_indices])
        # Get cell types for all control indices
        cell_types = dataset.get_all_cell_types(control_indices)

        # Group by cell type and store the control indices
        for ct in np.unique(cell_types):
            ct_mask = cell_types == ct
            ct_indices = control_indices[ct_mask]

            if ct not in self.split_control_pool[split]:
                self.split_control_pool[split][ct] = list(ct_indices)
            else:
                self.split_control_pool[split][ct].extend(ct_indices)

        build_mapping = self.cache_perturbation_control_pairs or self.use_consecutive_loading
        if build_mapping:
            logger.info(
                f"Creating cached perturbation-control mapping for split '{split}' with {len(perturbed_indices)} perturbed cells and {len(control_indices)} control cells"
            )

        # Create a fixed mapping from perturbed_idx -> list of control indices
        # Only if caching is enabled
        if build_mapping:
            pert_groups = {}

            # Group perturbed indices by cell type and perturbation name
            for pert_idx in all_indices:
                pert_cell_type = dataset.get_cell_type(pert_idx)
                pert_name = dataset.get_perturbation_name(pert_idx)
                key = (pert_cell_type, pert_name)

                if key not in pert_groups:
                    pert_groups[key] = []

                pert_groups[key].append(pert_idx)

            # For each cell type / perturbation, assign control cells to each perturbed cell
            for (cell_type, pert_name), pert_idxs_list in pert_groups.items():
                pool = self.split_control_pool[split].get(cell_type, None)

                if not pool:
                    # No control cells available for this cell type
                    for pert_idx in pert_idxs_list:
                        self.split_control_mapping[split][pert_idx] = []
                    continue

                # Calculate total assignments needed for this cell type / perturbation
                total_assignments_needed = len(pert_idxs_list) * self.n_basal_samples
                if self.use_consecutive_loading:
                    pool_arr = np.asarray(pool, dtype=np.int64)
                    if pool_arr.size == 0:
                        control_assignments = []
                    else:
                        repeats = int(
                            np.ceil(total_assignments_needed / pool_arr.size)
                        )
                        control_assignments = np.tile(pool_arr, repeats)[
                            :total_assignments_needed
                        ].tolist()
                else:
                    # Shuffle control pool for random assignment
                    shuffled_pool = pool.copy()
                    self.rng.shuffle(shuffled_pool)

                    # Ensure we have enough controls for all assignments
                    assert len(shuffled_pool) >= total_assignments_needed, (
                        f"Need {total_assignments_needed} controls for {cell_type} / {pert_name} but only have {len(shuffled_pool)}"
                    )

                    # Assign control cells without replacement to this cell type / perturbation
                    control_assignments = shuffled_pool[:total_assignments_needed]

                # Assign control cells to each perturbed cell
                for i, pert_idx in enumerate(pert_idxs_list):
                    start_idx = i * self.n_basal_samples
                    end_idx = start_idx + self.n_basal_samples
                    self.split_control_mapping[split][pert_idx] = control_assignments[
                        start_idx:end_idx
                    ]

            logger.info(
                f"Split '{split}' - Successfully created cached mapping for {len(self.split_control_mapping[split])} perturbed cells"
            )

    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Returns n_basal_samples control indices that are from the same cell type as the perturbed cell.

        If cache_perturbation_control_pairs is True (or use_consecutive_loading is enabled),
        uses the pre-computed mapping. Otherwise, samples new control cells each time
        (original behavior).
        """

        if self.cache_perturbation_control_pairs or self.use_consecutive_loading:
            # Use cached mapping
            control_idxs = self.split_control_mapping[split][perturbed_idx]
            if len(control_idxs) == 0:
                raise ValueError(
                    f"No control cells found in RandomMappingStrategy for cell type '{dataset.get_cell_type(perturbed_idx)}'"
                )
            return np.array(control_idxs)
        # Sample new control cells each time (original behavior)
        pert_cell_type = dataset.get_cell_type(perturbed_idx)
        pool = self.split_control_pool[split].get(pert_cell_type, None)
        if not pool:
            raise ValueError(
                f"No control cells found in RandomMappingStrategy for cell type '{pert_cell_type}'"
            )
        control_idxs = self.rng.choices(pool, k=self.n_basal_samples)
        return np.array(control_idxs)

    def get_control_index(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ):
        """
        Returns a single control index from the same cell type as the perturbed cell.

        If cache_perturbation_control_pairs is True (or use_consecutive_loading is enabled),
        uses the pre-computed mapping. Otherwise, samples a new control cell each time
        (original behavior).
        """

        if self.cache_perturbation_control_pairs or self.use_consecutive_loading:
            # Use cached mapping
            control_idxs = self.split_control_mapping[split][perturbed_idx]
            if len(control_idxs) == 0:
                return None
            return control_idxs[0]
        # Sample new control cell each time (original behavior)
        pert_cell_type = dataset.get_cell_type(perturbed_idx)
        pool = self.split_control_pool[split].get(pert_cell_type, None)
        if not pool:
            return None
        return self.rng.choice(pool)
