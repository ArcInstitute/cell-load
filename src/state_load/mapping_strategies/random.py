import random

import numpy as np

from .mapping_strategies import BaseMappingStrategy


class RandomMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control cell(s) drawn from the same plate.
    We ensure that only control cells with the same cell type
    as the perturbed cell are considered.
    """

    def __init__(self, name="random", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples, **kwargs)

        # Map cell type -> list of control indices.
        self.split_control_pool = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }

        # Initialize Python's random module with the same seed
        random.seed(random_state)

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
        """

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

    # #
    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Returns n_basal_samples control indices that are from the same cell type as the perturbed cell.
        Uses Python's random.choice instead of NumPy's random.choice for better performance.
        """
        # Get the cell type of the perturbed cell
        pert_cell_type = dataset.get_cell_type(perturbed_idx)
        pool = self.split_control_pool[split].get(pert_cell_type, None)

        if pool is None or len(pool) == 0:
            return None

        # Use Python's random.choices which allows replacement by default
        # and returns a list of the specified length
        selected_indices = random.choices(pool, k=self.n_basal_samples)

        # Convert to numpy array for compatibility with the rest of the code
        return np.array(selected_indices)

    def get_control_index(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ):
        """
        Returns a single control index from the same cell type as the perturbed cell.
        Uses Python's random.choice instead of NumPy's random.choice for potentially better performance.
        """
        # Get the cell type of the perturbed cell
        pert_cell_type = dataset.get_cell_type(perturbed_idx)
        pool = self.split_control_pool[split].get(pert_cell_type, None)

        # Return None if there is no pool or the pool is empty
        if not pool:
            return None

        # Use Python's random.choice to select a single item
        return random.choice(pool)
