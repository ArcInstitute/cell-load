import logging

import numpy as np

from .mapping_strategies import BaseMappingStrategy

logger = logging.getLogger(__name__)


class BatchMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control(s) drawn from the same batch and cell type.
    If no controls are available in the same batch, falls back to controls from the same cell type.

    This strategy matches the RandomMappingStrategy structure except it groups the control cells
    by the tuple (batch, cell_type) instead of just by cell type.
    """

    def __init__(self, name="batch", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples, **kwargs)
        # For each split, store a mapping: {(batch, cell_type): [ctrl_indices]}
        self.split_control_maps = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }

    def name():
        """Name of the mapping strategy."""
        return "batch"

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        _perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        Build a map from (batch, cell_type) to control indices for the given split.
        For each control cell, we retrieve both its batch and cell type, using that pair as the key.
        """
        for idx in control_indices:
            batch = dataset.get_batch(idx)
            cell_type = dataset.get_cell_type(idx)
            key = (batch, cell_type)
            if key not in self.split_control_maps[split]:
                self.split_control_maps[split][key] = []
            self.split_control_maps[split][key].append(idx)

    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Return n_basal_samples control indices for the perturbed cell that are
        from the same batch and the same cell type.

        If the batch group for the perturbed cell is empty, the method falls back to
        using all control cells from the same cell type (regardless of batch).
        """
        batch = dataset.get_batch(perturbed_idx)
        cell_type = dataset.get_cell_type(perturbed_idx)
        key = (batch, cell_type)
        pool = self.split_control_maps[split].get(key, [])

        if not pool:
            # Fallback: If no controls exist in this batch, select from all controls with the same cell type.
            pool = []
            for (b, ct), indices in self.split_control_maps[split].items():
                if ct == cell_type:
                    pool.extend(indices)

        if not pool:
            raise ValueError(
                "No control cells found in BatchMappingStrategy for cell type '{}'".format(
                    cell_type
                )
            )

        return self.rng.choice(pool, size=self.n_basal_samples, replace=True)

    def get_control_index(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ):
        """
        Returns a single control index for the perturbed cell.
        This method first attempts to select from controls in the same batch and cell type.
        If no controls are present in the same batch, it falls back to all controls from the same cell type.
        """
        batch = dataset.get_batch(perturbed_idx)
        cell_type = dataset.get_cell_type(perturbed_idx)
        key = (batch, cell_type)
        pool = self.split_control_maps[split].get(key, [])

        if not pool:
            # Fallback: select from controls that are of the same cell type regardless of batch.
            pool = []
            for (b, ct), indices in self.split_control_maps[split].items():
                if ct == cell_type:
                    pool.extend(indices)

        if not pool:
            return None

        return self.rng.choice(pool)
