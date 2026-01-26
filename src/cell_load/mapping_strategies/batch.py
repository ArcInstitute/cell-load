import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..dataset import PerturbationDataset

from .mapping_strategies import BaseMappingStrategy

logger = logging.getLogger(__name__)


class BatchMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control(s) drawn from the same batch and cell type.
    If no controls are available in the same batch, falls back to controls from the same cell type.

    This strategy matches the RandomMappingStrategy structure except it groups the control cells
    by the tuple (batch, cell_type) instead of just by cell type.
    """

    def __init__(
        self,
        name="batch",
        random_state=42,
        n_basal_samples=1,
        use_consecutive_loading=False,
        **kwargs,
    ):
        super().__init__(name, random_state, n_basal_samples, **kwargs)
        self.use_consecutive_loading = use_consecutive_loading
        # For each split, store a mapping: {(batch, cell_type): [ctrl_indices]}
        self.split_control_maps = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }
        # Fixed mapping from perturbed_idx -> list of control indices for consecutive loading.
        self.split_control_mapping: dict[str, dict[int, list[int]]] = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }
        if self.use_consecutive_loading:
            logger.info(
                "BatchMappingStrategy initialized with use_consecutive_loading=True; "
                "control mappings will be assigned in file order."
            )

    def name():
        """Name of the mapping strategy."""
        return "batch"

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
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

        if self.use_consecutive_loading:
            self._build_consecutive_mapping(
                dataset, split, perturbed_indices, control_indices
            )

    def _build_consecutive_mapping(
        self,
        dataset: "PerturbationDataset",
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ) -> None:
        """
        Build a fixed mapping from each index to control indices using sequential
        assignment within each (batch, cell_type) pool, with fallback to cell_type pools.
        """
        all_indices = np.concatenate([perturbed_indices, control_indices])

        # Fallback pools by cell type (sorted for deterministic order).
        fallback_pools: dict[str, list[int]] = {}
        for (batch, cell_type), indices in self.split_control_maps[split].items():
            fallback_pools.setdefault(cell_type, []).extend(indices)
        for cell_type, pool in fallback_pools.items():
            fallback_pools[cell_type] = sorted(pool)

        key_offsets: dict[tuple[int, str], int] = {}
        fallback_offsets: dict[str, int] = {}

        for idx in all_indices:
            batch = dataset.get_batch(idx)
            cell_type = dataset.get_cell_type(idx)
            key = (batch, cell_type)
            pool = self.split_control_maps[split].get(key, [])

            if not pool:
                pool = fallback_pools.get(cell_type, [])
                if not pool:
                    self.split_control_mapping[split][idx] = []
                    continue
                offset = fallback_offsets.get(cell_type, 0)
                control_idxs = [
                    pool[(offset + i) % len(pool)] for i in range(self.n_basal_samples)
                ]
                fallback_offsets[cell_type] = offset + self.n_basal_samples
            else:
                offset = key_offsets.get(key, 0)
                control_idxs = [
                    pool[(offset + i) % len(pool)] for i in range(self.n_basal_samples)
                ]
                key_offsets[key] = offset + self.n_basal_samples

            self.split_control_mapping[split][idx] = control_idxs

    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Return n_basal_samples control indices for the perturbed cell that are
        from the same batch and the same cell type.

        If the batch group for the perturbed cell is empty, the method falls back to
        using all control cells from the same cell type (regardless of batch).
        """
        if self.use_consecutive_loading:
            control_idxs = self.split_control_mapping[split].get(perturbed_idx, [])
            if not control_idxs:
                raise ValueError(
                    "No control cells found in BatchMappingStrategy for cell type '{}'".format(
                        dataset.get_cell_type(perturbed_idx)
                    )
                )
            return np.array(control_idxs)

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
        if self.use_consecutive_loading:
            control_idxs = self.split_control_mapping[split].get(perturbed_idx, [])
            if not control_idxs:
                return None
            return control_idxs[0]

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
