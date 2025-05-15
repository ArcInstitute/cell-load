from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BaseMappingStrategy(ABC):
    """
    Abstract base class for mapping a perturbed cell to one or more control cells.
    Each strategy can store internal data structures that assist in retrieving
    control indices for a perturbed cell (e.g. nearest neighbor graphs, OT plans, etc.).
    The main parameters for this class are:
    - random_state: seed for random number generation
    - n_basal_samples: number of control cells to return for each perturbed cell
    """

    def __init__(
        self,
        name=None,
        random_state: int = 42,
        n_basal_samples: int = 1,
        stage: str = "train",
        **kwargs,
    ):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.n_basal_samples = n_basal_samples
        self.name = name
        self.stage = stage
        self.map_controls = kwargs.get("map_controls", False)
        print(
            f"Using {self.name} mapping strategy with {self.map_controls} for map_controls."
        )

    def __setstate__(self, state):
        """
        Custom unpickling behavior to handle backward compatibility.
        This method is called when unpickling an object.
        """
        # First, restore all attributes that were pickled
        self.__dict__.update(state)

        # If the new attribute doesn't exist in the pickled state, set it to a default value
        if not hasattr(self, "map_controls"):
            self.map_controls = False
            logger.info(
                f"Adding missing 'map_controls' attribute to {self.name} mapping strategy."
            )

    @abstractmethod
    def register_split_indices(
        self,
        dataset: "PerturbationDataset",  # a reference to the PerturbationDataset
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        Called once per split (train/val/test) to initialize or compute
        the mapping information for that split.
        """
        pass

    @abstractmethod
    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Returns the control indices for a given perturbed index in a particular split.
        """
        pass

    @abstractmethod
    def get_control_index(self, dataset, split, perturbed_idx) -> Optional[int]:
        pass

    def get_mapped_expressions(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Base implementation where "perturbed_idx" confusingly refers to both control and perturbed cells.

        For control cells:
            - Returns (control_expr, control_expr, control_index) where control_expr is that cell's expression
            and control_index is the index of the mapped control.
        For perturbed cells:
            - Returns (perturbed_expr, control_expr) using get_control_indices()

        If get_basal_raw is True, returns the raw expression of the basal cells as well
        (control_expr, control_expr, control_raw), or (perturbed_expr, control_expr, control_raw) for perturbed cells.
        """

        # Get expression(s) based on embed_key
        if dataset.embed_key:
            control_index = self.get_control_index(dataset, split, perturbed_idx)
            pert_expr = torch.tensor(
                dataset.fetch_obsm_expression(perturbed_idx, dataset.embed_key)
            )
            if control_index is None:
                ctrl_expr = torch.zeros_like(pert_expr)  # default to zero vector
            else:
                ctrl_expr = dataset.fetch_obsm_expression(
                    control_index, dataset.embed_key
                )
            return pert_expr, ctrl_expr, control_index
        else:
            control_index = self.get_control_index(dataset, split, perturbed_idx)
            ctrl_expr = dataset.fetch_gene_expression(control_index)
            pert_expr = dataset.fetch_gene_expression(perturbed_idx)
            return pert_expr, ctrl_expr, control_index
