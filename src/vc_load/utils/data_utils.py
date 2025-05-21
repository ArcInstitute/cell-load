import logging
import warnings

import anndata
import h5py
import numpy as np
import torch

from .singleton import Singleton

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class H5MetadataCache:
    """Cache for H5 file metadata to avoid repeated disk reads."""

    def __init__(
        self,
        h5_path: str,
        pert_col: str = "drug",
        cell_type_key: str = "cell_name",
        control_pert: str = "DMSO_TF",
        batch_col: str = "sample",
    ):
        """
        Args:
            h5_path: Path to the .h5ad or .h5 file
            pert_col: obs column name for perturbation
            cell_type_key: obs column name for cell type
            control_pert: the perturbation to treat as control
            batch_col: obs column name for batch/plate
        """
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            obs = f["obs"]

            # -- Categories --
            self.pert_categories = safe_decode_array(obs[pert_col]["categories"][:])
            self.cell_type_categories = safe_decode_array(
                obs[cell_type_key]["categories"][:]
            )

            # -- Batch: handle categorical vs numeric storage --
            batch_ds = obs[batch_col]
            if "categories" in batch_ds:
                self.batch_is_categorical = True
                self.batch_categories = safe_decode_array(batch_ds["categories"][:])
                self.batch_codes = batch_ds["codes"][:].astype(np.int32)
            else:
                self.batch_is_categorical = False
                raw = batch_ds[:]
                self.batch_categories = raw.astype(str)
                self.batch_codes = raw.astype(np.int32)

            # -- Codes for pert & cell type --
            self.pert_codes = obs[pert_col]["codes"][:].astype(np.int32)
            self.cell_type_codes = obs[cell_type_key]["codes"][:].astype(np.int32)

            # -- Control mask & counts --
            idx = np.where(self.pert_categories == control_pert)[0]
            if idx.size == 0:
                raise ValueError(
                    f"control_pert='{control_pert}' not found in {pert_col} categories"
                )
            self.control_pert_code = int(idx[0])
            self.control_mask = self.pert_codes == self.control_pert_code

            self.n_cells = len(self.pert_codes)

    def get_batch_names(self, indices: np.ndarray) -> np.ndarray:
        """Return batch labels for the provided cell indices."""
        return self.batch_categories[indices]

    def get_cell_type_names(self, indices: np.ndarray) -> np.ndarray:
        """Return cell‐type labels for the provided cell indices."""
        return self.cell_type_categories[indices]

    def get_pert_names(self, indices: np.ndarray) -> np.ndarray:
        """Return perturbation labels for the provided cell indices."""
        return self.pert_categories[indices]


class GlobalH5MetadataCache(metaclass=Singleton):
    """
    Singleton managing a shared dict of H5MetadataCache instances.
    Keys by h5_path only (same as before).
    """

    def __init__(self):
        self._cache: dict[str, H5MetadataCache] = {}

    def get_cache(
        self,
        h5_path: str,
        pert_col: str = "drug",
        cell_type_key: str = "cell_name",
        control_pert: str = "DMSO_TF",
        batch_col: str = "drug",
    ) -> H5MetadataCache:
        """
        If a cache for this file doesn’t yet exist, create it with the
        given parameters; otherwise return the existing one.
        """
        if h5_path not in self._cache:
            self._cache[h5_path] = H5MetadataCache(
                h5_path, pert_col, cell_type_key, control_pert, batch_col
            )
        return self._cache[h5_path]


def safe_decode_array(arr) -> np.ndarray:
    """
    Decode any byte-strings in `arr` to UTF-8 and cast all entries to Python str.

    Args:
        arr: array-like of bytes or other objects
    Returns:
        np.ndarray[str]: decoded strings
    """
    decoded = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            # decode bytes, ignoring errors
            decoded.append(x.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(x))
    return np.array(decoded, dtype=str)


def generate_onehot_map(keys) -> dict:
    """
    Build a map from each unique key to a fixed-length one-hot torch vector.

    Args:
        keys: iterable of hashable items
    Returns:
        dict[key, torch.FloatTensor]: one-hot encoding of length = number of unique keys
    """
    unique_keys = sorted(set(keys))
    num_classes = len(unique_keys)
    # identity matrix rows are one-hot vectors
    onehots = torch.eye(num_classes)
    return {k: onehots[i] for i, k in enumerate(unique_keys)}


def data_to_torch_X(X):
    """
    Convert input data to a dense torch FloatTensor.

    If passed an AnnData, extracts its .X matrix.
    If the result isn’t a NumPy array (e.g. a sparse matrix), calls .toarray().
    Finally wraps with torch.from_numpy(...).float().

    Args:
        X: anndata.AnnData or array-like (dense or sparse).
    Returns:
        torch.FloatTensor of shape (n_cells, n_features).
    """
    if isinstance(X, anndata.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()


def split_perturbations_by_cell_fraction(
    pert_groups: dict,
    val_fraction: float,
    rng: np.random.Generator = None,
):
    """
    Partition the set of perturbations into two subsets: 'val' vs 'train',
    such that the fraction of total cells in 'val' is as close as possible
    to val_fraction, using a greedy approach.

    Here, pert_groups is a dictionary where the keys are perturbation names
    and the values are numpy arrays of cell indices.

    Returns:
        train_perts: list of perturbation names
        val_perts:   list of perturbation names
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # 1) Compute total # of cells across all perturbations
    total_cells = sum(len(indices) for indices in pert_groups.values())
    target_val_cells = val_fraction * total_cells

    # 2) Create a list of (pert_name, size), then shuffle
    pert_size_list = [(p, len(pert_groups[p])) for p in pert_groups.keys()]
    rng.shuffle(pert_size_list)

    # 3) Greedily add perts to the 'val' subset if it brings us closer to the target
    val_perts = []
    current_val_cells = 0
    for pert, size in pert_size_list:
        new_val_cells = current_val_cells + size

        # Compare how close we'd be to target if we add this perturbation vs. skip it
        diff_if_add = abs(new_val_cells - target_val_cells)
        diff_if_skip = abs(current_val_cells - target_val_cells)

        if diff_if_add < diff_if_skip:
            # Adding this perturbation gets us closer to the target fraction
            val_perts.append(pert)
            current_val_cells = new_val_cells
        # else: skip it; it goes to train

    train_perts = list(set(pert_groups.keys()) - set(val_perts))

    return train_perts, val_perts
