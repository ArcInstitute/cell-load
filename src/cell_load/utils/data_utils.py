import logging
import warnings

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
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
                # Get unique values and create proper categories/codes mapping
                unique_values, inverse_indices = np.unique(raw, return_inverse=True)
                self.batch_categories = unique_values.astype(str)
                self.batch_codes = inverse_indices.astype(np.int32)

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

    Note:
        We clone each row from the identity matrix so every tensor owns compact
        storage. This avoids pathological file sizes when maps are serialized
        with pickle (shared-storage tensor views can serialize very poorly).

    Args:
        keys: iterable of hashable items
    Returns:
        dict[key, torch.FloatTensor]: one-hot encoding of length = number of unique keys
    """
    unique_keys = sorted(set(keys))
    num_classes = len(unique_keys)
    # identity matrix rows are one-hot vectors
    onehots = torch.eye(num_classes)
    return {k: onehots[i].clone() for i, k in enumerate(unique_keys)}


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


def suspected_discrete_torch(x: torch.Tensor, n_cells: int = 100) -> bool:
    """Check if data appears to be discrete/raw counts by examining row sums.
    Adapted from validate_normlog function for PyTorch tensors.
    """
    top_n = min(x.shape[0], n_cells)
    rowsum = x[:top_n].sum(dim=1)

    # Check if row sums are integers (fractional part == 0)
    frac_part = rowsum - rowsum.floor()
    return torch.all(torch.abs(frac_part) < 1e-7)


def suspected_log_torch(x: torch.Tensor) -> bool:
    """Check if the data is log transformed already."""
    global_max = x.max()
    return global_max.item() < 15.0


def filter_on_target_knockdown(
    adata: anndata.AnnData,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,
    cell_residual_expression: float = 0.50,
    min_cells: int = 30,
    layer: str | None = None,
    var_gene_name: str = "gene_name",
    verbose: bool = False,
) -> anndata.AnnData:
    """
    1.  Keep perturbations whose *average* knock-down ≥ (1-residual_expression).
    2.  Within those, keep only cells whose knock-down ≥ (1-cell_residual_expression).
    3.  Discard perturbations that have < `min_cells` cells remaining
        after steps 1–2.  Control cells are always preserved.

    Returns
    -------
    AnnData
        Subset of `adata` satisfying all three criteria, with var index set to gene names.
    """
    if var_gene_name not in adata.var.columns:
        raise KeyError(f"Column {var_gene_name!r} not found in adata.var.")

    # pd.Index over the gene name column — used for membership testing and position lookup.
    # get_indexer returns the first occurrence for duplicate gene names, matching
    # the behaviour of var_names_make_unique (first occurrence keeps original name).
    gene_index = pd.Index(adata.var[var_gene_name])

    # ------------------------------------------------------------------
    # 1. Shared setup
    # ------------------------------------------------------------------
    X = adata.layers[layer] if layer is not None else adata.X
    perts = adata.obs[perturbation_column]
    n_cells = adata.n_obs

    control_mask = (perts == control_label).values   # (n_cells,) bool

    unique_perts  = [p for p in perts.unique() if p != control_label]
    matched_perts = [p for p in unique_perts if p in gene_index]
    n_matched     = len(matched_perts)

    if verbose:
        print(f"[input] {n_cells:,} cells | {len(unique_perts)} perturbations (excl. control)")

    if n_matched == 0:
        if verbose:
            print("[no matched perturbations] returning only control cells")
        return adata[control_mask].copy()

    # ------------------------------------------------------------------
    # 2. Extract dense submatrix for all matched genes at once
    #    Shape: (n_cells, n_matched)
    #    get_indexer translates gene names -> integer column positions, equivalent
    #    to what adata[:, matched_perts] does internally when var_names are gene names.
    # ------------------------------------------------------------------
    pert_positions = gene_index.get_indexer(matched_perts)   # (n_matched,) int array

    if sp.issparse(X):
        X_sub = X[:, pert_positions].toarray()   # (n_cells, n_matched)
    else:
        X_sub = np.asarray(X)[:, pert_positions]

    # ------------------------------------------------------------------
    # 3. Control means for all matched genes -- single matrix op
    # ------------------------------------------------------------------
    ctrl_means = X_sub[control_mask].mean(axis=0)   # (n_matched,)

    # ------------------------------------------------------------------
    # 4. Map each cell to its perturbation's column in X_sub
    #    (-1 for control / unmatched)
    # ------------------------------------------------------------------
    pert_to_col     = {p: i for i, p in enumerate(matched_perts)}
    cell_col_mapped = perts.map(pert_to_col)          # NaN for control / unmatched
    cell_col        = np.full(n_cells, -1, dtype=np.int32)
    valid           = ~cell_col_mapped.isna()
    cell_col[valid.values] = cell_col_mapped[valid].values.astype(np.int32)

    matched_cells = cell_col >= 0                     # (n_cells,) bool
    valid_row     = np.where(matched_cells)[0]        # (n_valid,) cell row indices
    valid_col     = cell_col[valid_row]               # (n_valid,) gene column indices into X_sub

    # ------------------------------------------------------------------
    # 5. Stage 1: perturbation-level filter
    #
    #    For each matched pert i:
    #      mean(X_sub[cells_of_pert_i, i]) / ctrl_means[i] < residual_expression
    #
    #    Fancy indexing gives each cell's expression at its own gene in one shot.
    #    np.bincount accumulates per-perturbation sums without a Python loop.
    # ------------------------------------------------------------------
    diag_expr = X_sub[valid_row, valid_col]           # (n_valid,) each cell's own-gene expr

    pert_sums   = np.bincount(valid_col, weights=diag_expr, minlength=n_matched)
    pert_counts = np.bincount(valid_col, minlength=n_matched).astype(np.float64)
    pert_means  = np.where(pert_counts > 0, pert_sums / np.maximum(pert_counts, 1.0), 0.0)

    valid_ctrl  = ~np.isclose(ctrl_means, 0.0)
    kd_ratio    = np.where(valid_ctrl, pert_means / np.where(valid_ctrl, ctrl_means, 1.0), np.inf)
    stage1_pass = valid_ctrl & (kd_ratio < residual_expression)   # (n_matched,) bool

    if verbose:
        cells_s1   = int(control_mask.sum()) + int(stage1_pass[valid_col].sum())
        perts_s1   = len(unique_perts) - int(stage1_pass.sum())
        print(f"[stage 1 (pert avg filter)] removed {n_cells - cells_s1:,} cells | {perts_s1} perturbations")
        prev_cells = cells_s1

    # ------------------------------------------------------------------
    # 6. Stage 2: cell-level filter
    #
    #    For each cell whose pert passed stage 1:
    #      X_sub[cell, pert_col] / ctrl_means[pert_col] < cell_residual_expression
    #
    #    Fancy indexing (rows=passed cell indices, cols=their pert column) handles
    #    the entire stage in three numpy operations.
    # ------------------------------------------------------------------
    passed_sel = stage1_pass[valid_col]               # (n_valid,) bool
    passed_row = valid_row[passed_sel]                # (n_passed,) cell row indices
    passed_col = valid_col[passed_sel]                # (n_passed,) gene column indices into X_sub

    expr_vals           = X_sub[passed_row, passed_col]      # (n_passed,)
    ctrl_means_per_cell = ctrl_means[passed_col]              # (n_passed,)

    nonzero_ctrl = ~np.isclose(ctrl_means_per_cell, 0.0)
    cell_keep    = np.zeros(len(passed_row), dtype=bool)
    cell_keep[nonzero_ctrl] = (
        expr_vals[nonzero_ctrl] / ctrl_means_per_cell[nonzero_ctrl] < cell_residual_expression
    )

    keep_mask = control_mask.copy()
    keep_mask[passed_row] = cell_keep

    if verbose:
        cells_s2       = int(keep_mask.sum())
        perts_after_s1 = set(np.array(matched_perts)[stage1_pass])
        perts_after_s2 = set(perts.values[keep_mask & matched_cells])
        print(f"[stage 2 (cell filter)    ] removed {prev_cells - cells_s2:,} cells | {len(perts_after_s1 - perts_after_s2)} perturbations")
        prev_cells = cells_s2

    # ------------------------------------------------------------------
    # 7. Stage 3: minimum cells per perturbation
    # ------------------------------------------------------------------
    kept_mask = keep_mask & matched_cells
    kept_row  = np.where(kept_mask)[0]
    kept_col  = cell_col[kept_row]

    if len(kept_col) > 0:
        pert_kept_counts = np.bincount(kept_col, minlength=n_matched)
        drop_pert        = pert_kept_counts < min_cells
        cell_drop        = drop_pert[kept_col]
        keep_mask[kept_row[cell_drop]] = False

    if verbose:
        cells_s3       = int(keep_mask.sum())
        perts_removed  = int((drop_pert & (pert_kept_counts > 0)).sum()) if len(kept_col) > 0 else 0
        print(f"[stage 3 (min cells filter)] removed {prev_cells - cells_s3:,} cells | {perts_removed} perturbations")
        out_perts = len(set(perts.values[keep_mask]) - {control_label})
        print(f"[output] {cells_s3:,} cells | {out_perts} perturbations (excl. control)")

    # ------------------------------------------------------------------
    # 8. Build output
    #    Subset first, then copy -- avoids copying the full expression matrix.
    # ------------------------------------------------------------------
    result = adata[keep_mask].copy()
    if var_gene_name in result.var.columns:
        result.var.index = result.var[var_gene_name].astype(str)
        result.var_names_make_unique()
    return result