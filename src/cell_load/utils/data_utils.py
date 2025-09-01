import logging
import warnings

import anndata
import h5py
import numpy as np
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


def _mean(expr) -> float:
    """Return the mean of a dense or sparse 1-D/2-D slice."""
    if sp.issparse(expr):
        return float(expr.mean())
    return float(np.asarray(expr).mean())


def is_on_target_knockdown(
    adata: anndata.AnnData,
    target_gene: str,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,
    layer: str | None = None,
) -> bool:
    """
    True ⇢ average expression of *target_gene* in perturbed cells is below
    `residual_expression` × (average expression in control cells).

    Parameters
    ----------
    adata : AnnData
    target_gene : str
        Gene symbol to check.
    perturbation_column : str, default "gene"
        Column in ``adata.obs`` holding perturbation identities.
    control_label : str, default "non-targeting"
        Category in *perturbation_column* marking control cells.
    residual_expression : float, default 0.30
        Residual fraction (0‒1). 0.30 → 70 % knock-down.
    layer : str | None, optional
        Use this matrix in ``adata.layers`` instead of ``adata.X``.

    Raises
    ------
    KeyError
        *target_gene* not present in ``adata.var_names``.
    ValueError
        No perturbed cells for *target_gene*, or control mean is zero.

    Returns
    -------
    bool
    """
    if target_gene == control_label:
        # Never evaluate the control itself
        return False

    if target_gene not in adata.var_names:
        raise KeyError(f"Gene {target_gene!r} not found in `adata.var_names`.")

    gene_idx = adata.var_names.get_loc(target_gene)
    X = adata.layers[layer] if layer is not None else adata.X

    control_cells = adata.obs[perturbation_column] == control_label
    perturbed_cells = adata.obs[perturbation_column] == target_gene

    if not perturbed_cells.any():
        raise ValueError(f"No cells labelled with perturbation {target_gene!r}.")

    try:
        control_mean = _mean(X[control_cells, gene_idx])
    except:
        control_cells = (adata.obs[perturbation_column] == control_label).values
        control_mean = _mean(X[control_cells, gene_idx])

    if control_mean == 0:
        raise ValueError(
            f"Mean {target_gene!r} expression in control cells is zero; "
            "cannot compute knock-down ratio."
        )

    try:
        perturbed_mean = _mean(X[perturbed_cells, gene_idx])
    except:
        perturbed_cells = (adata.obs[perturbation_column] == target_gene).values
        perturbed_mean = _mean(X[perturbed_cells, gene_idx])

    knockdown_ratio = perturbed_mean / control_mean
    return knockdown_ratio < residual_expression


def filter_on_target_knockdown(
    adata: anndata.AnnData,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,  # perturbation-level threshold
    cell_residual_expression: float = 0.50,  # cell-level threshold
    min_cells: int = 30,  # **NEW**: minimum cells/perturbation
    layer: str | None = None,
    var_gene_name: str = "gene_name",
) -> anndata.AnnData:
    """
    1.  Keep perturbations whose *average* knock-down ≥ (1-residual_expression).
    2.  Within those, keep only cells whose knock-down ≥ (1-cell_residual_expression).
    3.  Discard perturbations that have < `min_cells` cells remaining
        after steps 1–2.  Control cells are always preserved.

    Returns
    -------
    AnnData
        View of `adata` satisfying all three criteria.
    """
    # --- prep ---
    adata_ = set_var_index_to_col(adata.copy(), col=var_gene_name)
    X = adata_.layers[layer] if layer is not None else adata_.X
    perts = adata_.obs[perturbation_column]
    control_cells = (perts == control_label).values

    # ---------- stage 1: perturbation filter ----------
    perts_to_keep = [control_label]  # always keep controls
    for pert in perts.unique():
        if pert == control_label:
            continue
        if is_on_target_knockdown(
            adata_,
            target_gene=pert,
            perturbation_column=perturbation_column,
            control_label=control_label,
            residual_expression=residual_expression,
            layer=layer,
        ):
            perts_to_keep.append(pert)

    # ---------- stage 2: cell filter ----------
    keep_mask = np.zeros(adata_.n_obs, dtype=bool)
    keep_mask[control_cells] = True  # retain all controls

    # cache control means to avoid recomputation
    control_mean_cache: dict[str, float] = {}

    for pert in perts_to_keep:
        if pert == control_label:
            continue

        if pert not in adata_.var_names:
            continue

        gene_idx = adata_.var_names.get_loc(pert)

        # control mean for this gene
        if pert not in control_mean_cache:
            try:
                ctrl_mean = _mean(X[control_cells, gene_idx])
            except:
                print(control_cells.shape, control_cells)
                print(gene_idx)
                print(X[control_cells, gene_idx].shape)
            if ctrl_mean == 0:
                raise ValueError(
                    f"Mean {pert!r} expression in control cells is zero; "
                    "cannot compute knock-down ratio."
                )
            control_mean_cache[pert] = ctrl_mean
        else:
            ctrl_mean = control_mean_cache[pert]

        pert_cells = (perts == pert).values
        # FIX: Replace .A1 with .toarray().flatten() for scipy sparse matrices
        expr_vals = (
            X[pert_cells, gene_idx].toarray().flatten()
            if sp.issparse(X)
            else X[pert_cells, gene_idx]
        )
        ratios = expr_vals / ctrl_mean
        keep_mask[pert_cells] = ratios < cell_residual_expression

    # ---------- stage 3: minimum-cell filter ----------
    for pert in perts.unique():
        if pert == control_label:
            continue
        # cells of this perturbation *still* kept after stages 1-2
        pert_mask = (perts == pert).values & keep_mask
        if pert_mask.sum() < min_cells:
            keep_mask[pert_mask] = False  # drop them

    # return view with all criteria satisfied
    return adata_[keep_mask]


def set_var_index_to_col(
    adata: anndata.AnnData, col: str = "col", copy: bool = True
) -> anndata.AnnData:
    """
    Set `adata.var` index to the values in the specified column, allowing non-unique indices.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to modify.
    col : str
        Column in `adata.var` to use as the new index.

    Raises
    ------
    KeyError
        If the specified column does not exist in `adata.var`.
    """
    if copy:
        adata = adata.copy()

    if col not in adata.var.columns:
        raise KeyError(f"Column {col!r} not found in adata.var.")

    adata.var.index = adata.var[col].astype("str")
    adata.var_names_make_unique()
    return adata
