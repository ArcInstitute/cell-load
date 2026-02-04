import logging
import os
from pathlib import Path

from functools import lru_cache
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from ..mapping_strategies import BaseMappingStrategy
from ..utils.data_utils import (
    GlobalH5MetadataCache,
    safe_decode_array,
    suspected_discrete_torch,
    suspected_log_torch,
)


logger = logging.getLogger(__name__)


class PerturbationDataset(Dataset):
    """
    Dataset class for loading perturbation data from H5 files, handling multiple cell types per plate.
    Each instance serves a single dataset/cell_type combination, with configurable mapping strategies.
    """

    def __init__(
        self,
        name: str,
        h5_path: str | Path,
        mapping_strategy: BaseMappingStrategy,
        pert_onehot_map: dict[str, torch.Tensor] | None = None,
        batch_onehot_map: dict[str, torch.Tensor] | None = None,
        cell_type_onehot_map: dict[str, torch.Tensor] | None = None,
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "gem_group",
        control_pert: str = "non-targeting",
        embed_key: str = "X_hvg",
        store_raw_expression: bool = False,
        random_state: int = 42,
        should_yield_control_cells: bool = True,
        store_raw_basal: bool = False,
        barcode: bool = False,
        additional_obs: list[str] | None = None,
        downsample: float | None = None,
        is_log1p: bool = False,
        cell_sentence_len: int | None = None,
        h5_open_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Initialize a perturbation dataset for a specific dataset-celltype.

        Args:
            name: Identifier for this dataset
            h5_path: Path to H5 file
            mapping_strategy: Instance of BaseMappingStrategy to use
            pert_onehot_map: Optional global pert -> one-hot mapping
            batch_onehot_map: Optional global batch -> one-hot mapping
            pert_col: H5 obs column for perturbations
            cell_type_key: H5 obs column for cell types
            batch_col: H5 obs column for batches
            control_pert: Perturbation treated as control
            embed_key: Key under obsm for embeddings
            store_raw_expression: If True, include raw gene expression
            random_state: Seed for reproducibility
            should_yield_control_cells: Include control cells in output
            store_raw_basal: If True, include raw basal expression
            barcode: If True, include cell barcodes in output
            additional_obs: Optional list of obs column names to include in each sample
            downsample: Fraction of counts to retain via binomial downsampling (only for output_space="all")
            is_log1p: Whether raw counts in X are log1p-transformed (affects downsampling)
            cell_sentence_len: Optional sentence length for consecutive loading batches
            h5_open_kwargs: Optional kwargs to pass to h5py.File (e.g., rdcc_nbytes)
            **kwargs: Additional options (e.g. output_space)
        """
        super().__init__()
        self.name = name
        self.h5_path = Path(h5_path)
        self.rng = np.random.default_rng(random_state)
        self.mapping_strategy = mapping_strategy
        self.pert_onehot_map = pert_onehot_map
        self.batch_onehot_map = batch_onehot_map
        self.cell_type_onehot_map = cell_type_onehot_map
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.store_raw_expression = store_raw_expression
        self.should_yield_control_cells = should_yield_control_cells
        self.store_raw_basal = store_raw_basal
        self.barcode = barcode
        self.output_space = kwargs.get("output_space", "gene")
        if self.output_space not in {"gene", "all", "embedding"}:
            raise ValueError(
                f"output_space must be one of 'gene', 'all', or 'embedding'; got {self.output_space!r}"
            )
        if downsample is None:
            self.downsample = None
        else:
            try:
                downsample = float(downsample)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"downsample must be a float in (0, 1]; got {downsample!r}"
                ) from exc
            if not (0.0 < downsample <= 1.0):
                raise ValueError(f"downsample must be in (0, 1]; got {downsample!r}")
            self.downsample = downsample
        self.is_log1p = bool(is_log1p)
        self.cell_sentence_len = cell_sentence_len
        self.h5_open_kwargs = self._normalize_h5_open_kwargs(h5_open_kwargs)
        self.additional_obs = self._validate_additional_obs(additional_obs)

        # Load metadata cache and open file
        self.metadata_cache = GlobalH5MetadataCache().get_cache(
            str(self.h5_path), pert_col, cell_type_key, control_pert, batch_col
        )
        self.h5_file = None
        self._h5_pid = None
        self._open_h5_file()

        # Load cell barcodes if requested
        if self.barcode:
            self.cell_barcodes = self._load_cell_barcodes()
        else:
            self.cell_barcodes = None

        # Cached categories & masks
        self.pert_categories = self.metadata_cache.pert_categories
        self.cell_type_categories = self.metadata_cache.cell_type_categories
        self.control_mask = self.metadata_cache.control_mask

        # Global indices and counts
        self.all_indices = np.arange(self.metadata_cache.n_cells)
        self.n_cells = len(self.all_indices)
        self.n_genes = self._get_num_genes()

        # Initialize split index containers
        splits = ["train", "train_eval", "val", "test"]
        self.split_perturbed_indices = {s: set() for s in splits}
        self.split_control_indices = {s: set() for s in splits}
        self._init_split_index_cache()

    def _init_split_index_cache(self) -> None:
        self._split_code_to_name = ("train", "train_eval", "val", "test")
        self._split_name_to_code = {
            name: idx for idx, name in enumerate(self._split_code_to_name)
        }
        self._index_to_split_code = np.full(self.n_cells, -1, dtype=np.int8)
        for split, indices in self.split_perturbed_indices.items():
            if indices:
                self._index_to_split_code[list(indices)] = self._split_name_to_code[split]
        for split, indices in self.split_control_indices.items():
            if indices:
                self._index_to_split_code[list(indices)] = self._split_name_to_code[split]

    @staticmethod
    def _parse_env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(float(raw))
        except ValueError:
            logger.warning("Invalid %s=%r; using %d", name, raw, default)
            return default

    @staticmethod
    def _parse_env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
            return default

    def _default_h5_open_kwargs(self) -> dict:
        rdcc_nbytes = self._parse_env_int(
            "CELL_LOAD_H5_RDCC_NBYTES", 64 * 1024 * 1024
        )
        rdcc_nslots = self._parse_env_int("CELL_LOAD_H5_RDCC_NSLOTS", 1_000_003)
        rdcc_w0 = self._parse_env_float("CELL_LOAD_H5_RDCC_W0", 0.75)
        kwargs = {
            "rdcc_nbytes": rdcc_nbytes,
            "rdcc_nslots": rdcc_nslots,
            "rdcc_w0": rdcc_w0,
        }
        return self._sanitize_h5_open_kwargs(kwargs)

    @staticmethod
    def _sanitize_h5_open_kwargs(kwargs: dict) -> dict:
        cleaned = {}
        rdcc_nbytes = kwargs.get("rdcc_nbytes")
        if rdcc_nbytes is not None and rdcc_nbytes > 0:
            cleaned["rdcc_nbytes"] = int(rdcc_nbytes)
        rdcc_nslots = kwargs.get("rdcc_nslots")
        if rdcc_nslots is not None and rdcc_nslots > 0:
            cleaned["rdcc_nslots"] = int(rdcc_nslots)
        rdcc_w0 = kwargs.get("rdcc_w0")
        if rdcc_w0 is not None and 0.0 <= float(rdcc_w0) <= 1.0:
            cleaned["rdcc_w0"] = float(rdcc_w0)
        return cleaned

    def _normalize_h5_open_kwargs(self, h5_open_kwargs: dict | None) -> dict:
        if h5_open_kwargs is None:
            return self._default_h5_open_kwargs()
        return self._sanitize_h5_open_kwargs(h5_open_kwargs)

    def _open_h5_file(self) -> None:
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
        self.h5_file = h5py.File(self.h5_path, "r", **self.h5_open_kwargs)
        self._h5_pid = os.getpid()

    def _ensure_h5_open(self) -> None:
        if (
            self.h5_file is None
            or self._h5_pid != os.getpid()
            or not self.h5_file.id.valid
        ):
            self._open_h5_file()

    def ensure_h5_open(self) -> None:
        self._ensure_h5_open()

    def set_store_raw_expression(self, flag: bool) -> None:
        """
        Enable or disable inclusion of raw gene expression in each sample.
        """
        self.store_raw_expression = flag
        logger.info(f"[{self.name}] store_raw_expression set to {flag}")

    def reset_mapping_strategy(
        self,
        strategy_cls: BaseMappingStrategy,
        stage: str = "train",
        **strategy_kwargs,
    ) -> None:
        """
        Replace the current mapping strategy and re-register existing splits.
        """
        self.mapping_strategy = strategy_cls(**strategy_kwargs)
        self.mapping_strategy.stage = stage
        for split, pert_set in self.split_perturbed_indices.items():
            ctrl_set = self.split_control_indices[split]
            if pert_set and ctrl_set:
                pert_arr = np.array(sorted(pert_set))
                ctrl_arr = np.array(sorted(ctrl_set))
                self.mapping_strategy.register_split_indices(
                    self, split, pert_arr, ctrl_arr
                )

    def __getitem__(self, idx: int):
        """
        Fetch a sample (perturbed + mapped control) by filtered index.

        This returns a dictionary with:
        - pert_cell_emb: the embedding of the perturbed cell (either in gene space or embedding space)
        - ctrl_cell_emb: the control cell's embedding. control cells are chosen by the mapping strategy
        - pert_emb: the one-hot encoding (or other featurization) for the perturbation
        - pert_name: the perturbation name
        - cell_type: the cell type
        - batch: the batch (as an int or string)
        - batch_name: the batch name (as a string)
        - pert_cell_counts: the raw gene expression of the perturbed cell (if store_raw_expression is True)
        - ctrl_cell_counts: the raw gene expression of the control cell (if store_raw_basal is True)
        """
        self._ensure_h5_open()

        # Get the perturbed cell expression, control cell expression, and index of mapped control cell
        file_idx = int(self.all_indices[idx])
        split = self._find_split_for_idx(file_idx)
        pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(
            self, split, file_idx
        )

        # Perturbation info
        pert_code = self.metadata_cache.pert_codes[file_idx]
        pert_name = self.pert_categories[pert_code]
        pert_onehot = (
            self.pert_onehot_map.get(pert_name) if self.pert_onehot_map else None
        )

        # Cell type info
        cell_type = self.cell_type_categories[
            self.metadata_cache.cell_type_codes[file_idx]
        ]
        cell_type_onehot = (
            self.cell_type_onehot_map.get(cell_type)
            if self.cell_type_onehot_map
            else None
        )

        # Batch info
        batch_code = self.metadata_cache.batch_codes[file_idx]
        batch_name = self.metadata_cache.batch_categories[batch_code]
        batch_onehot = (
            self.batch_onehot_map.get(batch_name) if self.batch_onehot_map else None
        )

        sample = {
            "pert_cell_emb": pert_expr,
            "ctrl_cell_emb": ctrl_expr,
            "pert_emb": pert_onehot,
            "pert_name": pert_name,
            "batch_name": batch_name,
            "batch": batch_onehot,
            "cell_type": cell_type,
            "cell_type_onehot": cell_type_onehot,
        }

        # Optionally include raw expressions for the perturbed cell, for training a decoder
        if self.store_raw_expression and self.output_space != "embedding":
            if self.output_space == "gene":
                sample["pert_cell_counts"] = self.fetch_obsm_expression(
                    file_idx, "X_hvg"
                )
            elif self.output_space == "all":
                sample["pert_cell_counts"] = self.fetch_gene_expression(file_idx)

        # Optionally include raw expressions for the control cell
        if self.store_raw_basal and self.output_space != "embedding":
            if self.output_space == "gene":
                sample["ctrl_cell_counts"] = self.fetch_obsm_expression(
                    ctrl_idx, "X_hvg"
                )
            elif self.output_space == "all":
                sample["ctrl_cell_counts"] = self.fetch_gene_expression(ctrl_idx)

        # Optionally include cell barcodes
        if self.barcode and self.cell_barcodes is not None:
            sample["pert_cell_barcode"] = self.cell_barcodes[file_idx]
            sample["ctrl_cell_barcode"] = self.cell_barcodes[ctrl_idx]

        if self.additional_obs:
            for obs_key in self.additional_obs:
                sample[obs_key] = self._fetch_obs_value(file_idx, obs_key)

        return sample

    def __getitems__(self, indices):
        """
        Batch-aware fetch for consecutive loading with batched CSR densification.
        Falls back to per-item access when not applicable.
        """
        self._ensure_h5_open()
        if not self._use_batched_fetch():
            return [self.__getitem__(int(i)) for i in indices]

        idx_arr = np.asarray(indices, dtype=np.int64)
        if idx_arr.size == 0:
            return []

        file_indices = self.all_indices[idx_arr]
        splits = [self._find_split_for_idx(int(i)) for i in file_indices]

        ctrl_indices = []
        missing_ctrl = []
        sentence_len = self.cell_sentence_len
        use_sentence_blocks = (
            sentence_len is not None
            and sentence_len > 0
            and len(file_indices) % sentence_len == 0
            and getattr(self.mapping_strategy, "use_consecutive_loading", False)
            and hasattr(self.mapping_strategy, "_sample_consecutive_controls")
            and hasattr(self.mapping_strategy, "split_control_pool")
        )

        if use_sentence_blocks:
            for start in range(0, len(file_indices), sentence_len):
                sentence_idx = file_indices[start : start + sentence_len]
                split = splits[start]
                cell_type_code = self.get_cell_type_code(sentence_idx[0])
                pool = self.mapping_strategy.split_control_pool[split].get(
                    cell_type_code, None
                )
                if not pool:
                    raise ValueError(
                        f"No control cells found in RandomMappingStrategy for cell type '{self.get_cell_type(sentence_idx[0])}'"
                    )
                block = self.mapping_strategy._sample_consecutive_controls(
                    pool, len(sentence_idx)
                )
                ctrl_indices.extend(block.tolist())
                missing_ctrl.extend([False] * len(block))
        else:
            for file_idx, split in zip(file_indices, splits):
                ctrl_idx = self.mapping_strategy.get_control_index(
                    self, split, int(file_idx)
                )
                if ctrl_idx is None:
                    ctrl_indices.append(-1)
                    missing_ctrl.append(True)
                else:
                    ctrl_indices.append(int(ctrl_idx))
                    missing_ctrl.append(False)

        ctrl_indices_arr = np.asarray(ctrl_indices, dtype=np.int64)
        missing_ctrl_mask = np.asarray(missing_ctrl, dtype=bool)

        if missing_ctrl_mask.any():
            missing_pos = int(np.flatnonzero(missing_ctrl_mask)[0])
            missing_file_idx = int(file_indices[missing_pos])
            if not self.embed_key:
                raise ValueError(
                    f"No control cells found for cell type '{self.get_cell_type(missing_file_idx)}'"
                )
            if (
                (self.store_raw_basal and self.output_space != "embedding")
                or (self.barcode and self.cell_barcodes is not None)
            ):
                raise ValueError(
                    f"No control cells found for cell type '{self.get_cell_type(missing_file_idx)}'"
                )

        if self.embed_key:
            pert_expr_batch = self._fetch_obsm_expression_batch(
                file_indices, self.embed_key
            )
            if missing_ctrl_mask.any():
                ctrl_expr_batch = torch.zeros_like(pert_expr_batch)
                valid_positions = np.flatnonzero(~missing_ctrl_mask)
                if valid_positions.size:
                    valid_ctrl_indices = ctrl_indices_arr[valid_positions]
                    valid_positions_t = torch.from_numpy(
                        valid_positions.astype(np.int64)
                    )
                    ctrl_expr_batch[valid_positions_t] = (
                        self._fetch_obsm_expression_batch(
                            valid_ctrl_indices, self.embed_key
                        )
                    )
            else:
                ctrl_expr_batch = self._fetch_obsm_expression_batch(
                    ctrl_indices_arr, self.embed_key
                )
        else:
            pert_expr_batch = self._fetch_gene_expression_batch(file_indices)
            ctrl_expr_batch = self._fetch_gene_expression_batch(ctrl_indices_arr)

        pert_counts_batch = None
        ctrl_counts_batch = None
        if self.store_raw_expression and self.output_space != "embedding":
            if self.output_space == "gene":
                pert_counts_batch = self._fetch_obsm_expression_batch(
                    file_indices, "X_hvg"
                )
            elif self.output_space == "all":
                if self.embed_key:
                    pert_counts_batch = self._fetch_gene_expression_batch(file_indices)
                else:
                    pert_counts_batch = pert_expr_batch

        if self.store_raw_basal and self.output_space != "embedding":
            if self.output_space == "gene":
                ctrl_counts_batch = self._fetch_obsm_expression_batch(
                    ctrl_indices_arr, "X_hvg"
                )
            elif self.output_space == "all":
                if self.embed_key:
                    ctrl_counts_batch = self._fetch_gene_expression_batch(
                        ctrl_indices_arr
                    )
                else:
                    ctrl_counts_batch = ctrl_expr_batch

        samples = []
        for i, file_idx in enumerate(file_indices):
            pert_expr = pert_expr_batch[i]
            ctrl_expr = ctrl_expr_batch[i]
            ctrl_idx = ctrl_indices_arr[i]

            pert_code = self.metadata_cache.pert_codes[file_idx]
            pert_name = self.pert_categories[pert_code]
            pert_onehot = (
                self.pert_onehot_map.get(pert_name) if self.pert_onehot_map else None
            )

            cell_type = self.cell_type_categories[
                self.metadata_cache.cell_type_codes[file_idx]
            ]
            cell_type_onehot = (
                self.cell_type_onehot_map.get(cell_type)
                if self.cell_type_onehot_map
                else None
            )

            batch_code = self.metadata_cache.batch_codes[file_idx]
            batch_name = self.metadata_cache.batch_categories[batch_code]
            batch_onehot = (
                self.batch_onehot_map.get(batch_name) if self.batch_onehot_map else None
            )

            sample = {
                "pert_cell_emb": pert_expr,
                "ctrl_cell_emb": ctrl_expr,
                "pert_emb": pert_onehot,
                "pert_name": pert_name,
                "batch_name": batch_name,
                "batch": batch_onehot,
                "cell_type": cell_type,
                "cell_type_onehot": cell_type_onehot,
            }

            if pert_counts_batch is not None:
                sample["pert_cell_counts"] = pert_counts_batch[i]

            if ctrl_counts_batch is not None:
                sample["ctrl_cell_counts"] = ctrl_counts_batch[i]

            if self.barcode and self.cell_barcodes is not None:
                sample["pert_cell_barcode"] = self.cell_barcodes[file_idx]
                sample["ctrl_cell_barcode"] = self.cell_barcodes[ctrl_idx]

            if self.additional_obs:
                for obs_key in self.additional_obs:
                    sample[obs_key] = self._fetch_obs_value(file_idx, obs_key)

            samples.append(sample)

        return samples

    def _use_batched_fetch(self) -> bool:
        return bool(getattr(self.mapping_strategy, "use_consecutive_loading", False))

    def _validate_additional_obs(self, additional_obs: list[str] | None) -> list[str]:
        if additional_obs is None:
            return []
        if isinstance(additional_obs, (str, bytes, bytearray)):
            raise TypeError(
                "additional_obs must be a list of obs column names, not a string."
            )
        obs_list = [str(item) for item in additional_obs]
        if len(set(obs_list)) != len(obs_list):
            raise ValueError("additional_obs contains duplicate column names.")
        reserved_keys = {
            "pert_cell_emb",
            "ctrl_cell_emb",
            "pert_emb",
            "pert_name",
            "batch_name",
            "batch",
            "cell_type",
            "cell_type_onehot",
            "pert_cell_counts",
            "ctrl_cell_counts",
            "pert_cell_barcode",
            "ctrl_cell_barcode",
        }
        collision = reserved_keys & set(obs_list)
        if collision:
            raise ValueError(
                f"additional_obs contains reserved keys: {sorted(collision)}"
            )
        return obs_list

    def _fetch_obs_value(self, idx: int, key: str):
        obs = self.h5_file["obs"]
        if key not in obs:
            raise KeyError(f"obs/{key} not found in {self.h5_path}")

        entry = obs[key]
        if isinstance(entry, h5py.Group):
            if "codes" in entry and "categories" in entry:
                code = entry["codes"][idx]
                if isinstance(code, np.ndarray):
                    code = code.item()
                category = entry["categories"][int(code)]
                return self._decode_obs_value(category)
            raise KeyError(f"obs/{key} is a group without categorical codes/categories")

        value = entry[idx]
        return self._decode_obs_value(value)

    def _decode_obs_value(self, value):
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.dtype.kind in {"S", "U", "O"}:
                return safe_decode_array(value).tolist()
            else:
                return value
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            return value.decode("utf-8", errors="ignore")
        return value

    def get_batch(self, idx: int) -> torch.Tensor:
        """
        Get the batch information for a given cell index. Returns a scalar tensor.
        """
        assert self.batch_onehot_map is not None, "No batch onehot map, run setup."
        # Translate row index -> batch code -> batch category name
        batch_code = self.metadata_cache.batch_codes[idx]
        batch_name = self.metadata_cache.batch_categories[batch_code]
        batch = torch.argmax(self.batch_onehot_map[batch_name])
        return batch.item()

    def get_dim_for_obsm(self, key: str) -> int:
        """
        Get the feature dimensionality of obsm data with the specified key (e.g., 'X_uce').
        """
        return self.h5_file[f"obsm/{key}"].shape[1]

    def get_cell_type(self, idx):
        """
        Get the cell type for a given index.
        """
        # Convert idx to int in case it's a tensor or array
        idx = int(idx) if hasattr(idx, "__int__") else idx
        code = self.metadata_cache.cell_type_codes[idx]
        return self.metadata_cache.cell_type_categories[code]

    def get_cell_type_code(self, idx: int) -> int:
        """
        Get the cell type code for a given index.
        """
        idx = int(idx) if hasattr(idx, "__int__") else idx
        return int(self.metadata_cache.cell_type_codes[idx])

    def get_all_cell_types(self, indices):
        """
        Get the cell types for all given indices.
        """
        codes = self.metadata_cache.cell_type_codes[indices]
        return self.metadata_cache.cell_type_categories[codes]

    def get_all_cell_type_codes(self, indices) -> np.ndarray:
        """
        Get the cell type codes for all given indices.
        """
        return self.metadata_cache.cell_type_codes[indices]

    def get_perturbation_name(self, idx):
        """
        Get the perturbation name for a given index.
        """
        # Convert idx to int in case it's a tensor or array
        idx = int(idx) if hasattr(idx, "__int__") else idx
        pert_code = self.metadata_cache.pert_codes[idx]
        return self.metadata_cache.pert_categories[pert_code]

    def get_perturbation_code(self, idx: int) -> int:
        """
        Get the perturbation code for a given index.
        """
        idx = int(idx) if hasattr(idx, "__int__") else idx
        return int(self.metadata_cache.pert_codes[idx])

    def get_all_perturbation_codes(self, indices) -> np.ndarray:
        """
        Get the perturbation codes for all given indices.
        """
        return self.metadata_cache.pert_codes[indices]

    def get_batch_code(self, idx: int) -> int:
        """
        Get the batch code for a given index.
        """
        idx = int(idx) if hasattr(idx, "__int__") else idx
        return int(self.metadata_cache.batch_codes[idx])

    def get_all_batch_codes(self, indices) -> np.ndarray:
        """
        Get the batch codes for all given indices.
        """
        return self.metadata_cache.batch_codes[indices]

    def to_subset_dataset(
        self,
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ) -> Subset:
        """
        Creates a Subset of this dataset that includes only the specified perturbed_indices.
        If `self.should_yield_control_cells` flag is True, the Subset will also yield control cells.

        Args:
            split: Name of the split to create, one of 'train', 'val', 'test', or 'train_eval'
            perturbed_indices: Indices of perturbed cells to include
            control_indices: Indices of control cells to include
        """

        # sort them for stable ordering
        perturbed_indices = np.sort(perturbed_indices)
        control_indices = np.sort(control_indices)

        # Register them in the dataset
        self._register_split_indices(split, perturbed_indices, control_indices)

        # Return a Subset containing perturbed cells and optionally control cells
        if self.should_yield_control_cells:
            all_indices = np.concatenate([perturbed_indices, control_indices])
            return Subset(self, all_indices)
        else:
            return Subset(self, perturbed_indices)

    @lru_cache(
        maxsize=10000
    )  # cache the results of the function; lots of hits for batch mapping since most sentences have repeated cells
    def _fetch_gene_expression_raw(self, idx: int) -> torch.Tensor:
        """
        Fetch raw gene counts for a given cell index.

        Supports both CSR‐encoded storage (via `encoding-type = "csr_matrix"`)
        and dense storage in the 'X' dataset.

        Args:
            idx: row index in the X matrix
        Returns:
            1D FloatTensor of length self.n_genes
        """
        attrs = dict(self.h5_file["X"].attrs)
        if attrs["encoding-type"] == "csr_matrix":
            indptr = self.h5_file["/X/indptr"]
            start_ptr = indptr[idx]
            end_ptr = indptr[idx + 1]
            sub_data = torch.tensor(
                self.h5_file["/X/data"][start_ptr:end_ptr], dtype=torch.float32
            )
            sub_indices = torch.tensor(
                self.h5_file["/X/indices"][start_ptr:end_ptr], dtype=torch.long
            )
            counts = torch.sparse_csr_tensor(
                torch.tensor([0], dtype=torch.long),
                sub_indices,
                sub_data,
                (1, self.n_genes),
            )
            data = counts.to_dense().squeeze()
        else:
            row_data = self.h5_file["/X"][idx]
            data = torch.tensor(row_data, dtype=torch.float32)
        return data

    @lru_cache(
        maxsize=10000
    )  # cache row indices/data separately for sparse downsampling
    def _fetch_gene_expression_csr_row(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        indptr = self.h5_file["/X/indptr"]
        start_ptr = indptr[idx]
        end_ptr = indptr[idx + 1]
        sub_data = np.asarray(self.h5_file["/X/data"][start_ptr:end_ptr])
        sub_indices = np.asarray(self.h5_file["/X/indices"][start_ptr:end_ptr])
        return sub_indices.astype(np.int64), sub_data.astype(np.float32)

    def _maybe_downsample_counts(self, counts: torch.Tensor) -> torch.Tensor:
        if (
            self.downsample is None
            or self.downsample >= 1.0
            or self.output_space != "all"
        ):
            return counts

        counts_np = counts.detach().cpu().numpy()
        if self.is_log1p:
            counts_lin = np.expm1(counts_np)
            counts_int = np.rint(counts_lin).astype(np.int64)
        else:
            counts_int = counts_np.astype(np.int64)
        counts_int = np.maximum(counts_int, 0)
        sampled = self.rng.binomial(counts_int, self.downsample)
        if self.is_log1p:
            sampled = np.log1p(sampled)
        return torch.tensor(sampled, dtype=torch.float32)

    def _maybe_downsample_counts_array(self, counts: np.ndarray) -> np.ndarray:
        if (
            self.downsample is None
            or self.downsample >= 1.0
            or self.output_space != "all"
        ):
            return counts

        if self.is_log1p:
            counts_lin = np.expm1(counts)
            counts_int = np.rint(counts_lin).astype(np.int64)
        else:
            counts_int = counts.astype(np.int64)
        counts_int = np.maximum(counts_int, 0)
        sampled = self.rng.binomial(counts_int, self.downsample)
        if self.is_log1p:
            sampled = np.log1p(sampled)
        return sampled.astype(np.float32)

    def fetch_gene_expression(self, idx: int) -> torch.Tensor:
        """
        Fetch raw gene counts for a given cell index, applying optional downsampling.
        """
        attrs = dict(self.h5_file["X"].attrs)
        if (
            attrs.get("encoding-type") == "csr_matrix"
            and self.downsample is not None
            and self.downsample < 1.0
            and self.output_space == "all"
        ):
            sub_indices, sub_data = self._fetch_gene_expression_csr_row(idx)
            dense = np.zeros(self.n_genes, dtype=np.float32)
            if sub_indices.size:
                if self.is_log1p:
                    counts_lin = np.expm1(sub_data)
                    counts_int = np.rint(counts_lin).astype(np.int64)
                else:
                    counts_int = sub_data.astype(np.int64)
                counts_int = np.maximum(counts_int, 0)
                sampled = self.rng.binomial(counts_int, self.downsample).astype(
                    np.float32
                )
                if self.is_log1p:
                    sampled = np.log1p(sampled)
                dense[sub_indices] = sampled
            return torch.from_numpy(dense)

        data = self._fetch_gene_expression_raw(idx)
        return self._maybe_downsample_counts(data)

    def _fetch_dense_matrix_batch(
        self, ds, indices: np.ndarray, n_cols: int
    ) -> np.ndarray:
        if indices.size == 0:
            return np.empty((0, n_cols), dtype=np.float32)

        order = np.argsort(indices)
        sorted_rows = indices[order]
        dense_sorted = np.zeros((len(sorted_rows), n_cols), dtype=np.float32)

        run_start = 0
        for i in range(1, len(sorted_rows)):
            if sorted_rows[i] != sorted_rows[i - 1] + 1:
                row_start = int(sorted_rows[run_start])
                row_end = int(sorted_rows[i - 1])
                block = np.asarray(ds[row_start : row_end + 1], dtype=np.float32)
                if block.ndim == 1:
                    block = block[:, None]
                dense_sorted[run_start:i] = block
                run_start = i

        row_start = int(sorted_rows[run_start])
        row_end = int(sorted_rows[-1])
        block = np.asarray(ds[row_start : row_end + 1], dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        dense_sorted[run_start:len(sorted_rows)] = block

        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        return dense_sorted[inv_order]

    def _fetch_gene_expression_batch(self, indices: np.ndarray) -> torch.Tensor:
        """
        Fetch raw gene counts for multiple indices at once (CSR fast path).
        """
        if indices.size == 0:
            return torch.empty((0, self.n_genes), dtype=torch.float32)

        attrs = dict(self.h5_file["X"].attrs)
        if attrs.get("encoding-type") != "csr_matrix":
            dense = self._fetch_dense_matrix_batch(
                self.h5_file["/X"], indices, self.n_genes
            )
            dense = self._maybe_downsample_counts_array(dense)
            return torch.from_numpy(dense)

        indptr_ds = self.h5_file["/X/indptr"]
        data_ds = self.h5_file["/X/data"]
        indices_ds = self.h5_file["/X/indices"]

        order = np.argsort(indices)
        sorted_rows = indices[order]
        dense_sorted = np.zeros((len(sorted_rows), self.n_genes), dtype=np.float32)

        run_start = 0
        for i in range(1, len(sorted_rows)):
            if sorted_rows[i] != sorted_rows[i - 1] + 1:
                self._fill_dense_run(
                    sorted_rows,
                    run_start,
                    i,
                    dense_sorted,
                    indptr_ds,
                    data_ds,
                    indices_ds,
                )
                run_start = i

        self._fill_dense_run(
            sorted_rows,
            run_start,
            len(sorted_rows),
            dense_sorted,
            indptr_ds,
            data_ds,
            indices_ds,
        )

        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        dense = dense_sorted[inv_order]
        dense = self._maybe_downsample_counts_array(dense)

        return torch.from_numpy(dense)

    def _fetch_obsm_expression_batch(
        self, indices: np.ndarray, key: str
    ) -> torch.Tensor:
        ds = self.h5_file[f"/obsm/{key}"]
        n_cols = int(ds.shape[1]) if ds.ndim > 1 else 1
        if indices.size == 0:
            return torch.empty((0, n_cols), dtype=torch.float32)
        dense = self._fetch_dense_matrix_batch(ds, indices, n_cols)
        return torch.from_numpy(dense)

    def _fill_dense_run(
        self,
        sorted_rows: np.ndarray,
        start: int,
        end: int,
        dense_sorted: np.ndarray,
        indptr_ds,
        data_ds,
        indices_ds,
    ) -> None:
        if start >= end:
            return

        row_start = int(sorted_rows[start])
        row_end = int(sorted_rows[end - 1])
        indptr_slice = indptr_ds[row_start : row_end + 2]
        base_ptr = int(indptr_slice[0])
        end_ptr = int(indptr_slice[-1])

        if end_ptr <= base_ptr:
            return

        block_data = np.asarray(data_ds[base_ptr:end_ptr], dtype=np.float32)
        block_indices = np.asarray(indices_ds[base_ptr:end_ptr], dtype=np.int64)

        for offset, row in enumerate(sorted_rows[start:end]):
            ptr_start = int(indptr_slice[offset] - base_ptr)
            ptr_end = int(indptr_slice[offset + 1] - base_ptr)
            if ptr_end <= ptr_start:
                continue
            dense_sorted[start + offset, block_indices[ptr_start:ptr_end]] = (
                block_data[ptr_start:ptr_end]
            )

    @lru_cache(maxsize=10000)
    def fetch_obsm_expression(self, idx: int, key: str) -> torch.Tensor:
        """
        Fetch a single row from the /obsm/{key} embedding matrix.

        Args:
            idx: row index in the obsm matrix
            key: name of the obsm dataset (e.g. "X_uce", "X_hvg")
        Returns:
            1D FloatTensor of that embedding
        """
        row_data = self.h5_file[f"/obsm/{key}"][idx]
        return torch.tensor(row_data, dtype=torch.float32)

    def get_gene_names(self, output_space="all") -> list[str]:
        """
        Return the list of gene names from var/gene_name (or its categorical fallback).

        Tries, in order:
        1. var/gene_name directly
        2. var/gene_name/categories + codes
        3. var/_index as last resort
        """

        def _decode(x):
            return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)

        try:
            if (
                "var/gene_name/codes" in self.h5_file
                and "var/gene_name/categories" in self.h5_file
            ):
                gene_codes = self.h5_file["var/gene_name/codes"][:]
                gene_categories = self.h5_file["var/gene_name/categories"][:]
                raw = gene_categories[gene_codes]
            else:
                try:
                    raw = self.h5_file["var/gene_name"][:]
                except:
                    raw = self.h5_file["var/gene_name_index"][:]
            if (
                output_space == "gene"
                and "highly_variable" in self.h5_file["/var"].keys()
            ):
                hvg_mask = self.h5_file["/var/highly_variable"][:]
                raw = raw[hvg_mask]
            elif output_space == "gene":
                uns_key = "uns/hvg_names"
                if uns_key in self.h5_file:
                    hvg_names = self.h5_file[uns_key][:].astype(str)
                    raw = hvg_names
            return [_decode(x) for x in raw]
        except KeyError:
            try:
                cats = self.h5_file["var/gene_name/categories"][:]
                codes = self.h5_file["var/gene_name/codes"][:]
                if (
                    output_space == "gene"
                    and "highly_variable" in self.h5_file["/var"].keys()
                ):
                    hvg_mask = self.h5_file["/var/highly_variable"][:]
                    codes = codes[hvg_mask]
                decoded = [_decode(x) for x in cats]
                return [decoded[i] for i in codes]
            except KeyError:
                # Try var/_index, then var/feature as final fallbacks
                try:
                    fallback = self.h5_file["var/_index"][:]
                except KeyError:
                    fallback = self.h5_file["var/feature"][:]
                if (
                    output_space == "gene"
                    and "highly_variable" in self.h5_file["/var"].keys()
                ):
                    hvg_mask = self.h5_file["/var/highly_variable"][:]
                    fallback = fallback[hvg_mask]
                return [_decode(x) for x in fallback]

    ##############################
    # Static methods
    ##############################
    @staticmethod
    def collate_fn(batch, int_counts=False):
        """
        Optimized collate function with preallocated lists.
        Safely handles normalization when vectors sum to zero.
        """
        # Get batch size
        batch_size = len(batch)

        # Preallocate lists with exact size
        pert_cell_emb_list = [None] * batch_size
        ctrl_cell_emb_list = [None] * batch_size
        pert_emb_list = [None] * batch_size
        pert_name_list = [None] * batch_size
        cell_type_list = [None] * batch_size
        cell_type_onehot_list = [None] * batch_size
        batch_list = [None] * batch_size
        batch_name_list = [None] * batch_size

        # Check if optional fields exist
        has_pert_cell_counts = "pert_cell_counts" in batch[0]
        has_ctrl_cell_counts = "ctrl_cell_counts" in batch[0]
        has_barcodes = "pert_cell_barcode" in batch[0]

        # Preallocate optional lists if needed
        if has_pert_cell_counts:
            pert_cell_counts_list = [None] * batch_size

        if has_ctrl_cell_counts:
            ctrl_cell_counts_list = [None] * batch_size

        if has_barcodes:
            pert_cell_barcode_list = [None] * batch_size
            ctrl_cell_barcode_list = [None] * batch_size

        # Process all items in a single pass
        for i, item in enumerate(batch):
            pert_cell_emb_list[i] = item["pert_cell_emb"]
            ctrl_cell_emb_list[i] = item["ctrl_cell_emb"]
            pert_emb_list[i] = item["pert_emb"]
            pert_name_list[i] = item["pert_name"]
            cell_type_list[i] = item["cell_type"]
            cell_type_onehot_list[i] = item["cell_type_onehot"]
            batch_list[i] = item["batch"]
            batch_name_list[i] = item["batch_name"]

            if has_pert_cell_counts:
                pert_cell_counts_list[i] = item["pert_cell_counts"]

            if has_ctrl_cell_counts:
                ctrl_cell_counts_list[i] = item["ctrl_cell_counts"]

            if has_barcodes:
                pert_cell_barcode_list[i] = item["pert_cell_barcode"]
                ctrl_cell_barcode_list[i] = item["ctrl_cell_barcode"]

        # Create batch dictionary
        batch_dict = {
            "pert_cell_emb": torch.stack(pert_cell_emb_list),
            "ctrl_cell_emb": torch.stack(ctrl_cell_emb_list),
            "pert_emb": torch.stack(pert_emb_list),
            "pert_name": pert_name_list,
            "cell_type": cell_type_list,
            "cell_type_onehot": torch.stack(cell_type_onehot_list),
            "batch": torch.stack(batch_list),
            "batch_name": batch_name_list,
        }

        if has_pert_cell_counts:
            pert_cell_counts = torch.stack(pert_cell_counts_list)

            is_discrete = suspected_discrete_torch(pert_cell_counts)
            is_log = suspected_log_torch(pert_cell_counts)
            already_logged = (not is_discrete) and is_log
            batch_dict["pert_cell_counts"] = pert_cell_counts

            # if already_logged:  # counts are already log transformed
            #     if (
            #         int_counts
            #     ):  # if the user wants to model with raw counts, don't log transform
            #         batch_dict["pert_cell_counts"] = torch.expm1(pert_cell_counts)
            #     else:
            #         batch_dict["pert_cell_counts"] = pert_cell_counts
            # else:
            #     if int_counts:
            #         batch_dict["pert_cell_counts"] = pert_cell_counts
            #     else:
            #         batch_dict["pert_cell_counts"] = torch.log1p(pert_cell_counts)

        if has_ctrl_cell_counts:
            ctrl_cell_counts = torch.stack(ctrl_cell_counts_list)

            is_discrete = suspected_discrete_torch(pert_cell_counts)
            is_log = suspected_log_torch(pert_cell_counts)
            already_logged = (not is_discrete) and is_log
            batch_dict["ctrl_cell_counts"] = ctrl_cell_counts

            # if already_logged:  # counts are already log transformed
            #     if (
            #         int_counts
            #     ):  # if the user wants to model with raw counts, don't log transform
            #         batch_dict["ctrl_cell_counts"] = torch.expm1(ctrl_cell_counts)
            #     else:
            #         batch_dict["ctrl_cell_counts"] = ctrl_cell_counts
            # else:
            #     if int_counts:
            #         batch_dict["ctrl_cell_counts"] = ctrl_cell_counts
            #     else:
            #         batch_dict["ctrl_cell_counts"] = torch.log1p(ctrl_cell_counts)

        if has_barcodes:
            batch_dict["pert_cell_barcode"] = pert_cell_barcode_list
            batch_dict["ctrl_cell_barcode"] = ctrl_cell_barcode_list

        base_keys = {
            "pert_cell_emb",
            "ctrl_cell_emb",
            "pert_emb",
            "pert_name",
            "cell_type",
            "cell_type_onehot",
            "batch",
            "batch_name",
            "pert_cell_counts",
            "ctrl_cell_counts",
            "pert_cell_barcode",
            "ctrl_cell_barcode",
        }
        extra_keys = [key for key in batch[0].keys() if key not in base_keys]
        if extra_keys:

            def _collate_extra(values):
                first = values[0]
                if torch.is_tensor(first):
                    return torch.stack(values)
                if isinstance(first, np.ndarray):
                    if first.shape == ():
                        return torch.tensor(
                            [
                                v.item() if isinstance(v, np.ndarray) else v
                                for v in values
                            ]
                        )
                    if first.dtype.kind in {"S", "U", "O"}:
                        return [
                            safe_decode_array(v).tolist()
                            if isinstance(v, np.ndarray)
                            else v
                            for v in values
                        ]
                    return torch.as_tensor(np.stack(values))
                if isinstance(first, (np.generic, int, float, bool)):
                    return torch.tensor(
                        [v.item() if isinstance(v, np.generic) else v for v in values]
                    )
                if isinstance(first, (bytes, bytearray, np.bytes_)):
                    return [
                        v.decode("utf-8", errors="ignore")
                        if isinstance(v, (bytes, bytearray, np.bytes_))
                        else v
                        for v in values
                    ]
                return values

            for key in extra_keys:
                batch_dict[key] = _collate_extra([item[key] for item in batch])

        return batch_dict

    def _register_split_indices(
        self, split: str, perturbed_indices: np.ndarray, control_indices: np.ndarray
    ):
        """
        Register which cell indices belong to the perturbed vs. control set for
        a given split.

        These are passed to the mapping strategy to let it build its internal structures as needed.
        """
        if split not in self.split_perturbed_indices:
            raise ValueError(f"Invalid split {split}")

        # update them in the dataset
        self.split_perturbed_indices[split] |= set(perturbed_indices)
        self.split_control_indices[split] |= set(control_indices)
        if not hasattr(self, "_index_to_split_code"):
            self._init_split_index_cache()
        code = self._split_name_to_code[split]
        if len(perturbed_indices) > 0:
            self._index_to_split_code[perturbed_indices] = code
        if len(control_indices) > 0:
            self._index_to_split_code[control_indices] = code

        # forward these to the mapping strategy
        self.mapping_strategy.register_split_indices(
            self, split, perturbed_indices, control_indices
        )

    def _find_split_for_idx(self, idx: int) -> str | None:
        """Utility to find which split (train/val/test) this idx belongs to."""
        if hasattr(self, "_index_to_split_code"):
            code = int(self._index_to_split_code[idx])
            if code >= 0:
                return self._split_code_to_name[code]
            return None
        for s in self.split_perturbed_indices.keys():
            if (
                idx in self.split_perturbed_indices[s]
                or idx in self.split_control_indices[s]
            ):
                return s
        return None

    def _get_num_genes(self) -> int:
        """Return the number of genes in the X matrix."""
        try:
            # Try to get shape directly from metadata
            n_cols = self.h5_file["X"].attrs["shape"][1]
        except KeyError:
            try:
                # Fallback: if not stored, try the standard dataset shape
                n_cols = self.h5_file["X"].shape[1]
            except Exception:
                # Final fallback: if stored as sparse but shape isn't available, compute from indices
                try:
                    indices = self.h5_file["X/indices"][:]
                    n_cols = indices.max() + 1
                except KeyError:
                    n_cols = self.h5_file["obsm/X_hvg"].shape[1]
        return n_cols

    def get_num_hvgs(self) -> int:
        """Return the number of highly variable genes in the obsm matrix."""
        try:
            return self.h5_file["obsm/X_hvg"].shape[1]
        except:
            return 0

    def _get_num_cells(self) -> int:
        """Return the total number of cells in the file."""
        try:
            n_rows = self.h5_file["X"].shape[0]
        except Exception:
            try:
                # If stored as sparse
                indptr = self.h5_file["X/indptr"][:]
                n_rows = len(indptr) - 1
            except Exception:
                # if this also fails, fall back to obsm
                n_rows = self.h5_file["obsm/X_hvg"].shape[0]
        return n_rows

    def get_pert_name(self, idx: int) -> str:
        """Get perturbation name for a given index."""
        return self.metadata_cache.pert_names[idx]

    def __len__(self) -> int:
        """
        Return number of cells in the dataset
        """
        return self.n_cells

    def __getstate__(self):
        """
        Return a dictionary of this dataset's state without the open h5 file object.
        """
        # Copy the object's dict
        state = self.__dict__.copy()
        # Remove the open file object if it exists
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
        state.pop("h5_file", None)
        state.pop("_h5_pid", None)
        return state

    def __setstate__(self, state):
        """
        Reconstruct the dataset after unpickling. Re-open the HDF5 file by path.
        """
        # TODO-Abhi: remove this before release
        self.__dict__.update(state)
        if not hasattr(self, "h5_open_kwargs"):
            self.h5_open_kwargs = self._normalize_h5_open_kwargs(None)
        self.h5_file = None
        self._h5_pid = None
        self._open_h5_file()
        self.metadata_cache = GlobalH5MetadataCache().get_cache(
            str(self.h5_path),
            self.pert_col,
            self.cell_type_key,
            self.control_pert,
            self.batch_col,
        )
        if not hasattr(self, "_index_to_split_code"):
            self._init_split_index_cache()

    def _load_cell_barcodes(self) -> np.ndarray:
        """
        Load cell barcodes from obs/_index in the H5 file.

        Returns:
            np.ndarray: Array of cell barcode strings
        """
        try:
            # Try to load from obs/_index (AnnData's default storage for obs index)
            barcodes = self.h5_file["obs/_index"][:]
            # Decode bytes to strings if necessary
            decoded_barcodes = []
            for barcode in barcodes:
                if isinstance(barcode, (bytes, bytearray)):
                    decoded_barcodes.append(barcode.decode("utf-8", errors="ignore"))
                else:
                    decoded_barcodes.append(str(barcode))
            return np.array(decoded_barcodes, dtype=str)
        except KeyError:
            # If obs/_index doesn't exist, try obs/_index/categories and codes
            try:
                barcode_categories = self.h5_file["obs/_index/categories"][:]
                barcode_codes = self.h5_file["obs/_index/codes"][:]
                decoded_categories = []
                for cat in barcode_categories:
                    if isinstance(cat, (bytes, bytearray)):
                        decoded_categories.append(cat.decode("utf-8", errors="ignore"))
                    else:
                        decoded_categories.append(str(cat))
                return np.array(
                    [decoded_categories[i] for i in barcode_codes], dtype=str
                )
            except KeyError:
                # If no barcode information is available, generate generic ones
                logger.warning(
                    f"No cell barcode information found in {self.h5_path}. Generating generic barcodes."
                )
                return np.array(
                    [f"cell_{i:06d}" for i in range(self.metadata_cache.n_cells)],
                    dtype=str,
                )
