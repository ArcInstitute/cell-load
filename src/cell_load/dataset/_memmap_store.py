"""GIL-free memmap reader for h5ad sidecar directories.

MemmapStore provides numpy-memmap-based reads that bypass h5py's GIL-locked
HDF5 C library, enabling true parallel I/O across DataLoader workers.

The numba-JIT scatter kernel replaces the pure-Python CSR→dense loop,
giving an additional ~50% speedup on the scatter phase.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numba CSR → dense scatter kernel
# ---------------------------------------------------------------------------

_HAS_NUMBA = False
try:
    import numba

    @numba.njit(cache=True)
    def _scatter_csr_to_dense(
        sorted_rows: np.ndarray,     # int64 — original row indices (sorted)
        indptr_slice: np.ndarray,     # int64 — indptr[row_min : row_max+2]
        block_data: np.ndarray,       # float32 — CSR data for the block
        block_indices: np.ndarray,    # int32 — CSR col indices for the block
        dense: np.ndarray,            # float32 (n_rows, n_cols) — output
        row_min: int,                 # first row index in the block
        base_ptr: int,                # indptr_slice[0] (offset into data/indices)
    ) -> None:
        for i in range(len(sorted_rows)):
            offset = sorted_rows[i] - row_min
            ps = indptr_slice[offset] - base_ptr
            pe = indptr_slice[offset + 1] - base_ptr
            for j in range(ps, pe):
                dense[i, block_indices[j]] = block_data[j]

    _HAS_NUMBA = True
    logger.debug("numba scatter kernel available")
except ImportError:
    pass


def scatter_csr_to_dense_python(
    sorted_rows: np.ndarray,
    indptr_slice: np.ndarray,
    block_data: np.ndarray,
    block_indices: np.ndarray,
    dense: np.ndarray,
    row_min: int,
    base_ptr: int,
) -> None:
    """Pure-Python fallback for the CSR→dense scatter."""
    for i in range(len(sorted_rows)):
        offset = sorted_rows[i] - row_min
        ps = int(indptr_slice[offset] - base_ptr)
        pe = int(indptr_slice[offset + 1] - base_ptr)
        if pe > ps:
            dense[i, block_indices[ps:pe]] = block_data[ps:pe]


def scatter_csr_to_dense(
    sorted_rows: np.ndarray,
    indptr_slice: np.ndarray,
    block_data: np.ndarray,
    block_indices: np.ndarray,
    dense: np.ndarray,
    row_min: int,
    base_ptr: int,
) -> None:
    """Dispatch to numba if available, else pure-Python fallback."""
    if _HAS_NUMBA:
        _scatter_csr_to_dense(
            sorted_rows, indptr_slice, block_data, block_indices,
            dense, row_min, base_ptr,
        )
    else:
        scatter_csr_to_dense_python(
            sorted_rows, indptr_slice, block_data, block_indices,
            dense, row_min, base_ptr,
        )


class MemmapStore:
    """Read-side abstraction for memmap sidecar directories.

    Lazily opens numpy memmaps on first access. Handles ``spawn``-based
    multiprocessing safely via PID checks (re-opens memmaps in child processes).

    Usage::

        store = MemmapStore(Path("c1.mmap"))
        dense = store.fetch_obsm_batch(indices, "X_state")
        dense = store.fetch_csr_batch_dense(indices)
    """

    def __init__(self, mmap_dir: str | Path):
        self.mmap_dir = Path(mmap_dir)
        manifest_path = self.mmap_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {self.mmap_dir}")

        with open(manifest_path) as f:
            self._manifest = json.load(f)

        self.obsm_keys: set[str] = set(self._manifest.get("obsm_keys", []))
        self.has_csr: bool = self._manifest.get("csr", False)
        self.n_cells: int = self._manifest.get("n_cells", 0)

        if self.has_csr:
            shape = self._manifest.get("csr_shape", [0, 0])
            self.csr_n_rows = shape[0]
            self.csr_n_cols = shape[1]

        # Lazy-loaded memmap references
        self._obsm_mmaps: dict[str, np.ndarray] = {}
        self._indptr: np.ndarray | None = None
        self._data: np.ndarray | None = None
        self._indices: np.ndarray | None = None
        self._pid: int | None = None

    def _ensure_open(self) -> None:
        """Re-open memmaps if we're in a new process (spawn safety)."""
        pid = os.getpid()
        if self._pid == pid:
            return
        self._pid = pid
        self._obsm_mmaps.clear()
        self._indptr = None
        self._data = None
        self._indices = None

    def _get_obsm_mmap(self, key: str) -> np.ndarray:
        """Get or lazy-load an obsm memmap."""
        self._ensure_open()
        if key not in self._obsm_mmaps:
            path = self.mmap_dir / f"obsm_{key}.npy"
            if not path.exists():
                raise FileNotFoundError(f"obsm memmap not found: {path}")
            self._obsm_mmaps[key] = np.load(str(path), mmap_mode="r")
        return self._obsm_mmaps[key]

    def _get_csr_mmaps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get or lazy-load CSR memmaps."""
        self._ensure_open()
        if self._indptr is None:
            self._indptr = np.load(str(self.mmap_dir / "X_indptr.npy"), mmap_mode="r")
            self._data = np.load(str(self.mmap_dir / "X_data.npy"), mmap_mode="r")
            self._indices = np.load(str(self.mmap_dir / "X_indices.npy"), mmap_mode="r")
        return self._indptr, self._data, self._indices

    def fetch_obsm_batch(self, indices: np.ndarray, key: str) -> np.ndarray:
        """Fetch rows from an obsm matrix via memmap (GIL-free).

        Args:
            indices: Row indices to fetch.
            key: obsm key (e.g. "X_state", "X_hvg").

        Returns:
            float32 array of shape (len(indices), n_cols).
        """
        mmap = self._get_obsm_mmap(key)
        if indices.size == 0:
            return np.empty((0, mmap.shape[1]), dtype=np.float32)
        # numpy fancy indexing on memmap — triggers page faults, no GIL
        return np.asarray(mmap[indices], dtype=np.float32)

    def fetch_obsm_single(self, idx: int, key: str) -> np.ndarray:
        """Fetch a single row from an obsm matrix."""
        mmap = self._get_obsm_mmap(key)
        return np.asarray(mmap[idx], dtype=np.float32)

    def fetch_csr_batch_dense(self, indices: np.ndarray) -> np.ndarray:
        """Fetch CSR rows and scatter into a dense matrix.

        Uses contiguous-run detection + numba scatter kernel for speed.

        Args:
            indices: Row indices to fetch.

        Returns:
            float32 array of shape (len(indices), n_cols).
        """
        if not self.has_csr:
            raise RuntimeError("No CSR data in this memmap store")

        if indices.size == 0:
            return np.empty((0, self.csr_n_cols), dtype=np.float32)

        indptr, data, col_indices = self._get_csr_mmaps()

        order = np.argsort(indices)
        sorted_rows = indices[order]
        dense_sorted = np.zeros((len(sorted_rows), self.csr_n_cols), dtype=np.float32)

        # Process contiguous runs for optimal memmap access
        run_start = 0
        for i in range(1, len(sorted_rows)):
            if sorted_rows[i] != sorted_rows[i - 1] + 1:
                self._fill_dense_run_mmap(
                    sorted_rows, run_start, i, dense_sorted,
                    indptr, data, col_indices,
                )
                run_start = i

        self._fill_dense_run_mmap(
            sorted_rows, run_start, len(sorted_rows), dense_sorted,
            indptr, data, col_indices,
        )

        # Unsort back to original order
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        return dense_sorted[inv_order]

    @staticmethod
    def _fill_dense_run_mmap(
        sorted_rows: np.ndarray,
        start: int,
        end: int,
        dense_sorted: np.ndarray,
        indptr: np.ndarray,
        data: np.ndarray,
        col_indices: np.ndarray,
    ) -> None:
        if start >= end:
            return

        row_min = int(sorted_rows[start])
        row_max = int(sorted_rows[end - 1])

        # Read contiguous indptr slice — single memmap page fault
        indptr_slice = np.asarray(indptr[row_min : row_max + 2], dtype=np.int64)
        base_ptr = int(indptr_slice[0])
        end_ptr = int(indptr_slice[-1])

        if end_ptr <= base_ptr:
            return

        # Read contiguous data/indices block — single memmap page fault each
        block_data = np.asarray(data[base_ptr:end_ptr], dtype=np.float32)
        block_indices = np.asarray(col_indices[base_ptr:end_ptr], dtype=np.int32)

        # Scatter via numba or Python fallback
        run_rows = np.ascontiguousarray(sorted_rows[start:end])
        run_dense = dense_sorted[start:end]
        scatter_csr_to_dense(
            run_rows, indptr_slice, block_data, block_indices,
            run_dense, row_min, base_ptr,
        )
