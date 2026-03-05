"""Convert h5ad files to numpy memmap sidecar directories for GIL-free I/O.

Creates a `.mmap/` directory next to each `.h5ad` file containing:
  - obsm_<key>.npy   : dense obsm matrices (float32, C-contiguous)
  - X_indptr.npy     : CSR indptr array (int64)
  - X_data.npy       : CSR data array (float32)
  - X_indices.npy    : CSR column indices (int32)
  - manifest.json    : shapes, dtypes, obsm keys

Usage:
    from cell_load.dataset._memmap_convert import convert_h5ad_to_memmap, convert_directory
    convert_h5ad_to_memmap("/data/train/c1.h5ad")
    convert_directory("/data/train/")
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 10_000  # rows per streaming chunk


def convert_h5ad_to_memmap(
    h5_path: str | Path,
    output_dir: str | Path | None = None,
    obsm_keys: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert a single h5ad file to a memmap sidecar directory.

    Args:
        h5_path: Path to the .h5ad file.
        output_dir: Output directory. Defaults to ``<h5_path>.mmap/``.
        obsm_keys: List of obsm keys to convert. ``None`` = auto-detect all.
        overwrite: If True, overwrite existing sidecar directory.

    Returns:
        Path to the created sidecar directory.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"h5ad file not found: {h5_path}")

    if output_dir is None:
        output_dir = h5_path.with_suffix(".mmap")
    output_dir = Path(output_dir)

    if output_dir.exists() and not overwrite:
        logger.info("Sidecar already exists: %s (use overwrite=True to recreate)", output_dir)
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": str(h5_path), "obsm_keys": [], "csr": False}

    with h5py.File(h5_path, "r") as f:
        n_cells = _get_n_cells(f)
        manifest["n_cells"] = n_cells

        # --- Convert CSR X matrix ---
        if "X" in f:
            x_obj = f["X"]
            if _is_csr(x_obj):
                _convert_csr(x_obj, output_dir, manifest)
            else:
                logger.info("X is dense — skipping CSR conversion for %s", h5_path.name)

        # --- Convert obsm dense matrices ---
        if "obsm" in f:
            available_keys = list(f["obsm"].keys())
            if obsm_keys is None:
                obsm_keys = available_keys
            for key in obsm_keys:
                if key not in available_keys:
                    logger.warning("obsm key '%s' not found in %s, skipping", key, h5_path.name)
                    continue
                _convert_obsm(f, key, n_cells, output_dir, manifest)

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    logger.info("Converted %s → %s", h5_path.name, output_dir)
    return output_dir


def convert_directory(
    dir_path: str | Path,
    obsm_keys: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Convert all h5ad files in a directory to memmap sidecars.

    Args:
        dir_path: Directory containing .h5ad files.
        obsm_keys: obsm keys to convert (None = auto-detect).
        overwrite: If True, overwrite existing sidecars.

    Returns:
        List of created sidecar directory paths.
    """
    dir_path = Path(dir_path)
    h5_files = sorted(dir_path.glob("*.h5ad")) + sorted(dir_path.glob("*.h5"))
    if not h5_files:
        logger.warning("No h5/h5ad files found in %s", dir_path)
        return []

    results = []
    for h5_file in h5_files:
        try:
            result = convert_h5ad_to_memmap(h5_file, obsm_keys=obsm_keys, overwrite=overwrite)
            results.append(result)
        except Exception:
            logger.exception("Failed to convert %s", h5_file)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_n_cells(f: h5py.File) -> int:
    """Infer number of cells from the file."""
    if "X" in f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            return int(x.shape[0])
        if "indptr" in x:
            return int(x["indptr"].shape[0]) - 1
    # Fallback: obs index length
    for key in ("obs/_index", "obs/index"):
        if key in f:
            return int(f[key].shape[0])
    raise ValueError("Cannot determine n_cells from file")


def _is_csr(obj) -> bool:
    return isinstance(obj, h5py.Group) and (
        obj.attrs.get("encoding-type") == "csr_matrix"
        or all(k in obj for k in ("data", "indices", "indptr"))
    )


def _convert_csr(x_group: h5py.Group, output_dir: Path, manifest: dict) -> None:
    """Stream CSR arrays to memmap files."""
    indptr_ds = x_group["indptr"]
    data_ds = x_group["data"]
    indices_ds = x_group["indices"]

    n_rows = int(indptr_ds.shape[0]) - 1
    nnz = int(data_ds.shape[0])

    # Get number of columns from shape attribute
    shape_attr = x_group.attrs.get("shape")
    if shape_attr is not None:
        n_cols = int(np.asarray(shape_attr).reshape(-1)[1])
    else:
        n_cols = int(np.max(indices_ds[:]) + 1) if nnz > 0 else 0

    logger.info("  CSR X: %d rows, %d nnz, %d cols", n_rows, nnz, n_cols)

    # indptr — small enough to read at once
    indptr_mmap = np.lib.format.open_memmap(
        str(output_dir / "X_indptr.npy"), mode="w+", dtype=np.int64, shape=(n_rows + 1,)
    )
    indptr_mmap[:] = np.asarray(indptr_ds[:], dtype=np.int64)
    indptr_mmap.flush()
    del indptr_mmap

    # data — stream in chunks
    data_mmap = np.lib.format.open_memmap(
        str(output_dir / "X_data.npy"), mode="w+", dtype=np.float32, shape=(nnz,)
    )
    _stream_1d(data_ds, data_mmap, nnz, np.float32)
    data_mmap.flush()
    del data_mmap

    # indices — stream in chunks
    indices_mmap = np.lib.format.open_memmap(
        str(output_dir / "X_indices.npy"), mode="w+", dtype=np.int32, shape=(nnz,)
    )
    _stream_1d(indices_ds, indices_mmap, nnz, np.int32)
    indices_mmap.flush()
    del indices_mmap

    manifest["csr"] = True
    manifest["csr_shape"] = [n_rows, n_cols]
    manifest["csr_nnz"] = nnz


def _convert_obsm(
    f: h5py.File, key: str, n_cells: int, output_dir: Path, manifest: dict
) -> None:
    """Convert a single obsm matrix to a dense memmap file."""
    obj = f[f"obsm/{key}"]

    if isinstance(obj, h5py.Dataset):
        shape = obj.shape
        if len(shape) == 1:
            shape = (shape[0], 1)
        n_cols = int(shape[1])
    elif _is_csr(obj):
        # CSR obsm — need to densify
        shape_attr = obj.attrs.get("shape")
        if shape_attr is not None:
            n_cols = int(np.asarray(shape_attr).reshape(-1)[1])
        else:
            n_cols = int(np.max(obj["indices"][:]) + 1) if obj["data"].shape[0] > 0 else 0
    else:
        logger.warning("  Unsupported obsm format for '%s', skipping", key)
        return

    logger.info("  obsm/%s: (%d, %d)", key, n_cells, n_cols)

    out_path = output_dir / f"obsm_{key}.npy"
    mmap = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float32, shape=(n_cells, n_cols)
    )

    if isinstance(obj, h5py.Dataset):
        # Dense — stream in row chunks
        for start in range(0, n_cells, _CHUNK_SIZE):
            end = min(start + _CHUNK_SIZE, n_cells)
            block = np.asarray(obj[start:end], dtype=np.float32)
            if block.ndim == 1:
                block = block[:, None]
            mmap[start:end] = block
    else:
        # CSR obsm — densify in chunks
        indptr = np.asarray(obj["indptr"][:], dtype=np.int64)
        data_ds = obj["data"]
        indices_ds = obj["indices"]
        for start in range(0, n_cells, _CHUNK_SIZE):
            end = min(start + _CHUNK_SIZE, n_cells)
            ptr_s = int(indptr[start])
            ptr_e = int(indptr[end])
            if ptr_e > ptr_s:
                chunk_data = np.asarray(data_ds[ptr_s:ptr_e], dtype=np.float32)
                chunk_idx = np.asarray(indices_ds[ptr_s:ptr_e], dtype=np.int32)
                local_indptr = indptr[start : end + 1] - ptr_s
                for i in range(end - start):
                    ps = int(local_indptr[i])
                    pe = int(local_indptr[i + 1])
                    if pe > ps:
                        mmap[start + i, chunk_idx[ps:pe]] = chunk_data[ps:pe]

    mmap.flush()
    del mmap

    manifest["obsm_keys"].append(key)
    manifest.setdefault("obsm_shapes", {})[key] = [n_cells, n_cols]


def _stream_1d(src_ds, dst_mmap, total: int, dtype) -> None:
    """Stream a 1D HDF5 dataset into a memmap in chunks."""
    chunk = 1_000_000
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        dst_mmap[start:end] = np.asarray(src_ds[start:end], dtype=dtype)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m cell_load.dataset._memmap_convert <path> [--overwrite]")
        sys.exit(1)

    target = Path(sys.argv[1])
    ow = "--overwrite" in sys.argv

    if target.is_dir():
        convert_directory(target, overwrite=ow)
    else:
        convert_h5ad_to_memmap(target, overwrite=ow)
