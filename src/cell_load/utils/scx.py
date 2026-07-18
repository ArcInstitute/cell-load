from pathlib import Path


def is_scx_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".scx"


def open_scx_anndata(path: str | Path):
    try:
        import pyscx
    except ImportError as exc:
        raise ImportError(
            "Reading .scx files requires pyscx. Install a Cell Load build with "
            "SCX support."
        ) from exc

    return pyscx.open(str(path)).to_anndata(backed=True)
