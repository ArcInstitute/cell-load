#!/usr/bin/env python3
"""CLI for filter_on_target_knockdown function."""

import argparse
import logging
import sys
from pathlib import Path

import anndata
import scanpy as sc

logger = logging.getLogger(__name__)


def preprocess_state_paper(adata_pp: anndata.AnnData) -> anndata.AnnData:
    """
    Apply preprocessing as described in the state paper:
    1. Normalize to 10k read depth
    2. Log transform

    Args:
        adata: Input AnnData object

    Returns:
        Preprocessed AnnData object
    """
    logger.info("Applying state paper preprocessing.")

    # 1. Normalize to 10k read depth
    logger.info("Normalizing to 10k read depth.")
    sc.pp.normalize_total(adata_pp, target_sum=1e4)

    # 2. Log transform
    logger.info("Applying log1p transform.")
    sc.pp.log1p(adata_pp)

    return adata_pp


def main():
    """Main CLI entry point for filter_on_target_knockdown."""
    parser = argparse.ArgumentParser(
        description="Filter cells based on on-target knockdown criteria"
    )

    parser.add_argument("input", type=str, help="Path to input AnnData file (.h5ad)")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output AnnData file (.h5ad)",
    )

    parser.add_argument(
        "--perturbation-column",
        type=str,
        default="gene",
        help="Column in adata.obs holding perturbation identities (default: gene)",
    )

    parser.add_argument(
        "--control-label",
        type=str,
        default="non-targeting",
        help="Category in perturbation_column marking control cells (default: non-targeting)",
    )

    parser.add_argument(
        "--residual-expression",
        type=float,
        default=0.30,
        help="Residual fraction for perturbation-level threshold (default: 0.30)",
    )

    parser.add_argument(
        "--cell-residual-expression",
        type=float,
        default=0.50,
        help="Residual fraction for cell-level threshold (default: 0.50)",
    )

    parser.add_argument(
        "--min-cells",
        type=int,
        default=30,
        help="Minimum cells per perturbation after filtering (default: 30)",
    )

    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Use this matrix in adata.layers instead of adata.X",
    )

    parser.add_argument(
        "--var-gene-name",
        type=str,
        default="gene_name",
        help="Column in adata.var containing gene names (default: gene_name)",
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply preprocessing as in state paper: normalize to 10k read depth "
        "and log transform before writing output",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Python logging level (default: WARNING)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file %s does not exist.", args.input)
        sys.exit(1)

    # Load data
    try:
        logger.info("Loading data from %s.", args.input)
        adata = anndata.read_h5ad(args.input)
        logger.info(
            "Loaded AnnData with %d cells and %d genes.",
            adata.n_obs,
            adata.n_vars,
        )
    except Exception as e:
        logger.exception("Error loading data: %s", e)
        sys.exit(1)

    # Import and apply filter
    try:
        from ..utils.data_utils import filter_on_target_knockdown

        logger.info("Applying on-target knockdown filter.")
        filtered_adata = filter_on_target_knockdown(
            adata,
            perturbation_column=args.perturbation_column,
            control_label=args.control_label,
            residual_expression=args.residual_expression,
            cell_residual_expression=args.cell_residual_expression,
            min_cells=args.min_cells,
            layer=args.layer,
            var_gene_name=args.var_gene_name,
        )

        logger.info(
            "Filtered to %d cells and %d genes.",
            filtered_adata.n_obs,
            filtered_adata.n_vars,
        )

    except Exception as e:
        logger.exception("Error applying filter: %s", e)
        sys.exit(1)

    # Apply preprocessing if requested
    if args.preprocess:
        try:
            filtered_adata = preprocess_state_paper(filtered_adata)
        except Exception as e:
            logger.exception("Error during preprocessing: %s", e)
            sys.exit(1)

    # Save output
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Fix potential DataFrame index conflicts before saving
        if filtered_adata.var.index.name in filtered_adata.var.columns:
            # Check if the values are actually different
            index_values = filtered_adata.var.index.values
            column_values = filtered_adata.var[filtered_adata.var.index.name].values
            if not all(index_values == column_values):
                # Rename the index to avoid conflict
                logger.info(
                    "Fixing var index name conflict: %s -> %s",
                    filtered_adata.var.index.name,
                    f"{filtered_adata.var.index.name}_index",
                )
                filtered_adata.var.index.name = f"{filtered_adata.var.index.name}_index"

        logger.info("Saving filtered data to %s.", args.output)
        filtered_adata.write_h5ad(args.output)
        logger.info("Done.")

    except Exception as e:
        logger.exception("Error saving output: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
