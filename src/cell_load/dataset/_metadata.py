from bisect import bisect_right

from torch.utils.data import ConcatDataset, Dataset, Subset


class MetadataConcatDataset(ConcatDataset):
    """
    ConcatDataset that enforces consistent metadata across all constituent datasets.
    """

    def __init__(self, datasets: list[Dataset]):
        super().__init__(datasets)
        self.base = datasets[0].dataset
        self.embed_key = self.base.embed_key
        self.control_pert = self.base.control_pert
        self.pert_col = self.base.pert_col
        self.batch_col = self.base.batch_col

        for ds in datasets:
            md = ds.dataset
            if (
                md.embed_key != self.embed_key
                or md.control_pert != self.control_pert
                or md.pert_col != self.pert_col
                or md.batch_col != self.batch_col
            ):
                raise ValueError(
                    "All datasets must share the same embed_key, control_pert, pert_col, and batch_col"
                )

    def __getitems__(self, indices):
        """
        Batch-aware fetch to enable fast-path loading when supported by datasets.
        Falls back to per-item access for non-consecutive loading.
        """
        if not getattr(self.base.mapping_strategy, "use_consecutive_loading", False):
            return [self[i] for i in indices]

        results = [None] * len(indices)
        grouped = {}

        for out_pos, idx in enumerate(indices):
            dataset_idx = bisect_right(self.cumulative_sizes, idx)
            sample_idx = (
                idx
                if dataset_idx == 0
                else idx - self.cumulative_sizes[dataset_idx - 1]
            )
            grouped.setdefault(dataset_idx, []).append((out_pos, sample_idx))

        for dataset_idx, pos_samples in grouped.items():
            ds = self.datasets[dataset_idx]
            positions, sample_indices = zip(*pos_samples)

            if isinstance(ds, Subset) and hasattr(ds.dataset, "__getitems__"):
                underlying_indices = [ds.indices[i] for i in sample_indices]
                samples = ds.dataset.__getitems__(underlying_indices)
            elif hasattr(ds, "__getitems__"):
                samples = ds.__getitems__(list(sample_indices))
            else:
                samples = [ds[i] for i in sample_indices]

            for pos, sample in zip(positions, samples):
                results[pos] = sample

        return results

    def ensure_h5_open(self) -> None:
        """
        Ensure all underlying H5 files are open in the current process.
        """
        for ds in self.datasets:
            base = ds.dataset if isinstance(ds, Subset) else ds
            if hasattr(base, "ensure_h5_open"):
                base.ensure_h5_open()
