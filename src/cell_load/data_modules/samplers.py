import logging
import time
from typing import Iterator
import copy

import numpy as np
from torch.utils.data import Sampler, Subset
import torch.distributed as dist

from ..dataset import MetadataConcatDataset, PerturbationDataset
from ..utils.data_utils import H5MetadataCache

logger = logging.getLogger(__name__)


class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that cells in each batch share the same
    (cell_type, perturbation) combination, using only H5 codes.

    Instead of grouping by cell type and perturbation names, this sampler
    groups based on integer codes stored in the H5 file (e.g. `cell_type_codes`
    and `pert_codes` in the H5MetadataCache). This avoids repeated string operations.

    Supports distributed training.
    """

    def __init__(
        self,
        dataset: "MetadataConcatDataset",
        batch_size: int,
        drop_last: bool = False,
        cell_sentence_len: int = 512,
        test: bool = False,
        use_batch: bool = False,
        use_consecutive_loading: bool = False,
        downsample_cells: int | None = None,
        seed: int = 0,
        epoch: int = 0,
    ):
        logger.info(
            "Creating perturbation batch sampler with metadata caching (using codes)..."
        )
        start_time = time.time()

        # If the provided dataset has a `.data_source` attribute, use that.
        self.dataset = (
            dataset.data_source if hasattr(dataset, "data_source") else dataset
        )
        self.batch_size = batch_size
        self.test = test
        self.use_batch = use_batch
        self.use_consecutive_loading = use_consecutive_loading
        self.seed = seed
        self.epoch = epoch

        if self.test and self.batch_size != 1:
            logger.warning(
                "Batch size should be 1 for test mode. Setting batch size to 1."
            )
            self.batch_size = 1

        self.cell_sentence_len = cell_sentence_len
        self.drop_last = drop_last
        self.downsample_cells = self._validate_downsample_cells(downsample_cells)

        # Setup distributed settings if distributed mode is enabled.
        self.distributed = False
        self.num_replicas = 1
        self.rank = 0

        if dist.is_available() and dist.is_initialized():
            self.distributed = True
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(
                f"Distributed mode enabled. World size: {self.num_replicas}, rank: {self.rank}."
            )

        # Create caches for all unique H5 files.
        self.metadata_caches = {}
        for subset in self.dataset.datasets:
            base_dataset: PerturbationDataset = subset.dataset
            self.metadata_caches[base_dataset.h5_path] = base_dataset.metadata_cache
        if self.use_consecutive_loading:
            self._validate_file_level_consecutive_groups()

        # Create batches using the code-based grouping.
        self.sentences = self._create_sentences()
        sentence_lens = [len(sentence) for sentence in self.sentences]
        avg_num = np.mean(sentence_lens)
        std_num = np.std(sentence_lens)
        tot_num = np.sum(sentence_lens)
        logger.info(
            f"Total # cells {tot_num}. Cell set size mean / std before resampling: {avg_num:.2f} / {std_num:.2f}."
        )

        # combine sentences into batches that are flattened
        logger.info(
            f"Creating meta-batches with cell_sentence_len={cell_sentence_len}..."
        )
        start_time = time.time()
        self.batches = self._create_batches()
        self.tot_num = tot_num
        end_time = time.time()
        logger.info(
            f"Sampler created with {len(self.batches)} batches in {end_time - start_time:.2f} seconds."
        )

    def _create_batches(self) -> list[list[int]]:
        """
        Combines existing batches into meta-batches of size batch_size * cell_sentence_len,
        sampling with replacement if needed to reach cell_sentence_len.

        IF distributed, each rank will process a subset of the sentences.
        """

        if self.distributed:
            rank_sentences = self._get_rank_sentences()

        else:
            rank_sentences = self.sentences

        all_batches = []
        current_batch = []

        num_full = 0
        num_partial = 0
        for sentence in rank_sentences:
            # If batch is smaller than cell_sentence_len, sample with replacement
            if len(sentence) < self.cell_sentence_len and not self.test:
                # during inference, don't sample by replacement
                if self.use_consecutive_loading:
                    repeats = int(
                        np.ceil(self.cell_sentence_len / max(len(sentence), 1))
                    )
                    new_sentence = (sentence * repeats)[: self.cell_sentence_len]
                else:
                    new_sentence = np.random.choice(
                        sentence, size=self.cell_sentence_len, replace=True
                    ).tolist()
                num_partial += 1
            else:
                new_sentence = copy.deepcopy(sentence)
                assert len(new_sentence) == self.cell_sentence_len or self.test
                num_full += 1

            sentence_len = len(new_sentence) if self.test else self.cell_sentence_len

            if len(current_batch) + len(new_sentence) <= self.batch_size * sentence_len:
                current_batch.extend(new_sentence)
            else:
                if current_batch:  # Add the completed meta-batch
                    all_batches.append(current_batch)
                current_batch = new_sentence

        if self.distributed:
            logger.info(
                f"Rank {self.rank}: Of {len(rank_sentences)} sentences, "
                f"{num_full} were full and {num_partial} were partial."
            )
        else:
            logger.info(
                f"Of all batches, {num_full} were full and {num_partial} were partial."
            )

        # Add the last meta-batch if it exists
        if current_batch and not self.drop_last:
            all_batches.append(current_batch)

        return all_batches

    def _validate_downsample_cells(self, downsample_cells: int | None) -> int | None:
        if downsample_cells is None:
            return None
        if isinstance(downsample_cells, bool):
            raise ValueError("downsample_cells must be a positive integer or None.")
        if isinstance(downsample_cells, float):
            if not downsample_cells.is_integer():
                raise ValueError("downsample_cells must be a positive integer or None.")
            downsample_cells = int(downsample_cells)
        elif not isinstance(downsample_cells, (int, np.integer)):
            raise ValueError("downsample_cells must be a positive integer or None.")
        downsample_cells = int(downsample_cells)
        if downsample_cells <= 0:
            raise ValueError("downsample_cells must be a positive integer or None.")
        return downsample_cells

    def _apply_downsample_cells(self, sentences: list[list[int]]) -> list[list[int]]:
        if self.downsample_cells is None or not sentences:
            return sentences

        total = sum(len(sentence) for sentence in sentences)
        if total <= self.downsample_cells:
            return sentences

        order = np.random.permutation(len(sentences))
        selected: list[list[int]] = []
        remaining = self.downsample_cells
        for idx in order:
            if remaining <= 0:
                break
            sentence = sentences[idx]
            if len(sentence) <= remaining:
                selected.append(sentence)
                remaining -= len(sentence)
            else:
                if not self.drop_last and remaining > 0:
                    selected.append(sentence[:remaining])
                remaining = 0
                break

        return selected

    def _get_rank_sentences(self) -> list[list[int]]:
        """
        Get the subset of sentences that this rank should process.
        Sentences are shuffled using epoch-based seed, then distributed across ranks.
        """
        # Shuffle sentences using epoch-based seed for consistent ordering across ranks
        shuffled_sentences = self.sentences.copy()
        np.random.RandomState(self.seed + self.epoch).shuffle(shuffled_sentences)

        # Calculate sentence distribution across processes
        total_sentences = len(shuffled_sentences)
        base_sentences = total_sentences // self.num_replicas
        remainder = total_sentences % self.num_replicas

        # Calculate number of sentences for this specific rank
        if self.rank < remainder:
            num_sentences_for_rank = base_sentences + 1
        else:
            num_sentences_for_rank = base_sentences

        # Calculate starting sentence index for this rank
        start_sentence_idx = self.rank * base_sentences + min(self.rank, remainder)
        end_sentence_idx = start_sentence_idx + num_sentences_for_rank

        rank_sentences = shuffled_sentences[start_sentence_idx:end_sentence_idx]

        logger.info(
            f"Rank {self.rank}: Processing {len(rank_sentences)} sentences "
            f"(indices {start_sentence_idx} to {end_sentence_idx - 1} of {total_sentences})"
        )

        return rank_sentences

    def _format_group_key(self, cache: H5MetadataCache, key: tuple[int, ...]) -> str:
        if self.use_batch:
            batch_code, cell_code, pert_code = key
            batch_name = cache.batch_categories[batch_code]
            cell_name = cache.cell_type_categories[cell_code]
            pert_name = cache.pert_categories[pert_code]
            return (
                f"(batch='{batch_name}', cell_type='{cell_name}', "
                f"perturbation/condition='{pert_name}')"
            )

        cell_code, pert_code = key
        cell_name = cache.cell_type_categories[cell_code]
        pert_name = cache.pert_categories[pert_code]
        return f"(cell_type='{cell_name}', perturbation/condition='{pert_name}')"

    def _validate_consecutive_groups(
        self,
        cache: H5MetadataCache,
        indices: np.ndarray,
        code_change: np.ndarray,
        cell_codes: np.ndarray,
        pert_codes: np.ndarray,
        batch_codes: np.ndarray | None = None,
        h5_path: str | None = None,
    ) -> None:
        """
        Ensure each grouping key appears in exactly one contiguous run.

        With use_batch=False, key is (cell_type, perturbation).
        With use_batch=True, key is (batch, cell_type, perturbation).
        """
        run_starts = np.r_[0, np.where(code_change)[0] + 1]
        seen: dict[tuple[int, ...], int] = {}

        for run_idx, start in enumerate(run_starts):
            if batch_codes is None:
                key = (int(cell_codes[start]), int(pert_codes[start]))
            else:
                key = (
                    int(batch_codes[start]),
                    int(cell_codes[start]),
                    int(pert_codes[start]),
                )

            prev_run_idx = seen.get(key)
            if prev_run_idx is None:
                seen[key] = run_idx
                continue

            curr_pos = int(start)
            prev_pos = int(run_starts[prev_run_idx])
            group_str = self._format_group_key(cache, key)

            between_key = None
            if run_idx - prev_run_idx > 1:
                between_start = int(run_starts[prev_run_idx + 1])
                if batch_codes is None:
                    between_key = (
                        int(cell_codes[between_start]),
                        int(pert_codes[between_start]),
                    )
                else:
                    between_key = (
                        int(batch_codes[between_start]),
                        int(cell_codes[between_start]),
                        int(pert_codes[between_start]),
                    )

            grouping = (
                "(batch, cell_type, perturbation/condition)"
                if self.use_batch
                else "(cell_type, perturbation/condition)"
            )
            detail = (
                f"Observed pattern: {group_str} -> {self._format_group_key(cache, between_key)} -> {group_str}. "
                if between_key is not None
                else ""
            )
            raise ValueError(
                "use_consecutive_loading=True requires each "
                f"{grouping} group to appear in one contiguous run in file order. "
                f"Found a non-consecutive group in '{h5_path or cache.h5_path}': {group_str}. "
                f"It appears at positions {prev_pos} and {curr_pos} "
                f"(file indices {int(indices[prev_pos])} and {int(indices[curr_pos])}). "
                f"{detail}"
                f"Please sort data so identical {grouping} groups are contiguous."
            )

    def _validate_file_level_consecutive_groups(self) -> None:
        """
        Validate consecutive-loading contiguity against full file order.

        We validate the raw file-level code streams (not split subsets) so that
        interleaving like (ct1, pert1), (ct2, pertX), (ct1, pert1) is rejected.
        """
        for h5_path, cache in self.metadata_caches.items():
            indices = np.arange(cache.n_cells, dtype=np.int64)
            cell_codes = cache.cell_type_codes
            pert_codes = cache.pert_codes

            if getattr(self, "use_batch", False):
                batch_codes = cache.batch_codes
                code_change = (
                    (batch_codes[1:] != batch_codes[:-1])
                    | (cell_codes[1:] != cell_codes[:-1])
                    | (pert_codes[1:] != pert_codes[:-1])
                )
                self._validate_consecutive_groups(
                    cache=cache,
                    indices=indices,
                    code_change=code_change,
                    cell_codes=cell_codes,
                    pert_codes=pert_codes,
                    batch_codes=batch_codes,
                    h5_path=str(h5_path),
                )
            else:
                code_change = (cell_codes[1:] != cell_codes[:-1]) | (
                    pert_codes[1:] != pert_codes[:-1]
                )
                self._validate_consecutive_groups(
                    cache=cache,
                    indices=indices,
                    code_change=code_change,
                    cell_codes=cell_codes,
                    pert_codes=pert_codes,
                    batch_codes=None,
                    h5_path=str(h5_path),
                )

    def _process_subset(self, global_offset: int, subset: Subset) -> list[list[int]]:
        """
        Process a single subset to create batches based on H5 codes.

        Optimized version with integer group encoding:
        - Groups are encoded into a single integer via np.ravel_multi_index.
        - Sorting/grouping is done on simple integers instead of structured dtypes.
        - Much faster for large numbers of groups.
        """
        base_dataset = subset.dataset
        indices = np.array(subset.indices)
        cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]

        # Codes
        cell_codes = cache.cell_type_codes[indices]
        pert_codes = cache.pert_codes[indices]

        if getattr(self, "use_batch", False):
            batch_codes = cache.batch_codes[indices]
            # Encode (batch, cell, pert) into one integer
            group_keys = np.ravel_multi_index(
                (batch_codes, cell_codes, pert_codes),
                (batch_codes.max() + 1, cell_codes.max() + 1, pert_codes.max() + 1),
            )
        else:
            # Encode (cell, pert) into one integer
            group_keys = np.ravel_multi_index(
                (cell_codes, pert_codes), (cell_codes.max() + 1, pert_codes.max() + 1)
            )

        # Global indices
        global_indices = np.arange(global_offset, global_offset + len(indices))

        # Sort once by group key
        order = np.argsort(group_keys)
        sorted_keys = group_keys[order]
        sorted_indices = global_indices[order]

        # Find group boundaries
        unique_keys, group_starts = np.unique(sorted_keys, return_index=True)
        group_starts = np.r_[group_starts, len(sorted_keys)]

        subset_batches = []

        # Iterate groups
        for start, end in zip(group_starts[:-1], group_starts[1:]):
            group_indices = sorted_indices[start:end]
            np.random.shuffle(group_indices)

            group_sentences = []
            for i in range(0, len(group_indices), self.cell_sentence_len):
                sentence = group_indices[i : i + self.cell_sentence_len]
                if len(sentence) < self.cell_sentence_len and self.drop_last:
                    continue
                group_sentences.append(sentence.tolist())

            group_sentences = self._apply_downsample_cells(group_sentences)
            subset_batches.extend(group_sentences)

        return subset_batches

    def _process_subset_consecutive(
        self, global_offset: int, subset: Subset
    ) -> list[list[int]]:
        """
        Process a single subset to create consecutive sentences based on H5 codes.

        This assumes the input indices are already in file order and splits
        sentences at code-change boundaries without shuffling.
        """
        base_dataset = subset.dataset
        indices = np.array(subset.indices)
        if indices.size == 0:
            return []

        cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]

        # Codes in file order
        cell_codes = cache.cell_type_codes[indices]
        pert_codes = cache.pert_codes[indices]
        if getattr(self, "use_batch", False):
            batch_codes = cache.batch_codes[indices]
            code_change = (
                (batch_codes[1:] != batch_codes[:-1])
                | (cell_codes[1:] != cell_codes[:-1])
                | (pert_codes[1:] != pert_codes[:-1])
            )
        else:
            code_change = (cell_codes[1:] != cell_codes[:-1]) | (
                pert_codes[1:] != pert_codes[:-1]
            )

        # Global indices in dataset order
        global_indices = np.arange(global_offset, global_offset + len(indices))

        # Split into contiguous segments when codes change
        boundaries = np.where(code_change)[0] + 1
        segments = np.split(global_indices, boundaries)

        subset_batches = []
        for segment in segments:
            group_sentences = []
            for i in range(0, len(segment), self.cell_sentence_len):
                sentence = segment[i : i + self.cell_sentence_len]
                if len(sentence) < self.cell_sentence_len and self.drop_last:
                    continue
                group_sentences.append(sentence.tolist())

            group_sentences = self._apply_downsample_cells(group_sentences)
            subset_batches.extend(group_sentences)

        return subset_batches

    def _create_sentences(self) -> list[list[int]]:
        """
        Process each subset sequentially (across all datasets) and combine the batches.
        """
        global_offset = 0
        all_batches = []
        for subset in self.dataset.datasets:
            if self.use_consecutive_loading:
                subset_batches = self._process_subset_consecutive(global_offset, subset)
            else:
                subset_batches = self._process_subset(global_offset, subset)
            all_batches.extend(subset_batches)
            global_offset += len(subset)
        np.random.shuffle(all_batches)

        return all_batches

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle the order of batches each time we iterate in non-distributed mode.
        if not self.distributed:
            self.batches = self._create_batches()
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        This ensures all replicas use a different random ordering for each epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        # Recreate batches for new epoch (sentences remain the same)
        self.batches = self._create_batches()
