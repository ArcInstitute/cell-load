from torch.utils.data import DataLoader
from cell_load.dataset.cell_sentence_dataset import FilteredGenesCounts
from cell_load.dataset.cell_sentence_dataset import CellSentenceCollator


def create_dataloader(
    cfg,
    workers=1,
    data_dir=None,
    datasets=None,
    shape_dict=None,
    adata=None,
    adata_name=None,
    shuffle=False,
    sentence_collator=None,
):
    """
    Expected to be used for inference
    Either datasets and shape_dict or adata and adata_name should be provided
    """
    if datasets is None and adata is None:
        raise ValueError(
            "Either datasets and shape_dict or adata and adata_name should be provided"
        )

    if adata is not None:
        shuffle = False

    if data_dir:
        cfg.model.data_dir = data_dir
        # ? utils.get_dataset_cfg(cfg).data_dir = data_dir

    dataset = FilteredGenesCounts(
        cfg,
        datasets=datasets,
        shape_dict=shape_dict,
        adata=adata,
        adata_name=adata_name,
    )
    if sentence_collator is None:
        sentence_collator = CellSentenceCollator(
            cfg,
            valid_gene_mask=dataset.valid_gene_index,
            ds_emb_mapping_inference=dataset.ds_emb_map,
            is_train=False,
        )

    # validation should not use cell augmentations
    sentence_collator.training = False

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        shuffle=shuffle,
        collate_fn=sentence_collator,
        num_workers=workers,
        persistent_workers=True,
    )
    return dataloader
