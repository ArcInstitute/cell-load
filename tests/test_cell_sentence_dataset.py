import pytest
import numpy as np
import torch
from cell_load.dataset.cell_sentence_dataset import CellSentenceDataset


class DummyAdata:
    def __init__(self, shape=(3, 5)):
        self.X = np.arange(np.prod(shape)).reshape(shape)
        self.var_names = [f"gene{i}" for i in range(shape[1])]
        self.shape = shape
        self.var = {"gene_name": np.array(self.var_names)}


@pytest.fixture
def dummy_adata():
    return DummyAdata()


def test_cell_sentence_dataset_basic(dummy_adata):
    cfg = type("cfg", (), {})()
    cfg.model = type("model", (), {})()
    cfg.model.batch_size = 2
    cfg.dataset = type("dataset", (), {})()
    cfg.dataset.pad_length = 5
    cfg.dataset.P = 1
    cfg.dataset.N = 1
    cfg.dataset.S = 1
    dataset = CellSentenceDataset(cfg, adata=dummy_adata, adata_name="dummy")
    assert len(dataset) == dummy_adata.shape[0]
    counts, idx, dataset_name, dataset_num = dataset[0]
    assert isinstance(counts, torch.Tensor)
    assert counts.shape[1] == dummy_adata.shape[1]
    assert dataset_name == "dummy"
    assert dataset_num == 0
