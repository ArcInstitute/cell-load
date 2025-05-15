from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class TaskType(Enum):
    """
    Different tasks at the dataset / cell type level.

    ZEROSHOT: if specified for a given dataset/cell_type, that cell type is only used in test
    FEWSHOT: if specified for a given dataset/cell_type, that cell type is partially used in train / val and mostly used in test
    TRAINING: if specified for a given dataset/cell_type, that cell type is used in train / val and not in test
    """

    ZEROSHOT = "zeroshot"
    FEWSHOT = "fewshot"
    TRAINING = "training"


@dataclass
class TaskSpec:
    """Specification for a training or testing task"""

    dataset: str  # e.g. "replogle"
    cell_type: Optional[str] = None  # e.g. "jurkat"
    task_type: TaskType = TaskType.ZEROSHOT


def parse_dataset_specs(specs: List[str]) -> List[TaskSpec]:
    """Parse dataset specifications into TaskSpec objects

    Format: dataset[_celltype][,tasktype]
    Examples:
    - replogle
    - replogle_jurkat:zeroshot
    - sciplex_k562:fewshot
    """
    parsed_specs = []

    for spec in specs:
        parts = spec.split(":")
        dataset_part = parts[0]
        cell_type = parts[1] if len(parts) > 2 else None
        task_type = TaskType.TRAINING  # Default

        if len(parts) > 2:
            task_type = TaskType[parts[2].upper()]

        # Parse dataset and optional cell type
        dataset_parts = dataset_part.split(":")
        dataset = dataset_parts[0]

        parsed_specs.append(
            TaskSpec(dataset=dataset, cell_type=cell_type, task_type=task_type)
        )

    return parsed_specs
