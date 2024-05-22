"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline
from botaniqube.pipelines import dataset_loading_pipeline, training_pipeline, testing_pipeline, dataset_downloading_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ddp = dataset_downloading_pipeline.create_pipeline()
    dlp = dataset_loading_pipeline.create_pipeline()
    trp = training_pipeline.create_pipeline()
    tsp = testing_pipeline.create_pipeline()
    
    return {
        "__default__": ddp + dlp + trp + tsp,
        "dataset_downloading": ddp,
        "dataset_loading": dlp,
        "training": trp,
        "testing": tsp,
        "dataset_loading_training": ddp + dlp + trp,
    }