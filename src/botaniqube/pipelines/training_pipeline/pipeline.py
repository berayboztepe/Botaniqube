from kedro.pipeline import Pipeline
from kedro.pipeline import node
from pipelines.dataset_loading_pipeline.pipeline import create_pipeline as create_dataset_loading_pipeline
from pipelines.training_pipeline.nodes import create_nodes as create_training_nodes

def create_pipeline(**kwargs):
    dataset_loading_pipeline = create_dataset_loading_pipeline()
    training_nodes = create_training_nodes()

    training_nodes[0].inputs = dataset_loading_pipeline.outputs

    return Pipeline(dataset_loading_pipeline + training_nodes)
