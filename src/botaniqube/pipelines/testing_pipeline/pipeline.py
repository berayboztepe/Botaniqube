from kedro.pipeline import Pipeline, node
from pipelines.dataset_loading_pipeline.pipeline import create_pipeline as create_dataset_loading_pipeline
from training_pipeline.pipeline import create_pipeline as create_training_pipeline
from .nodes import get_model, prepare_test_data, evaluate_model

def create_pipeline(**kwargs):
    dataset_loading_pipeline = create_dataset_loading_pipeline()
    dataset_loading_output = dataset_loading_pipeline.run()

    params = dataset_loading_output["params"]

    training_pipeline = create_training_pipeline()
    CNN = training_pipeline.nodes.get("create_cnn_model")

    testing_nodes = Pipeline(
        [
            node(
                func=get_model,
                inputs=dict(params=params, CNN=CNN),
                outputs="model",
                name="get_model_node",
            ),
            node(
                func=prepare_test_data,
                inputs=params,
                outputs="test_loader",
                name="prepare_test_data_node",
            ),
            node(
                func=evaluate_model,
                inputs=dict(params=params, CNN=CNN),
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )

    return testing_nodes
