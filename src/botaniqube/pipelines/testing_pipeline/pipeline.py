from kedro.pipeline import Pipeline, node
from .nodes import prepare_test_data, evaluate_model

def create_pipeline(**kwargs):
    testing_nodes = Pipeline(
       [
            node(
                func=prepare_test_data,
                inputs= {
                    "img_size": "params:image_size",
                    "batch_size": "params:training.batch_size",
                },
                outputs="test_loader",
                name="prepare_test_data_node",
            ),
            node(
                func=evaluate_model,
                inputs={
                    "PATH": "PATH",
                    "params" : "params:model",
                    "img_size": "params:image_size",
                    "batch_size": "params:training.batch_size",
                },
                outputs="evaluation_result",
                name="evaluate_model_node",
            ),
        ],
    )

    return testing_nodes
