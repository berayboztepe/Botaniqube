from kedro.pipeline import Pipeline, node
from .nodes import prepare_test_data, evaluate_model, fetch_model

def create_pipeline(**kwargs):
    testing_nodes = Pipeline(
       [
            node(
                func=prepare_test_data,
                inputs= {
                    "params": "params:testing",
                },
                outputs="test_loader",
                name="prepare_test_data_node",
            ),
            node(
                func=fetch_model,
                inputs={
                    "params" : "params:model",
                },
                outputs="trained_model",
                name="fetch_model",
            ),
            node(
                func=evaluate_model,
                inputs={
                    "test_loader": "test_loader",
                    "trained_model" : "trained_model",
                },
                outputs="evaluation_result",
                name="evaluate_model_node",
            ),
        ],
    )

    return testing_nodes
