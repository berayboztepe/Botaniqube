from kedro.pipeline import Pipeline, node
from .nodes import create_cnn_model, train_model, save_model

def create_pipeline(**kwargs):
    training_nodes = Pipeline(
        [
            node(
                func=create_cnn_model,
                inputs="params",
                outputs="model",
                name="create_cnn_model_node",
            ),
            node(
                func=train_model,
                inputs=["model", "dataloaders", "dataset_sizes", "params"],
                outputs="model_trained",
                name="train_model_node",
            ),
            node(
                func=save_model,
                inputs=["model_trained"],
                outputs=None,
                name="save_model_node",
            )
        ]
    )
    
    return training_nodes
