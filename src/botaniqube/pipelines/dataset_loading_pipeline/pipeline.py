from kedro.pipeline import Pipeline
from .nodes import get_param, get_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_param,
                inputs=None,
                outputs="params",
                name="get_params_node",
            ),
            node(
                func=get_data,
                inputs=None,
                outputs=["dataloaders", "dataset_sizes"],
                name="get_data_node",
            ),
        ]
    )
