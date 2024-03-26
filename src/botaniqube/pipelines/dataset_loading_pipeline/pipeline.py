from kedro.pipeline import Pipeline, node
from .nodes import get_project_path, get_sizes, get_loaders, get_images

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_project_path,
                inputs=None,
                outputs="path",
                name="get_project_path_node",
            ),
            node(
                func=get_images,
                inputs="params",
                outputs="image_datasets",
                name="get_images_node",
            ),
            node(
                func=get_loaders,
                inputs=["image_datasets", "params"],
                outputs="dataloaders",
                name="get_loaders_node",
            ),
            node(
                func=get_sizes,
                inputs="image_datasets",
                outputs="dataset_sizes",
                name="get_sizes_node",
            ),
        ]
    )
