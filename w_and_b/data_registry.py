import wandb

# Assuming you've already logged in and initialized W&B
wandb.init(project='botaniqube', entity='antoni-krzysztof-czapski')

# Create a new artifact (dataset version)
# Replace 'dataset_name' with your dataset's name and 'type' with "dataset"
# The 'description' and 'metadata' are optional but helpful for documentation
artifact = wandb.Artifact('sample_data', type='dataset', description='Plant disease images', metadata={'version': '2.0', 'partition': 'test'})

# Add files to your artifact
# Use artifact.add_file() for individual files or artifact.add_dir() for directories
# Replace 'path/to/your/dataset' with the path to your dataset files or directory
artifact.add_dir('data/01_raw/sample')

# Log the artifact to W&B
wandb.log_artifact(artifact)

# Finish the run
wandb.finish()
