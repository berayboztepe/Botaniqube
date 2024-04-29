import wandb

wandb.init(project='botaniqube', entity='antoni-krzysztof-czapski')

my_accuracy = 0.95
wandb.log({'accuracy': my_accuracy})

wandb.finish()

