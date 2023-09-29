from openood.utils import config
from openood.experiments import finetune
config_files = [
    './configs/finetuning/full-ft.yml',
]
config = config.Config(*config_files)

experiment = finetune.FullFTExperiment(config)
experiment.run()

