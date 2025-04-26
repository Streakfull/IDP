import pdb
import torch
import numpy as np
from training.ModelTrainer import ModelTrainer
from datasets.SimMatches import SimMatches
from datasets.VisualSimMatches import VisualSimMatches
from lutils.general import seed_all
seed_all(111)
x = torch.cuda.mem_get_info()
print(x)

trainer = ModelTrainer(dataset_type=VisualSimMatches,
                       options={"tdm_notebook": True})
dataset = trainer.data_loader_handler.dataset
print("Dataset length: ", len(dataset))
torch.cuda.empty_cache()
print(torch.cuda.mem_get_info())
model = trainer.model
str(trainer.logger.experiment_dir)
exp = f"{str(trainer.logger.experiment_dir)}/tb"
exp
#!tensorboard --logdir exp --|bind_all
print(f"tensorboard --logdir {exp} --bind_all")

trainer.train()
