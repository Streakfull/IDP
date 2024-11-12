import json
import time
from datetime import datetime
from pathlib import Path
from torchinfo import summary
from typing import Any

from cprint import *
from torch.utils.tensorboard import SummaryWriter

from lutils.general import mkdir
from lutils.model_utils import summarize_model


class Logger:
    def __init__(self, global_config):
        training_config = global_config['training']
        cprint.ok(training_config)

        self.append_loss_every = training_config["append_loss_every"]
        self.experiment_dir = self.get_experiment_dir(training_config)
        mkdir(self.experiment_dir)
        self.loss_log_file_name = f"{self.experiment_dir}/loss_log.txt"
        self.directories = ["checkpoints", 'tb', 'visuals', 'modelsummary']
        self.make_log_files(global_config)
        self.make_dirs()
        self._writer = SummaryWriter(f"{self.experiment_dir}/tb")

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int = None, walltime: float = None,
                   new_style: bool = False, double_precision: bool = False):
        self._writer.add_scalar(
            tag, scalar_value, global_step, walltime, new_style, double_precision)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int = None, walltime: float = None):
        self._writer.add_scalars(
            main_tag, tag_scalar_dict, global_step, walltime)

    def flush_writer(self):
        self._writer.flush()

    def close_writer(self):
        self._writer.close()

    def make_log_files(self, global_config):
        training_config = global_config['training']
        with open(f"{self.experiment_dir}/description.txt", "w") as file1:
            description = training_config["description"]
            file1.write(description)

        with open(f"{self.experiment_dir}/global_configs.json", "w") as file1:
            json_object = json.dumps(global_config, indent=4)
            file1.write(str(json_object))

        with open(self.loss_log_file_name, "w") as file1:
            loss_log_title = "Loss Log " + time.strftime("%Y-%m-%d")
            file1.write(loss_log_title)
            file1.write("\n")

    def make_dirs(self):
        for directory in self.directories:
            mkdir(f"{self.experiment_dir}/{directory}")

    def start_new_epoch(self, epoch):
        with open(self.loss_log_file_name, "a") as log_file:
            log_file.write(f'** Epoch: {epoch} **\n')

    def log_loss(self, epoch, iteration, loss):
        if iteration % self.append_loss_every == (self.append_loss_every - 1) or (epoch == 0 and iteration == 0):
            message = '(epoch: %d, iters: %d, loss: %.6f)' % (
                epoch, iteration, loss.item())
            with open(self.loss_log_file_name, "a") as log_file:
                log_file.write('%s\n' % message)

    def log_model_summary(self, model, input_shape=(64,), batch_size=1):
        return
        input_shape = (batch_size,) + input_shape
        params = summarize_model(model)
        shapes_archi = str(summary(model, input_shape, verbose=0))
        pytorch_model = str(model)
        with open(f"{self.experiment_dir}/modelsummary/paramLayers.txt", "w") as params_writer:
            params_writer.write(params)
        with open(f"{self.experiment_dir}/modelsummary/output_shapes.txt", "w") as shapes_writer:
            shapes_writer.write(shapes_archi)
        with open(f"{self.experiment_dir}/modelsummary/pytorch_model.txt", "w") as model_writer:
            model_writer.write(pytorch_model)

    def get_experiment_dir(self, config):
        experiment_id = config["experiment_id"]
        logs_dir = config["logs_dir"]
        if experiment_id == "None":
            return Path(logs_dir) / config['name'] / datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S")
        return Path(logs_dir) / config['name'] / experiment_id / datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S")

    def log_image(self, tag, img, iteration):
        self._writer.add_images(tag=tag, img_tensor=img,
                                global_step=iteration, dataformats="NHWC",


                                )

    def log_text(self, tag, text, iteration):
        self._writer.add_text(tag=tag, text_string=text, global_step=iteration)
