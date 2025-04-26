from typing import Type
import numpy as np
import torch
import yaml
from cprint import *
from tqdm import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_normal


from datasets.base_dataset import BaseDataSet
from training.DataLoaderHandler import DataLoaderHandler
from training.Logger import Logger
from training.ModelBuilder import ModelBuilder
from training.TrainingVariables import TrainingVariables
from datasets.SimMatches import SimMatches
from lutils.wandb import Wandb


class ModelTrainer:
    def __init__(self, dataset_type: Type[BaseDataSet] = SimMatches, configs_path="./configs/global_configs.yaml", options: dict = {}):
        self.dataset_type = dataset_type
        self.options = options
        self.tqdm = tqdm_notebook if self.options.get(
            "tdm_notebook", False) else tqdm_normal
        with open(configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
        self.training_config = self.global_configs["training"]
        self._set_device()
        self.logger = Logger(self.global_configs)
        self.experiment_dir = self.logger.experiment_dir

        self.data_loader_handler = DataLoaderHandler(global_configs=self.global_configs,
                                                     batch_size=self.training_config['batch_size'],
                                                     test_size=self.training_config['test_size'],
                                                     num_workers=self.training_config['num_workers'])
        self.dataset_type = self.data_loader_handler.dataset_type

        self.train_dataloader, self.validation_dataloader = self.data_loader_handler.get_dataloaders()

        self.train_vars = TrainingVariables(experiment_dir=self.experiment_dir, train_loss_running=0.,
                                            best_loss_val=np.inf,
                                            start_iteration=self.training_config["start_iteration"], last_loss=0.)
        self.model = ModelBuilder(
            self.global_configs["model"], self.training_config).get_model()

        if (self.global_configs["model"]["model_field"] != "decoder_transformer"):
            self.logger.log_model_summary(
                self.model, batch_size=self.training_config["batch_size"])
        self.model.to(self.device)

        self.is_overfit = self.global_configs["dataset"]["is_overfit"]
        self.logger.wandb.set_cfg(self.global_configs)
        self.logger.wandb.init()
       # self.test_reid = self.data_loader_handler.test_reid

    def _train_one_epoch(self, epoch):
        train_loss_running = self._init_train_loss_dict()
        batch_iteration = 0
        self.logger.start_new_epoch(epoch)
        additional_metrics_dict_train = self._init_metrics_dict(
            self.model.get_additional_metrics()
        )
        self.model.set_epoch(epoch)
        metric_batch_indices = np.random.randint(len(
            self.train_dataloader), size=self.training_config["apply_metrics_batch_count"])

        enumartion = None
        if (self.is_overfit):
            enumartion = enumerate(self.train_dataloader)
        else:
            enumartion = self.tqdm(
                enumerate(self.train_dataloader), total=len(self.train_dataloader))
        self.model.set_iter_per_epoch(len(self.train_dataloader))
        for batch_idx, batch in enumartion:
            # for batch_idx, batch in enumerate(self.train_dataloader):
            self.model.train()
            # torch.cuda.empty_cache()
            iteration = epoch * len(self.train_dataloader) + batch_idx
            self.logger.wandb.set_global_step(iteration)
            if iteration < self.train_vars.start_iteration:
                continue
            batch = self.dataset_type.move_batch_to_device(batch, self.device)
            x = self.model.get_batch_input(batch)
            self.model.set_iteration(iteration)
            self.model.step(x)
            losses = self.model.get_metrics()
            loss = losses.get("loss")
            # iou = losses.get("signedIou")
            train_loss_running["loss"] += loss
            # train_loss_running["signedIou"] += iou
            self._add_losses_to_dict(train_loss_running)
            batch_iteration += 1
            # if (self.training_config['use_scheduler']):
            #     self.model.update_lr()

            # if (batch_idx in metric_batch_indices):
            #     cprint.ok(
            #         f"Calculating additional metrics for training batch {batch_idx}")
            #     additional_metrics = self.model.calculate_additional_metrics()
            #     for key in additional_metrics.keys():
            #         additional_metrics_dict_train[key] = additional_metrics.get(
            #             key)
            #         for key in additional_metrics_dict_train.keys():
            #             self.logger.add_scalar(
            #                 f"Train/{key}", additional_metrics_dict_train[key], iteration)

            # log loss
            self.logger.log_loss(epoch, iteration, loss)

            intial_pass = epoch == 0 and iteration == 0
            # visualization step
            if (iteration % self.training_config["visualize_every"] == (self.training_config["visualize_every"] - 1) or intial_pass):

                players_stack, goalkeepers_stack, main_referee_stack, side_referee_stack, staff_members_stack = self.model.prepare_visuals()
                if (players_stack is not None):
                    self.logger.log_grid_images(
                        "Train/Players", players_stack, iteration)
                if (goalkeepers_stack is not None):
                    self.logger.log_grid_images(
                        "Train/GK", goalkeepers_stack, iteration)
                if (main_referee_stack is not None):
                    self.logger.log_grid_images(
                        "Train/MainRef", main_referee_stack, iteration)
                if (side_referee_stack is not None):
                    self.logger.log_grid_images(
                        "Train/SideRef", side_referee_stack, iteration)
                if (staff_members_stack is not None):
                    self.logger.log_grid_images(
                        "Train/Staff", staff_members_stack, iteration)

                # m, um = self.model.prepare_visuals()
                # if m is not None:
                #     self.logger.log_grid_images("Train/Matches", m, iteration)
                # if um is not None:
                #     self.logger.log_grid_images(
                #         "Train/Non-Matches", um, iteration)

                # m = self.model.prepare_visuals()
                # if m is not None:
                #     self.logger.log_grid_images(
                #         "Train/LegibleImages", m, iteration)

                # q, g = self.model.prepare_visuals()
                # if (q is not None):
                #     self.logger.log_grid_images(
                #         "Train/query", q, iteration)
                # if (g is not None):
                #     self.logger.log_grid_images(
                #         "Train/gallery", g, iteration)

                # self.visualizer.visualize(visuals, epoch, iteration)
                # t0, t1, t2 = self.model.prepare_visuals()
                # if t0 is not None:
                #     self.logger.log_grid_images("Train/T0", t0, iteration)
                # if t1 is not None:
                #     self.logger.log_grid_images(
                #         "Train/T1", t1, iteration)
                # if t2 is not None:
                #     self.logger.log_grid_images(
                #         "Train/T2", t2, iteration)

                # log writer
            if iteration % self.training_config['print_every'] == (self.training_config['print_every'] - 1) or (
                    intial_pass):
                avg_train_loss = self._avg_losses(
                    train_loss_running, batch_iteration)
                # If we want to be 100% accurate in the train vs valid comparison ->
                # do an inference run on the train data and log the loss then, however not worth it
                cprint.warn(
                    f'[{epoch:03d}/{batch_idx:05d}] train_loss: {avg_train_loss["loss"]:.6f}')
                self.logger.add_scalar(
                    "Train/Loss", avg_train_loss["loss"], iteration)
                model_field = self.global_configs["model"]["model_field"]

                self.logger.add_scalar(
                    "Train/LR", self.model.optimizer.param_groups[0]['lr'], iteration)
                self._add_scalars(avg_train_loss, iteration)
                self.train_vars.last_loss = avg_train_loss
                train_loss_running = self._init_train_loss_dict()
                batch_iteration = 0

            # saving step
            if iteration % self.training_config['save_every'] == (self.training_config['save_every'] - 1):
                self.model.save(
                    self.train_vars.model_checkpoint_path, "latest")
                pass

            # validation step
            if iteration % self.training_config['validate_every'] == (self.training_config['validate_every'] - 1) or (
                    intial_pass) and not self.is_overfit:
                cprint.ok("Running Validation")
                self.model.eval()

                val_loss_running = 0.
                index_batch = 0
                losses_dict = self._init_metrics_dict(
                    self.model.get_additional_losses())
                additional_metrics_dict = self._init_metrics_dict(
                    self.model.get_additional_metrics()
                )
                metric_batch_indices = np.random.randint(len(
                    self.validation_dataloader), size=self.training_config["apply_metrics_batch_count"])
               # Also use self.validation_dataloader['soccernetv3']['gallery']

                for batch_index, batch_val in self.tqdm(enumerate(self.validation_dataloader), total=len(self.validation_dataloader)):
                    with torch.no_grad():
                        apply_additional_metrics = False and (
                            batch_index in metric_batch_indices)
                        self.dataset_type.move_batch_to_device(
                            batch_val, self.device)
                        self.model.inference(
                            self.model.get_batch_input(batch_val))
                        self.model.set_loss()
                        metrics = self.model.get_metrics()
                        val_loss = metrics.get("loss")
                        val_loss_running += val_loss
                        # val_iou_running = metrics.get("signedIou")
                        for key in metrics.keys():
                            if (key == "loss" or key == "signedIou"):
                                continue
                            losses_dict[key] += metrics.get(key)
                        if (apply_additional_metrics):
                            cprint.ok(
                                f"Calculating additional metrics for batch {batch_index}")
                            # additional_metrics = self.model.calculate_additional_metrics()
                            # for key in additional_metrics.keys():
                            #     additional_metrics_dict[key] += additional_metrics.get(
                            #         key)
                        index_batch += 1
                avg_loss_val = val_loss_running / index_batch
               # avg_iou_val = val_iou_running / index_batch
                for key in losses_dict.keys():
                    losses_dict[key] = losses_dict[key] / index_batch

                # for key in additional_metrics_dict.keys():
                #     additional_metrics_dict[key] = additional_metrics_dict[key] / \
                #         metric_batch_indices.shape[0]

                # Do visualizations here
                if avg_loss_val < self.train_vars.best_loss_val:
                    self.model.save(
                        self.train_vars.model_checkpoint_path, "best")
                    self.train_vars.best_loss_val = avg_loss_val

                cprint.warn(
                    f'[{epoch:03d}/{batch_idx:05d}] val_loss: {avg_loss_val:.6f} | best_loss_val: '
                    f'{self.train_vars.best_loss_val:.6f}')
                self.logger.add_scalar(
                    "Validation/Loss", avg_loss_val, iteration)
                self.logger.add_scalars('Validation/LossComparison',
                                        {'Training':  self.train_vars.last_loss["loss"],
                                            'Validation': avg_loss_val},
                                        iteration)
                for key in losses_dict.keys():
                    self.logger.add_scalar(
                        f"Validation/{key}", losses_dict[key], iteration)

                for key in additional_metrics_dict.keys():
                    self.logger.add_scalar(
                        f"Validation/{key}", additional_metrics_dict[key], iteration)
                self.logger.flush_writer()
                self.logger.flush_writer()

    def _init_train_loss_dict(self):
        train_loss_running = {
            "loss": 0.0,

        }
        model_field = self.global_configs["model"]["model_field"]
        model_configs = self.global_configs["model"][model_field]
        losses = model_configs.get("losses", None)
        if (losses is None):
            return train_loss_running
        losses = losses.split(",")
        for loss in losses:
            train_loss_running[loss] = 0.
        return train_loss_running

    def _avg_losses(self, loss_dict, iteration):
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / iteration
        return loss_dict

    def _set_device(self):
        self.device = torch.device('cpu')
        if torch.cuda.is_available() and self.training_config['device'].startswith('cuda'):
            self.device = torch.device(self.training_config['device'])
            cprint.ok('Using device:', self.training_config['device'])
        else:
            cprint.warn('Using CPU')

    def _init_metrics_dict(self, keys):
        metrics_dict = {}
        for metric_key in keys:
            metrics_dict[metric_key] = 0.0
        return metrics_dict

    def _add_scalars(self, avgs_dict, iteration):
        for key in avgs_dict.keys():
            if (key == "loss"):
                continue
            self.logger.add_scalar(
                f"Train/{key}", avgs_dict[key], iteration)

    def _add_losses_to_dict(self, dict):
        metrics = self.model.get_metrics()
        for key in metrics:
            if (key == "loss"):
                continue
            dict[key] += metrics[key]

    def train(self):
        start_epoch = self.training_config["start_epoch"]
        for epoch in self.tqdm(range(self.training_config['n_epochs'])):
            self.logger.wandb.set_epoch(epoch)
            if epoch < start_epoch:
                continue
            self._train_one_epoch(epoch)
            if epoch % self.training_config["save_every_nepochs"] == 0 and not self.is_overfit:
                self.model.save(self.train_vars.model_checkpoint_path, epoch)
            # if (self.training_config['use_scheduler']):
            #     self.model.update_lr()
            self.logger.close_writer()
        self.model.save(self.train_vars.model_checkpoint_path)
