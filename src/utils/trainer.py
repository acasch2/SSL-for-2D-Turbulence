from models.vision_transformer import ViT
from utils.data_loaders import get_dataloader
import os
import time
import logging
import torch


class Trainer():
    def __init__(self, params):
        self.params = params

        # Set cuda/cpu device
        if params["gpu"]:
            self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        # Init wandb if logging

        # Construct training/validation dataloaders
        logging.info("initialize data loader")
        self.train_dataloader, self.train_dataset = get_dataloader(data_dir=params["data_dir"],
                                                                   file_range=params["train_file_range"],
                                                                   target_step=params["target_step"],
                                                                   batch_size=params["batch_size"],
                                                                   train=True,
                                                                   num_workers=params["num_workers"],
                                                                   pin_memory=params["pin_memory"])
        self.valid_dataloader, self.valid_dataset = get_dataloader(data_dir=params["data_dir"],
                                                                   file_range=params["valid_file_range"],
                                                                   target_step=params["target_step"],
                                                                   batch_size=params["batch_size"],
                                                                   train=False,
                                                                   num_workers=params["num_workers"],
                                                                   pin_memory=params["pin_memory"])
        logging.info("data loader initialized")

        # Construct model
        self.model = ViT(
            img_size=params["img_size"],
            patch_size=params["patch_size"],
            num_frames=params["num_frames"],
            tubelet_size=params["tubelet_size"],
            in_chans=params["in_chans"],
            encoder_embed_dim=params["encoder_embed_dim"],
            encoder_depth=params["encoder_depth"],
            encoder_num_heads=params["encoder_num_heads"],
            decoder_embed_dim=params["decoder_embed_dim"],
            decoder_depth=params["decoder_depth"],
            decoder_num_heads=params["decoder_num_heads"],
            mlp_ratio=params["mlp_ratio"],
            num_out_frames=params["num_out_frames"],
            checkpointing=params["checkpointing"]).to(self.device)

        # Watch model gradients with wandb

        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

        
        self.startEpoch = 0
        self.epoch = self.startEpoch

        # Set learning rate scheduluer
        if params["scheduler"] == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params["scheduler"] == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params["max_epochs"],
                                                                        last_epoch=self.startEpoch-1)
        else:
            self.scheduler = None

        # Construct training lr scheduler if using

        if params["log_to_screen"]:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))
            print("Number of trainable model parameters: {}".format(self.count_parameters()))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self):
        if self.params["log_to_screen"]:
            logging.info("Starting training loop ...")

        best_valid_loss = 1.e6
        for epoch in range(self.params["max_epochs"]):
            start = time.time()

            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()

            # Adjust lr rate schedule if using
            if self.params["scheduler"] == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['loss'])
            elif self.params["scheduler"] == 'ConsineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    break

            # Save model checkpoint
            if self.params["save_checkpoint"]:
                # Save at end of every epoch ....
                if valid_logs["loss"] <= best_valid_loss:
                    self.save_checkpoint(self.params["model_save_path"])
                    best_valid_loss = valid_logs["loss"]

            if self.params["log_to_screen"]:
                logging.info("Time taken for epoch {} is {} sec".format(epoch+1, time.time()-start))
                logging.info("Train loss: {}. Valid loss: {}".format(train_logs['loss'], valid_logs["loss"]))

            print("Time taken for epoch {} is {} sec".format(epoch+1, time.time()-start))
            print("Train loss: {}. Valid loss: {}".format(train_logs['loss'], valid_logs["loss"]))
        
        logging.info("----- DONE -----")

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        for i, data in enumerate(self.train_dataloader):

            data_start = time.time()
            inputs, labels = data[0].to(self.device, dtype=torch.float32), data[1].to(self.device, dtype=torch.float32)
            if self.params["checkpointing"]:
                inputs.requires_grad_()

            data_time += time.time() - data_start

            tr_start = time.time()

            self.model.zero_grad()
            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(inputs, train=True)
            loss = self.model.forward_loss(labels, outputs)

            loss.backward()

            self.optimizer.step()

            tr_time += time.time() - tr_start

            logs = {'loss': loss}

        return tr_time, data_time, logs

    def validate_one_epoch(self):
        self.model.eval()
        n_valid_batches = 50

        valid_start = time.time()

        with torch.no_grad():
            for i, data in enumerate(self.valid_dataloader):
                if i >= n_valid_batches:
                    break

                inputs, labels = data[0].to(self.device, dtype=torch.float32), data[1].to(self.device, dtype=torch.float32)

                outputs = self.model(inputs, train=False)
                loss = self.model.forward_loss(labels, outputs)

                # check valid pred
                self.val_pred = outputs

        valid_time = time.time() - valid_start

        logs = {"loss": loss}

        return valid_time, logs

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
