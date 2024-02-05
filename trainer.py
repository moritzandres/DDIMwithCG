import pytorch_lightning as pl
import torch
from tqdm import tqdm, trange
import os
import os.path as osp
import matplotlib.pyplot as plt


class Trainer:

    """ similar to pytorch-lighting trainer """

    def __init__(self, max_epochs, log_dir='./my_logs', early_stopping=None) -> None:
        self.epoch = max_epochs
        self.log_dir = log_dir
        self.e_stop = early_stopping if early_stopping else max_epochs
        self.csv_file = osp.join(self.log_dir, 'metrics.csv')
        self.plot_file = osp.join(self.log_dir, 'metrics.png')
        self.weights_file = osp.join(self.log_dir, 'weights.pth')
        self.metrics_hist = {'epoch': [], 'loss': [], 'val_loss': []}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, model:pl.LightningModule, datamodule=None, train_dataloaders=None, val_dataloaders=None):

        # clean up log directory
        self.metrics_hist = {'epoch': [], 'loss': [], 'val_loss': []}
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.csv_file, 'w') as f:
            f.write('epoch,loss,val_loss\n')

        es_counter = 0  # early stopping counter

        model = model.to(self.device)

        if datamodule:
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
        elif train_dataloaders and val_dataloaders:
            train_loader = train_dataloaders
            val_loader = val_dataloaders

        min_val_loss = float('inf')
        t = trange(self.epoch, desc='', leave=True)
        for epoch in t:

            model.train()
            loss = 0
            for i, batch in enumerate(train_loader):
                if type(batch) is dict:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)

                l = model.training_step(batch, i)
                loss += l.item()
            loss = loss / len(train_loader)

            model.eval()
            val_loss = 0
            for i, batch in enumerate(val_loader):
                if type(batch) is dict:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)

                vl = model.validation_step(batch, i)
                val_loss += vl.item()
            val_loss = val_loss / len(val_loader)

            # save best model with min val_loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                es_counter = 0
                torch.save(model.state_dict(), self.weights_file)
            else:
                es_counter += 1
            
            # early stopping
            if es_counter > self.e_stop:
                print(f"Early stopping at epoch {epoch}")
                break

            # update tqdm bar
            t.set_description(f"Epoch: {epoch}, loss: {loss:.4f}, val_loss: {val_loss:.4f}")

            # add line to csv file
            with open(self.csv_file, 'a') as f:
                f.write(f"{epoch},{loss},{val_loss}\n")
            
            # add to metrics_hist
            self.metrics_hist['epoch'].append(epoch)
            self.metrics_hist['val_loss'].append(val_loss)
            self.metrics_hist['loss'].append(loss)
            
            # plot metrics
            if epoch > 1:
                plt.plot(self.metrics_hist['epoch'], self.metrics_hist['loss'], label='loss')
                plt.plot(self.metrics_hist['epoch'], self.metrics_hist['val_loss'], label='val_loss')
                plt.xlabel('epoch')
                plt.legend()
                plt.savefig(self.plot_file)
                plt.close()

        return model