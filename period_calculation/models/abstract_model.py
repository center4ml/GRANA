import pytorch_lightning as pl
import torch

from hydra.utils import instantiate


class AbstractModel(pl.LightningModule):
    def __init__(self,
                 lr=0.001,
                 optimizer_hparams=dict(),
                 scheduler=dict(classname='MultiStepLR', kwargs=dict(milestones=[100, 150], gamma=0.1))
                 ):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

    def forward(self, x):
        raise NotImplementedError("Subclass needs to implement this method")

    def configure_optimizers(self):
        # AdamW is Adam with a correct implementation of weight decay (see here
        # for details: https://arxiv.org/pdf/1711.05101.pdf)
        print("configuring the optimizer and lr scheduler with learning rate=%.5f"%self.hparams.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, **self.hparams.optimizer_hparams)
        # scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_hparams['classname'])(optimizer, **self.hparams.lr_hparams['kwargs'])
        if self.hparams.scheduler is not None:
            scheduler = instantiate({**self.hparams.scheduler, '_partial_': True})(optimizer)

            return [optimizer], [scheduler]
        else:
            return optimizer

    def additional_losses(self):
        """get additional_losses"""
        return torch.zeros((1))

    def process_batch_supervised(self, batch):
        """get predictions, losses and mean errors (MAE)"""
        raise NotImplementedError("Subclass needs to implement this method")

    def log_all(self, losses, metrics, prefix=''):
        for k, v in losses.items():
            self.log(f'{prefix}{k}_loss', v.item() if isinstance(v, torch.Tensor) else v)

        for k, v in metrics.items():
            self.log(f'{prefix}{k}', v.item() if isinstance(v, torch.Tensor) else v)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='train_')

        return losses['final']

    def validation_step(self, batch, batch_idx):
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='val_')

    def test_step(self, batch, batch_idx):
        preds, losses, metrics = self.process_batch_supervised(batch)
        self.log_all(losses, metrics, prefix='test_')