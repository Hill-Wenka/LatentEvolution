import os

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from framework.utils.config.config_utils import load_args
from framework.utils.lightning.device_utils import seed_everything


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.leave = True
        return bar


def get_pl_trainer(args) -> pl.Trainer:
    progress_bar_callback = LitProgressBar(**args.progress_bar_callback)
    logger = None if args.logger.save_dir is None else [TensorBoardLogger(**args.logger)]
    checkpoint_callback = None if args.ckpt_callback is None else ModelCheckpoint(**args.ckpt_callback)
    early_stop_callback = None if args.early_stop_callback is None else EarlyStopping(**args.early_stop_callback)
    swa_callback = None if args.swa_callback is None or args.swa_callback == '' else StochasticWeightAveraging(
        **args.swa_callback
    )

    callbacks = [checkpoint_callback, early_stop_callback, progress_bar_callback, swa_callback]
    callbacks = [x for x in callbacks if x is not None]

    if args.trainer.strategy is not None and 'ddp' in args.trainer.strategy:
        print('Using DDPStrategy')
        ddp_strategy = DDPStrategy(find_unused_parameters=False)
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            strategy=ddp_strategy,
            **args.trainer
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            **args.trainer
        )
    return trainer


def get_ckpt(log_dir, version, epoch):
    if type(version) == int:
        version = f'version_{version}'

    if type(epoch) == str and 'last' in epoch:
        string = epoch
    else:
        if type(epoch) == int and epoch < 10:
            epoch = f'0{epoch}'
        string = f'epoch={epoch}'

    args_dir = log_dir + f'/{version}/'
    ckpt_dir = log_dir + f'/{version}/checkpoints/'
    files = os.listdir(ckpt_dir)
    for file in files:
        if string in file:
            return ckpt_dir + file, args_dir + 'hparams.yaml'


class LitInference:
    def __init__(self, LightningModule, DataModule, log_dir, version, epoch):
        super(LitInference, self).__init__()
        ckpt_params, ckpt_args = get_ckpt(log_dir, version, epoch)

        self.ckpt_args = load_args(ckpt_args)
        self.ckpt_model = LightningModule.load_from_checkpoint(ckpt_params)
        self.pl_data_module = DataModule(self.ckpt_args, log=True)
        self.ckpt_trainer = get_pl_trainer(self.ckpt_args)
        self.ckpt_trainer.logger = None
        self.model = self.ckpt_model.model

        seed_everything(self.ckpt_args.seed)

    def get_model(self):
        return self.model

    def set_batch_size(self, batch_size=None, num_workers=20):
        if batch_size is None:
            batch_size = self.ckpt_args.predict_dataloader.batch_size
        self.ckpt_args.predict_dataloader.batch_size = batch_size
        self.ckpt_args.predict_dataloader.num_workers = num_workers
        self.ckpt_args.predict_dataloader.pin_memory = False
        self.ckpt_args.predict_dataloader.persistent_workers = False
        torch.multiprocessing.set_sharing_strategy('file_system')

    def predict(self, predict_data):
        assert predict_data is not None, 'predict_data is None'
        self.pl_data_module.prepare_data('predict', predict_data=predict_data)
        self.pl_data_module.setup('test')
        predictions = self.ckpt_trainer.predict(
            model=self.ckpt_model, dataloaders=self.pl_data_module.predict_dataloader(), return_predictions=True
        )
        return predictions
