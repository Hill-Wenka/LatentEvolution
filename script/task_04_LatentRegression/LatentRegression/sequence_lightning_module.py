import copy

import lightning as pl
import nni
import omegaconf
import torch
from scipy.stats import pearsonr, spearmanr

from framework.module.optim import get_optimizer, get_scheduler
from script.task_04_LatentRegression.LatentRegression import Regressor


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, args: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.configure_model()
        self.configure_metric()
        self.configure_nni()
        self.configure_loss()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_nni(self):
        self.use_best = self.args.nni.use_best
        self.nni_best_metric = 0
        self.last_model_state_dict = None
        self.best_model_state_dict = None

    def configure_metric(self):
        if self.args.ckpt_callback is not None:
            self.ckpt_best_metric = -1e10 if self.args.ckpt_callback.mode == 'max' else 1e10

    def configure_model(self):
        self.model = Regressor.Net(**self.args.hparams)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = get_optimizer(self.args.optimizer, params)
        optimizers = [optimizer]
        scheduler = get_scheduler(self.args.scheduler, optimizer)
        schedulers = [scheduler]

        if scheduler is None:
            return optimizers
        else:
            return optimizers, schedulers

    def configure_loss(self):
        self.mse_loss = torch.nn.MSELoss()

    def compute_loss(self, pred_ddG, pred_dS, ddG, dS):
        ddG_loss = self.mse_loss(pred_ddG.reshape(-1), ddG.reshape(-1))
        dS_loss = self.mse_loss(pred_dS.reshape(-1), dS.reshape(-1))
        return ddG_loss + dS_loss, ddG_loss, dS_loss

    def batch_forward(self, batch):
        tokens, features, ddG, dS = batch
        pred_ddG, pred_dS = self.model(features)
        loss, ddG_loss, dS_loss = self.compute_loss(pred_ddG, pred_dS, ddG, dS)
        return {'loss': loss, 'ddG_mse': ddG_loss, 'dS_mse': dS_loss, 'pred_ddG': pred_ddG, 'pred_dS': pred_dS, 'ddG': ddG, 'dS': dS}

    def training_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def validation_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def test_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def on_train_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_pred_ddG = batch_outputs['pred_ddG'].detach().cpu().numpy()
        batch_pred_dS = batch_outputs['pred_dS'].detach().cpu().numpy()
        batch_ddG = batch_outputs['ddG'].detach().cpu().numpy()
        batch_dS = batch_outputs['dS'].detach().cpu().numpy()
        ddG_pearsonr = pearsonr(batch_pred_ddG, batch_ddG)[0]
        ddG_spearmanr = spearmanr(batch_pred_ddG, batch_ddG)[0]
        dS_pearsonr = pearsonr(batch_pred_dS, batch_dS)[0]
        dS_spearmanr = spearmanr(batch_pred_dS, batch_dS)[0]

        batch_metrics = {}
        batch_metrics['loss'] = batch_outputs['loss']
        batch_metrics['ddG_mse'] = batch_outputs['ddG_mse']
        batch_metrics['dS_mse'] = batch_outputs['dS_mse']
        batch_metrics['ddG_pearsonr'] = ddG_pearsonr
        batch_metrics['ddG_spearmanr'] = ddG_spearmanr
        batch_metrics['dS_pearsonr'] = dS_pearsonr
        batch_metrics['dS_spearmanr'] = dS_spearmanr
        batch_metrics['avg_pearsonr'] = (ddG_pearsonr + dS_pearsonr) / 2
        batch_metrics['avg_spearmanr'] = (ddG_spearmanr + dS_spearmanr) / 2
        batch_metrics = {'train/' + k + '_step': v for k, v in batch_metrics.items()}

        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.training_step_outputs.append(batch_outputs)

    def on_validation_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_pred_ddG = batch_outputs['pred_ddG'].detach().cpu().numpy()
        batch_pred_dS = batch_outputs['pred_dS'].detach().cpu().numpy()
        batch_ddG = batch_outputs['ddG'].detach().cpu().numpy()
        batch_dS = batch_outputs['dS'].detach().cpu().numpy()
        ddG_pearsonr = pearsonr(batch_pred_ddG, batch_ddG)[0]
        ddG_spearmanr = spearmanr(batch_pred_ddG, batch_ddG)[0]
        dS_pearsonr = pearsonr(batch_pred_dS, batch_dS)[0]
        dS_spearmanr = spearmanr(batch_pred_dS, batch_dS)[0]

        batch_metrics = {}
        batch_metrics['loss'] = batch_outputs['loss']
        batch_metrics['ddG_mse'] = batch_outputs['ddG_mse']
        batch_metrics['dS_mse'] = batch_outputs['dS_mse']
        batch_metrics['ddG_pearsonr'] = ddG_pearsonr
        batch_metrics['ddG_spearmanr'] = ddG_spearmanr
        batch_metrics['dS_pearsonr'] = dS_pearsonr
        batch_metrics['dS_spearmanr'] = dS_spearmanr
        batch_metrics['avg_pearsonr'] = (ddG_pearsonr + dS_pearsonr) / 2
        batch_metrics['avg_spearmanr'] = (ddG_spearmanr + dS_spearmanr) / 2
        batch_metrics = {'valid/' + k + '_step': v for k, v in batch_metrics.items()}

        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.validation_step_outputs.append(batch_outputs)

    def on_test_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_pred_ddG = batch_outputs['pred_ddG'].detach().cpu().numpy()
        batch_pred_dS = batch_outputs['pred_dS'].detach().cpu().numpy()
        batch_ddG = batch_outputs['ddG'].detach().cpu().numpy()
        batch_dS = batch_outputs['dS'].detach().cpu().numpy()
        ddG_pearsonr = pearsonr(batch_pred_ddG, batch_ddG)[0]
        ddG_spearmanr = spearmanr(batch_pred_ddG, batch_ddG)[0]
        dS_pearsonr = pearsonr(batch_pred_dS, batch_dS)[0]
        dS_spearmanr = spearmanr(batch_pred_dS, batch_dS)[0]

        batch_metrics = {}
        batch_metrics['loss'] = batch_outputs['loss']
        batch_metrics['ddG_mse'] = batch_outputs['ddG_mse']
        batch_metrics['dS_mse'] = batch_outputs['dS_mse']
        batch_metrics['ddG_pearsonr'] = ddG_pearsonr
        batch_metrics['ddG_spearmanr'] = ddG_spearmanr
        batch_metrics['dS_pearsonr'] = dS_pearsonr
        batch_metrics['dS_spearmanr'] = dS_spearmanr
        batch_metrics['avg_pearsonr'] = (ddG_pearsonr + dS_pearsonr) / 2
        batch_metrics['avg_spearmanr'] = (ddG_spearmanr + dS_spearmanr) / 2
        batch_metrics = {'test/' + k + '_step': v for k, v in batch_metrics.items()}

        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.test_step_outputs.append(batch_outputs)

    def on_train_epoch_end(self):
        batch_outputs = self.training_step_outputs
        pred_ddG = torch.cat([batch['pred_ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        pred_dS = torch.cat([batch['pred_dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        ddG = torch.cat([batch['ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        dS = torch.cat([batch['dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        ddG_pearsonr = pearsonr(pred_ddG, ddG)[0]
        ddG_spearmanr = spearmanr(pred_ddG, ddG)[0]
        dS_pearsonr = pearsonr(pred_dS, dS)[0]
        dS_spearmanr = spearmanr(pred_dS, dS)[0]

        epoch_metrics = {}
        epoch_metrics['loss'] = torch.tensor([batch['loss'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ddG_mse'] = torch.tensor([batch['ddG_mse'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['dS_mse'] = torch.tensor([batch['dS_mse'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ddG_pearsonr'] = ddG_pearsonr
        epoch_metrics['ddG_spearmanr'] = ddG_spearmanr
        epoch_metrics['dS_pearsonr'] = dS_pearsonr
        epoch_metrics['dS_spearmanr'] = dS_spearmanr
        epoch_metrics['avg_pearsonr'] = (ddG_pearsonr + dS_pearsonr) / 2
        epoch_metrics['avg_spearmanr'] = (ddG_spearmanr + dS_spearmanr) / 2
        epoch_metrics = {'train/' + k + '_epoch': v for k, v in epoch_metrics.items()}

        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        batch_outputs = self.validation_step_outputs
        pred_ddG = torch.cat([batch['pred_ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        pred_dS = torch.cat([batch['pred_dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        ddG = torch.cat([batch['ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        dS = torch.cat([batch['dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        ddG_pearsonr = pearsonr(pred_ddG, ddG)[0]
        ddG_spearmanr = spearmanr(pred_ddG, ddG)[0]
        dS_pearsonr = pearsonr(pred_dS, dS)[0]
        dS_spearmanr = spearmanr(pred_dS, dS)[0]

        epoch_metrics = {}
        epoch_metrics['loss'] = torch.tensor([batch['loss'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ddG_mse'] = torch.tensor([batch['ddG_mse'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['dS_mse'] = torch.tensor([batch['dS_mse'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ddG_pearsonr'] = ddG_pearsonr
        epoch_metrics['ddG_spearmanr'] = ddG_spearmanr
        epoch_metrics['dS_pearsonr'] = dS_pearsonr
        epoch_metrics['dS_spearmanr'] = dS_spearmanr
        epoch_metrics['avg_pearsonr'] = (ddG_pearsonr + dS_pearsonr) / 2
        epoch_metrics['avg_spearmanr'] = (ddG_spearmanr + dS_spearmanr) / 2
        epoch_metrics = {'valid/' + k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)

        if self.args.ckpt_callback is not None:
            assert self.args.nni.auto_ml is False, 'ckpt_callback and nni.auto_ml cannot be used together'
            ckpt_metric = epoch_metrics[self.args.ckpt_callback.monitor].item()
            if self.args.ckpt_callback.mode == 'max' and ckpt_metric > self.ckpt_best_metric:
                self.ckpt_best_metric = ckpt_metric
            elif self.args.ckpt_callback.mode == 'min' and ckpt_metric < self.ckpt_best_metric:
                self.ckpt_best_metric = ckpt_metric
            else:
                pass

        if self.args.nni.auto_ml:
            assert self.args.ckpt_callback is None, 'ckpt_callback and nni.auto_ml cannot be used together'
            nni_metric = epoch_metrics['valid_' + self.args.nni.metric + '_epoch'].item()
            nni.report_intermediate_result(nni_metric)
            if self.use_best and nni_metric > self.nni_best_metric:
                self.nni_best_metric = nni_metric
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                pass

        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        batch_outputs = self.test_step_outputs
        pred_ddG = torch.cat([batch['pred_ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        pred_dS = torch.cat([batch['pred_dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        ddG = torch.cat([batch['ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        dS = torch.cat([batch['dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
        ddG_pearsonr = pearsonr(pred_ddG, ddG)[0]
        ddG_spearmanr = spearmanr(pred_ddG, ddG)[0]
        dS_pearsonr = pearsonr(pred_dS, dS)[0]
        dS_spearmanr = spearmanr(pred_dS, dS)[0]

        epoch_metrics = {}
        epoch_metrics['loss'] = torch.tensor([batch['loss'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ddG_mse'] = torch.tensor([batch['ddG_mse'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['dS_mse'] = torch.tensor([batch['dS_mse'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ddG_pearsonr'] = ddG_pearsonr
        epoch_metrics['ddG_spearmanr'] = ddG_spearmanr
        epoch_metrics['dS_pearsonr'] = dS_pearsonr
        epoch_metrics['dS_spearmanr'] = dS_spearmanr
        epoch_metrics['avg_pearsonr'] = (ddG_pearsonr + dS_pearsonr) / 2
        epoch_metrics['avg_spearmanr'] = (ddG_spearmanr + dS_spearmanr) / 2
        epoch_metrics = {'test/' + k + '_epoch': v for k, v in epoch_metrics.items()}

        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.test_step_outputs = []

        if self.args.nni.auto_ml:
            nni_metric = epoch_metrics['test_' + self.args.nni.metric + '_epoch'].item()
            nni.report_final_result(nni_metric)

    def on_test_start(self):
        if self.args.nni.auto_ml and self.use_best:
            print('=' * 10, 'use the model of best valid performance to test', '=' * 10)
            assert self.best_model_state_dict is not None, 'Error: self.best_model_state_dict is None'
            self.last_model_state_dict = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(self.best_model_state_dict)

    def predict_step(self, batch, batch_idx, **kwargs):
        tokens, features, ddG, dS = batch
        pred_ddG, pred_dS = self.model(features)
        return {
            'tokens': tokens,
            'features': features,
            'ddG': ddG,
            'dS': dS,
            'pred_ddG': pred_ddG,
            'pred_dS': pred_dS,
        }
