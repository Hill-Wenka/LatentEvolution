import copy

import lightning.pytorch as pl
import nni
import omegaconf
import torch
from scipy.stats import pearsonr, spearmanr

from framework.module.loss import get_loss
from framework.module.optim import get_optimizer, get_scheduler
from script.task_02_ProteinVAE.ProteinVAE import ProteinVAE


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
        self.model = ProteinVAE.ProteinVAE(**self.args.hparams)

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
        self.ce_loss = get_loss(self.args.loss.ce_loss, reduction='sum')
        self.mse_loss = get_loss(self.args.loss.mse_loss, reduction='sum')
        self.mmd_loss = get_loss(self.args.loss.mmd_loss, reduction='sum')

    def compute_loss(self, recon_x, x, mu, logvar, pred_ddG=None, pred_dS=None, ddG=None, dS=None):
        recon_x = recon_x.reshape(-1, recon_x.shape[-1])
        x = x.reshape(-1)
        CE = self.ce_loss(recon_x, x) * self.args.loss.ce_weight
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # REG = KLD
        MMD = self.mmd_loss(mu)
        REG = MMD * self.args.loss.reg_weight
        if pred_ddG is not None and pred_dS is not None:
            # ddG_loss = self.mse_loss(pred_ddG.reshape(-1), ddG.reshape(-1))
            # dS_loss = self.mse_loss(pred_dS.reshape(-1), dS.reshape(-1))
            # MSE = ddG_loss + dS_loss
            ddG_pearsonr = torch.corrcoef(torch.stack([pred_ddG.reshape(-1), ddG.reshape(-1)]))[0, 1]
            dS_pearsonr = torch.corrcoef(torch.stack([pred_dS.reshape(-1), dS.reshape(-1)]))[0, 1]
            MSE = 1 - (torch.abs(ddG_pearsonr) + torch.abs(dS_pearsonr)) / 2
        else:
            MSE = 0
        MSE = MSE * self.args.loss.mse_weight
        return CE + REG + MSE, CE, REG, MSE

    def batch_forward(self, batch):
        tokens, ddG, dS = batch
        logits, mean, logvar, pred_ddG, pred_dS = self.model(tokens)
        mask = torch.ones_like(tokens)  # [N, L]
        mask[tokens <= 2] = 0  # 0, 1, 2分别是cls, pad, eos
        tokens = tokens * mask  # [N, L]
        logits = logits * mask.unsqueeze(-1)  # [N, L, V]
        loss, ce, reg, mse = self.compute_loss(logits, tokens, mean, logvar, pred_ddG, pred_dS, ddG, dS)
        loss = loss / tokens.shape[0]
        ce = ce / tokens.shape[0]
        reg = reg / tokens.shape[0]
        mse = mse / tokens.shape[0]
        return {'loss': loss, 'ce': ce, 'reg': reg, 'mse': mse, 'pred_ddG': pred_ddG, 'pred_dS': pred_dS, 'ddG': ddG, 'dS': dS}

    def training_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def validation_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def test_step(self, batch, batch_idx):
        return self.batch_forward(batch)

    def on_train_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = {}
        batch_metrics['loss'] = batch_outputs['loss']
        batch_metrics['ce'] = batch_outputs['ce']
        batch_metrics['reg'] = batch_outputs['reg']
        batch_metrics['mse'] = batch_outputs['mse']

        if batch_outputs['pred_ddG'] is not None and batch_outputs['pred_dS'] is not None:
            batch_pred_ddG = batch_outputs['pred_ddG'].detach().cpu().numpy()
            batch_pred_dS = batch_outputs['pred_dS'].detach().cpu().numpy()
            batch_ddG = batch_outputs['ddG'].detach().cpu().numpy()
            batch_dS = batch_outputs['dS'].detach().cpu().numpy()
            ddG_pearsonr = pearsonr(batch_pred_ddG, batch_ddG)[0]
            ddG_spearmanr = spearmanr(batch_pred_ddG, batch_ddG)[0]
            dS_pearsonr = pearsonr(batch_pred_dS, batch_dS)[0]
            dS_spearmanr = spearmanr(batch_pred_dS, batch_dS)[0]
            batch_metrics['ddG_pearsonr'] = ddG_pearsonr
            batch_metrics['ddG_spearmanr'] = ddG_spearmanr
            batch_metrics['dS_pearsonr'] = dS_pearsonr
            batch_metrics['dS_spearmanr'] = dS_spearmanr
            batch_metrics['avg_pearsonr'] = (abs(ddG_pearsonr) + abs(dS_pearsonr)) / 2
            batch_metrics['avg_spearmanr'] = (abs(ddG_spearmanr) + abs(dS_spearmanr)) / 2

        batch_metrics = {'train/' + k + '_step': v for k, v in batch_metrics.items()}
        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.training_step_outputs.append(batch_outputs)

    def on_validation_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = {}
        batch_metrics['loss'] = batch_outputs['loss']
        batch_metrics['ce'] = batch_outputs['ce']
        batch_metrics['reg'] = batch_outputs['reg']
        batch_metrics['mse'] = batch_outputs['mse']

        if batch_outputs['pred_ddG'] is not None and batch_outputs['pred_dS'] is not None:
            batch_pred_ddG = batch_outputs['pred_ddG'].detach().cpu().numpy()
            batch_pred_dS = batch_outputs['pred_dS'].detach().cpu().numpy()
            batch_ddG = batch_outputs['ddG'].detach().cpu().numpy()
            batch_dS = batch_outputs['dS'].detach().cpu().numpy()
            ddG_pearsonr = pearsonr(batch_pred_ddG, batch_ddG)[0]
            ddG_spearmanr = spearmanr(batch_pred_ddG, batch_ddG)[0]
            dS_pearsonr = pearsonr(batch_pred_dS, batch_dS)[0]
            dS_spearmanr = spearmanr(batch_pred_dS, batch_dS)[0]
            batch_metrics['ddG_pearsonr'] = ddG_pearsonr
            batch_metrics['ddG_spearmanr'] = ddG_spearmanr
            batch_metrics['dS_pearsonr'] = dS_pearsonr
            batch_metrics['dS_spearmanr'] = dS_spearmanr
            batch_metrics['avg_pearsonr'] = (abs(ddG_pearsonr) + abs(dS_pearsonr)) / 2
            batch_metrics['avg_spearmanr'] = (abs(ddG_spearmanr) + abs(dS_spearmanr)) / 2

        batch_metrics = {'valid/' + k + '_step': v for k, v in batch_metrics.items()}

        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.validation_step_outputs.append(batch_outputs)

    def on_test_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = {}
        batch_metrics['loss'] = batch_outputs['loss']
        batch_metrics['ce'] = batch_outputs['ce']
        batch_metrics['reg'] = batch_outputs['reg']
        batch_metrics['mse'] = batch_outputs['mse']

        if batch_outputs['pred_ddG'] is not None and batch_outputs['pred_dS'] is not None:
            batch_pred_ddG = batch_outputs['pred_ddG'].detach().cpu().numpy()
            batch_pred_dS = batch_outputs['pred_dS'].detach().cpu().numpy()
            batch_ddG = batch_outputs['ddG'].detach().cpu().numpy()
            batch_dS = batch_outputs['dS'].detach().cpu().numpy()
            ddG_pearsonr = pearsonr(batch_pred_ddG, batch_ddG)[0]
            ddG_spearmanr = spearmanr(batch_pred_ddG, batch_ddG)[0]
            dS_pearsonr = pearsonr(batch_pred_dS, batch_dS)[0]
            dS_spearmanr = spearmanr(batch_pred_dS, batch_dS)[0]
            batch_metrics['ddG_pearsonr'] = ddG_pearsonr
            batch_metrics['ddG_spearmanr'] = ddG_spearmanr
            batch_metrics['dS_pearsonr'] = dS_pearsonr
            batch_metrics['dS_spearmanr'] = dS_spearmanr
            batch_metrics['avg_pearsonr'] = (abs(ddG_pearsonr) + abs(dS_pearsonr)) / 2
            batch_metrics['avg_spearmanr'] = (abs(ddG_spearmanr) + abs(dS_spearmanr)) / 2

        batch_metrics = {'test/' + k + '_step': v for k, v in batch_metrics.items()}

        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.test_step_outputs.append(batch_outputs)

    def on_train_epoch_end(self):
        batch_outputs = self.training_step_outputs

        epoch_metrics = {}
        epoch_metrics['loss'] = torch.tensor([batch['loss'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ce'] = torch.tensor([batch['ce'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['reg'] = torch.tensor([batch['reg'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['mse'] = torch.tensor([batch['mse'] for batch in batch_outputs]).mean().detach()

        if batch_outputs[0]['pred_ddG'] is not None and batch_outputs[0]['pred_dS'] is not None:
            pred_ddG = torch.cat([batch['pred_ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            pred_dS = torch.cat([batch['pred_dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            ddG = torch.cat([batch['ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            dS = torch.cat([batch['dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            ddG_pearsonr = pearsonr(pred_ddG, ddG)[0]
            ddG_spearmanr = spearmanr(pred_ddG, ddG)[0]
            dS_pearsonr = pearsonr(pred_dS, dS)[0]
            dS_spearmanr = spearmanr(pred_dS, dS)[0]
            epoch_metrics['ddG_pearsonr'] = ddG_pearsonr
            epoch_metrics['ddG_spearmanr'] = ddG_spearmanr
            epoch_metrics['dS_pearsonr'] = dS_pearsonr
            epoch_metrics['dS_spearmanr'] = dS_spearmanr
            epoch_metrics['avg_pearsonr'] = (abs(ddG_pearsonr) + abs(dS_pearsonr)) / 2
            epoch_metrics['avg_spearmanr'] = (abs(ddG_spearmanr) + abs(dS_spearmanr)) / 2

        epoch_metrics = {'train/' + k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        batch_outputs = self.validation_step_outputs

        epoch_metrics = {}
        epoch_metrics['loss'] = torch.tensor([batch['loss'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ce'] = torch.tensor([batch['ce'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['reg'] = torch.tensor([batch['reg'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['mse'] = torch.tensor([batch['mse'] for batch in batch_outputs]).mean().detach()

        if batch_outputs[0]['pred_ddG'] is not None and batch_outputs[0]['pred_dS'] is not None:
            pred_ddG = torch.cat([batch['pred_ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            pred_dS = torch.cat([batch['pred_dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            ddG = torch.cat([batch['ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            dS = torch.cat([batch['dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            ddG_pearsonr = pearsonr(pred_ddG, ddG)[0]
            ddG_spearmanr = spearmanr(pred_ddG, ddG)[0]
            dS_pearsonr = pearsonr(pred_dS, dS)[0]
            dS_spearmanr = spearmanr(pred_dS, dS)[0]
            epoch_metrics['ddG_pearsonr'] = ddG_pearsonr
            epoch_metrics['ddG_spearmanr'] = ddG_spearmanr
            epoch_metrics['dS_pearsonr'] = dS_pearsonr
            epoch_metrics['dS_spearmanr'] = dS_spearmanr
            epoch_metrics['avg_pearsonr'] = (abs(ddG_pearsonr) + abs(dS_pearsonr)) / 2
            epoch_metrics['avg_spearmanr'] = (abs(ddG_spearmanr) + abs(dS_spearmanr)) / 2

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

        epoch_metrics = {}
        epoch_metrics['loss'] = torch.tensor([batch['loss'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['ce'] = torch.tensor([batch['ce'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['reg'] = torch.tensor([batch['reg'] for batch in batch_outputs]).mean().detach()
        epoch_metrics['mse'] = torch.tensor([batch['mse'] for batch in batch_outputs]).mean().detach()

        if batch_outputs[0]['pred_ddG'] is not None and batch_outputs[0]['pred_dS'] is not None:
            pred_ddG = torch.cat([batch['pred_ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            pred_dS = torch.cat([batch['pred_dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            ddG = torch.cat([batch['ddG'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            dS = torch.cat([batch['dS'] for batch in batch_outputs], dim=0).detach().cpu().numpy()
            ddG_pearsonr = pearsonr(pred_ddG, ddG)[0]
            ddG_spearmanr = spearmanr(pred_ddG, ddG)[0]
            dS_pearsonr = pearsonr(pred_dS, dS)[0]
            dS_spearmanr = spearmanr(pred_dS, dS)[0]
            epoch_metrics['ddG_pearsonr'] = ddG_pearsonr
            epoch_metrics['ddG_spearmanr'] = ddG_spearmanr
            epoch_metrics['dS_pearsonr'] = dS_pearsonr
            epoch_metrics['dS_spearmanr'] = dS_spearmanr
            epoch_metrics['avg_pearsonr'] = (abs(ddG_pearsonr) + abs(dS_pearsonr)) / 2
            epoch_metrics['avg_spearmanr'] = (abs(ddG_spearmanr) + abs(dS_spearmanr)) / 2

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
        tokens, ddG, dS = batch
        logits, mean, logvar, pred_ddG, pred_dS = self.model(tokens)
        logits = logits[:, 1:-1]
        tokens = tokens[:, 1:-1]
        recon_tokens = logits.argmax(dim=-1)
        z = self.model.reparameterize(mean, logvar)
        return {
            'recon_tokens': recon_tokens,
            'tokens': tokens,
            'ddG': ddG,
            'dS': dS,
            'mean': mean,
            'logvar': logvar,
            'z': z,
            'pred_ddG': pred_ddG,
            'pred_dS': pred_dS,
        }
