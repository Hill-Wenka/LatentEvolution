import numpy as np
import pandas as pd
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from torchmetrics import Metric
from torchmetrics import functional

key_list = ['ACC', 'AUC', 'MCC', 'Q-value', 'F1', 'F0.5', 'F2', 'SE', 'SP', 'PPV', 'NPV', 'TN', 'FP', 'FN', 'TP']


class BinaryClassificationMetrics(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False  # important! see documentation

    def __init__(self, prefix='', target_metric='F1', mode='max', search_threshold=False, t_start=0.3, t_end=0.8, t_step=0.01):
        super().__init__()
        self.prefix = prefix
        self.target_metric = target_metric
        self.mode = mode
        self.search_threshold = search_threshold
        self.best_t = 0.5
        self.t_start = t_start
        self.t_end = t_end
        self.t_step = t_step
        self.key_list = ['ACC', 'AUC', 'MCC', 'Q-value', 'F1', 'F0.5', 'F2', 'SE', 'SP', 'PPV', 'NPV', 'TN', 'FP', 'FN', 'TP']
        self.key_list = [prefix + key for key in self.key_list]
        self.add_state('probs', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        # logtis: (N, 2), labels: (N,)
        if logits.shape[1] == 2:
            probs = logits.softmax(dim=1)[:, 1]
        elif logits.shape[1] == 1:
            probs = logits[:, 0].sigmoid()
        else:
            raise RuntimeError(f'logits.shape[1] should be 1 or 2, but got {logits.shape[1]}')
        assert probs.shape == labels.shape

        self.probs = self.probs + probs.tolist()
        self.labels = self.labels + labels.long().tolist()
        # 踩坑！千万不能使用.cpu().numpy().to_list()，否则在pl_model.cpu()的过程中会报错！

    def compute(self):
        probs = torch.tensor(self.probs)
        labels = torch.tensor(self.labels)

        # search the best threshold
        best_t = self.best_t
        if self.search_threshold:
            best_metric, self.best_t = find_best_threshold(probs, labels, self.target_metric, self.mode,
                                                           self.t_start, self.t_end, self.t_step, softmax=False)
            # print('best_metric, self.best_t', best_metric, self.best_t)
            self.best_t = 0.5 if self.best_t is None else self.best_t

        # compute the metrics with the selected threshold
        ACC = functional.accuracy(probs, labels, task='binary', threshold=best_t)
        SE = functional.recall(probs, labels, task='binary', threshold=best_t)
        SP = functional.specificity(probs, labels, task='binary', threshold=best_t)
        AUC = functional.auroc(probs, labels, task='binary')
        MCC = functional.matthews_corrcoef(probs, labels, num_classes=2, task='binary', threshold=best_t)
        F1 = functional.f1_score(probs, labels, task='binary', threshold=best_t)
        F05 = functional.fbeta_score(probs, labels, beta=0.5, task='binary', threshold=best_t)
        F2 = functional.fbeta_score(probs, labels, beta=2.0, task='binary', threshold=best_t)
        Q = (SE + SP) / 2
        PPV = functional.precision(probs, labels, task='binary', threshold=best_t)
        ConfusionMatrix = functional.confusion_matrix(probs, labels, task='binary', num_classes=2, threshold=best_t)
        TN = ConfusionMatrix[0][0]
        FP = ConfusionMatrix[0][1]
        FN = ConfusionMatrix[1][0]
        TP = ConfusionMatrix[1][1]
        NPV = TN / (TN + FN)
        metric_list = [ACC, AUC, MCC, Q, F1, F05, F2, SE, SP, PPV, NPV, TN, FP, FN, TP]
        metric_list = [x.float() for x in metric_list]
        metric_dict = dict(zip(self.key_list, metric_list))
        return metric_dict


def find_best_threshold(logits, labels, metric, mode='max', start=0.3, end=0.8, step=0.01, softmax=True):
    best_metric = 0
    best_t = None
    for t in np.arange(start, end + 0.0001, step):
        target_metric = get_functional_torch_metrics(logits, labels, threshold=t, softmax=softmax)[metric]
        if mode == 'max' and target_metric > best_metric:
            best_metric, best_t = target_metric, t
        if mode == 'min' and target_metric < best_metric:
            best_metric, best_t = target_metric, t
    return best_metric, best_t


def get_sklearn_metrics(pred, target, threshold=0.5, softmax=True, only_metric_keys=False):
    '''
    基于torchmetrics.functional实现的二分类指标
    :param pred: 预测结果 (logtis/pred_probs) [B, 2]
    :param target: 真实标签 (label) [B]
    :param threshold: 分类阈值
    :param softmax: 是否需要对pred参数进行softmax归一化
    :param only_metric_keys: 返回的dict是否只需要包含基本指标的键值对 (即不返回list和dataframe)
    :return: metric_dict: dict
    '''
    if not type(pred) == torch.Tensor:
        pred = torch.tensor(pred, dtype=torch.float)
    if not type(target) == torch.Tensor:
        target = torch.tensor(target)

    if len(pred.shape) == 1:
        pred = pred.reshape([-1, 1])
    pred = pred.softmax(dim=-1)[:, -1] if softmax else pred[:, -1]
    target = list(map(int, target))  # 将标签转为int类型
    pred_label = np.array((pred >= threshold), dtype=np.int)

    ConfusionMatrix = confusion_matrix(target, pred_label, task='binary', num_classes=2)
    TN, FP, FN, TP = ConfusionMatrix[0][0], ConfusionMatrix[0][1], ConfusionMatrix[1][0], ConfusionMatrix[1][1]
    ACC = accuracy_score(target, pred_label)
    AUC = roc_auc_score(target, pred)
    MCC = matthews_corrcoef(target, pred_label)
    F1 = f1_score(target, pred_label)
    F05 = fbeta_score(target, pred_label, beta=0.5)
    F2 = fbeta_score(target, pred_label, beta=2)
    SE = Recall = recall_score(target, pred_label)
    SP = TN / (TN + FP)
    Q = (SE + SP) / 2  # BACC
    PPV = Precision = precision_score(target, pred_label)
    NPV = TN / (TN + FN)

    metric_list = [ACC, AUC, MCC, Q, F1, F05, F2, SE, SP, PPV, NPV, TN, FP, FN, TP]
    metric_dict = dict(zip(key_list, metric_list))
    if not only_metric_keys:
        metric_df = pd.DataFrame(metric_dict, index=[0])
        metric_dict['metric_keys'] = key_list
        metric_dict['metric_values'] = metric_list
        metric_dict['metric_df'] = metric_df
    return metric_dict


def get_module_torch_metrics(pred, target, threshold=0.5, softmax=True, only_metric_keys=False):
    '''
    基于torchmetrics.functional实现的二分类指标
    :param pred: 预测结果 (logtis/pred_probs) [B, 2]
    :param target: 实际标签 (label) [B]
    :param threshold: 分类阈值
    :param softmax: 是否需要对pred参数进行softmax归一化
    :param only_metric_keys: 返回的dict是否只需要包含基本指标的键值对就可以了 (即不返回list和dataframe)
    :return: 指标字典 (dict)
    '''
    if not type(pred) == torch.Tensor:
        pred = torch.tensor(pred, dtype=torch.float)
    if not type(target) == torch.Tensor:
        target = torch.tensor(target)

    if len(pred.shape) == 1:
        pred = pred.reshape([-1, 1])
    pred = pred.softmax(dim=-1)[:, -1] if softmax else pred[:, -1]
    pred_label = (pred >= threshold).long()

    metric_ACC = torchmetrics.Accuracy(task='binary')
    metric_SE = torchmetrics.Recall(task='binary')
    metric_SP = torchmetrics.Specificity(task='binary')
    metric_AUC = torchmetrics.AUROC(num_classes=2, task='binary')
    metric_MCC = torchmetrics.MatthewsCorrCoef(num_classes=2, task='binary')
    metric_F1 = torchmetrics.F1Score(task='binary')
    metric_F05 = torchmetrics.FBetaScore(beta=0.5, task='binary')
    metric_F2 = torchmetrics.FBetaScore(beta=2.0, task='binary')
    metric_Precision = torchmetrics.Precision(task='binary')
    metric_ConfusionMatrix = torchmetrics.ConfusionMatrix(num_classes=2)

    ACC = metric_ACC(pred_label, target)
    SE = metric_SE(pred_label, target)
    SP = metric_SP(pred_label, target)
    AUC = metric_AUC(pred, target)
    MCC = metric_MCC(pred_label, target)
    F1 = metric_F1(pred_label, target)
    F05 = metric_F05(pred_label, target)
    F2 = metric_F2(pred_label, target)
    Q = (SE + SP) / 2
    PPV = metric_Precision(pred_label, target)
    ConfusionMatrix = metric_ConfusionMatrix(pred_label, target)
    TN = ConfusionMatrix[0][0]
    FP = ConfusionMatrix[0][1]
    FN = ConfusionMatrix[1][0]
    TP = ConfusionMatrix[1][1]
    NPV = TN / (TN + FN)

    metric_list = [ACC, AUC, MCC, Q, F1, F05, F2, SE, SP, PPV, NPV, TN, FP, FN, TP]
    metric_list = [value.item() for value in metric_list]
    metric_dict = dict(zip(key_list, metric_list))
    if not only_metric_keys:
        metric_df = pd.DataFrame(metric_dict, index=[0])
        metric_dict['metric_keys'] = key_list
        metric_dict['metric_values'] = metric_list
        metric_dict['metric_df'] = metric_df
    return metric_dict


def get_functional_torch_metrics(pred, target, threshold=0.5, softmax=True, only_metric_keys=False):
    '''
    基于torchmetrics.functional实现的二分类指标
    :param pred: 预测结果 (logtis/pred_probs) [B, 2]
    :param target: 实际标签 (label) [B]
    :param threshold: 分类阈值
    :param softmax: 是否需要对pred参数进行softmax归一化
    :param only_metric_keys: 返回的dict是否只需要包含基本指标的键值对就可以了 (即不返回list和dataframe)
    :return: 指标字典 (dict)
    '''
    if not type(pred) == torch.Tensor:
        pred = torch.tensor(pred, dtype=torch.float)
    if not type(target) == torch.Tensor:
        target = torch.tensor(target)

    if len(pred.shape) == 1:
        pred = pred.reshape([-1, 1])
    pred = pred.softmax(dim=-1)[:, -1] if softmax else pred[:, -1]
    pred_label = (pred >= threshold).long()

    ACC = functional.accuracy(pred_label, target, task='binary')
    SE = functional.recall(pred_label, target, task='binary')
    SP = functional.specificity(pred_label, target, task='binary')
    AUC = functional.auroc(pred, target, task='binary')
    MCC = functional.matthews_corrcoef(pred_label, target, num_classes=2, task='binary')
    F1 = functional.f1_score(pred_label, target, task='binary')
    F05 = functional.fbeta_score(pred_label, target, beta=0.5, task='binary')
    F2 = functional.fbeta_score(pred_label, target, beta=2.0, task='binary')
    Q = (SE + SP) / 2
    PPV = functional.precision(pred_label, target, task='binary')
    ConfusionMatrix = functional.confusion_matrix(pred_label, target, task='binary', num_classes=2)
    TN = ConfusionMatrix[0][0]
    FP = ConfusionMatrix[0][1]
    FN = ConfusionMatrix[1][0]
    TP = ConfusionMatrix[1][1]
    NPV = TN / (TN + FN)

    metric_list = [ACC, AUC, MCC, Q, F1, F05, F2, SE, SP, PPV, NPV, TN, FP, FN, TP]
    metric_list = [float(value.item()) for value in metric_list]
    metric_dict = dict(zip(key_list, metric_list))
    if not only_metric_keys:
        metric_df = pd.DataFrame(metric_dict, index=[0])
        metric_dict['metric_keys'] = key_list
        metric_dict['metric_values'] = metric_list
        metric_dict['metric_df'] = metric_df
    return metric_dict
