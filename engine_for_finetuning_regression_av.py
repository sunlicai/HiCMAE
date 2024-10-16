import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax

from regression_metrics import cal_regression_metrics

def train_class_batch(model, samples, samples_audio, target, criterion):
    outputs = model(samples, samples_audio)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, ((samples, samples_audio), targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        samples_audio = samples_audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            samples_audio = samples_audio.half()
            loss, output = train_class_batch(
                model, samples, samples_audio, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, samples_audio, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            # for regression, 1 - L1
            class_acc = (1 - torch.abs(output.cpu().detach() - targets.cpu().detach())).mean().clip(min=0)
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, args):
    if args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'smooth_l1':
        criterion = torch.nn.SmoothL1Loss()
    elif args.loss == 'ccc':
        from regression_metrics import CCCLoss
        criterion = CCCLoss(label_dim=args.nb_classes)
    elif args.loss == 'pcc':
        from regression_metrics import PCCLoss
        criterion = PCCLoss(label_dim=args.nb_classes)
    else:
        raise NotImplementedError
    # criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    outputs, targets = [], []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos, audios = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        audios = audios.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos, audios)
            loss = criterion(output, target)

        output, target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        if args.keep_temporal_dim: # merge temporal dim into batch dim
            output = output.reshape(-1, args.nb_classes) # (B, T*C) -> (B*T, C)
            target = target.reshape(-1, args.nb_classes) # (B, T*C) -> (B*T, C)
        outputs.append(output)
        targets.append(target)
        metrics = cal_regression_metrics(output, target)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(metrics['acc'], n=batch_size)
        metric_logger.meters['mae'].update(metrics['mae'], n=batch_size)
        metric_logger.meters['mse'].update(metrics['mse'], n=batch_size)
        metric_logger.meters['rmse'].update(metrics['rmse'], n=batch_size)
        metric_logger.meters['pcc'].update(metrics['pcc'], n=batch_size)
        metric_logger.meters['ccc'].update(metrics['ccc'], n=batch_size)
    # cal total metrics across the val set
    metrics = cal_regression_metrics(np.concatenate(outputs), np.concatenate(targets))
    metric_logger.meters['acc_total'].update(metrics['acc'], n=1)
    metric_logger.meters['mae_total'].update(metrics['mae'], n=1)
    metric_logger.meters['mse_total'].update(metrics['mse'], n=1)
    metric_logger.meters['rmse_total'].update(metrics['rmse'], n=1)
    metric_logger.meters['pcc_total'].update(metrics['pcc'], n=1)
    metric_logger.meters['ccc_total'].update(metrics['ccc'], n=1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* ACC: {acc.global_avg:.4f} MAE: {mae.global_avg:.4f} MSE: {mse.global_avg:.4f} RMSE: {rmse.global_avg:.4f} PCC: {pcc.global_avg:.4f} CCC: {ccc.global_avg:.4f}  loss: {losses.global_avg:.4f}'
          .format(acc=metric_logger.acc, mae=metric_logger.mae, mse=metric_logger.mse, rmse=metric_logger.rmse, pcc=metric_logger.pcc, ccc=metric_logger.ccc, losses=metric_logger.loss))

    print('* Total ACC: {acc.global_avg:.4f} Total MAE: {mae.global_avg:.4f} Total MSE: {mse.global_avg:.4f} Total RMSE: {rmse.global_avg:.4f} Total PCC: {pcc.global_avg:.4f} Total CCC: {ccc.global_avg:.4f}'
          .format(acc=metric_logger.acc_total, mae=metric_logger.mae_total, mse=metric_logger.mse_total, rmse=metric_logger.rmse_total, pcc=metric_logger.pcc_total, ccc=metric_logger.ccc_total))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, args):
    if args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'smooth_l1':
        criterion = torch.nn.SmoothL1Loss()
    elif args.loss == 'ccc':
        from regression_metrics import CCCLoss
        criterion = CCCLoss(label_dim=args.nb_classes)
    elif args.loss == 'pcc':
        from regression_metrics import PCCLoss
        criterion = PCCLoss(label_dim=args.nb_classes)
    else:
        raise NotImplementedError

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    outputs, targets = [], []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos, audios = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        audios = audios.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos, audios)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(target.data[i].cpu().numpy().tolist()), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        output, target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        if args.keep_temporal_dim: # merge temporal dim into batch dim
            output = output.reshape(-1, args.nb_classes) # (B, T*C) -> (B*T, C)
            target = target.reshape(-1, args.nb_classes) # (B, T*C) -> (B*T, C)
        outputs.append(output)
        targets.append(target)
        metrics = cal_regression_metrics(output, target)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(metrics['acc'], n=batch_size)
        metric_logger.meters['mae'].update(metrics['mae'], n=batch_size)
        metric_logger.meters['mse'].update(metrics['mse'], n=batch_size)
        metric_logger.meters['rmse'].update(metrics['rmse'], n=batch_size)
        metric_logger.meters['pcc'].update(metrics['pcc'], n=batch_size)
        metric_logger.meters['ccc'].update(metrics['ccc'], n=batch_size)
    # cal total metrics across the val set
    metrics = cal_regression_metrics(np.concatenate(outputs), np.concatenate(targets))
    metric_logger.meters['acc_total'].update(metrics['acc'], n=1)
    metric_logger.meters['mae_total'].update(metrics['mae'], n=1)
    metric_logger.meters['mse_total'].update(metrics['mse'], n=1)
    metric_logger.meters['rmse_total'].update(metrics['rmse'], n=1)
    metric_logger.meters['pcc_total'].update(metrics['pcc'], n=1)
    metric_logger.meters['ccc_total'].update(metrics['ccc'], n=1)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(metrics['acc'], metrics['mse'])) # me: no use
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* ACC: {acc.global_avg:.4f} MAE: {mae.global_avg:.4f} MSE: {mse.global_avg:.4f} RMSE: {rmse.global_avg:.4f} PCC: {pcc.global_avg:.4f} CCC: {ccc.global_avg:.4f}  loss: {losses.global_avg:.4f}'
          .format(acc=metric_logger.acc, mae=metric_logger.mae, mse=metric_logger.mse, rmse=metric_logger.rmse, pcc=metric_logger.pcc, ccc=metric_logger.ccc, losses=metric_logger.loss))

    print('* Total ACC: {acc.global_avg:.4f} Total MAE: {mae.global_avg:.4f} Total MSE: {mse.global_avg:.4f} Total RMSE: {rmse.global_avg:.4f} Total PCC: {pcc.global_avg:.4f} Total CCC: {ccc.global_avg:.4f}'
          .format(acc=metric_logger.acc_total, mae=metric_logger.mae_total, mse=metric_logger.mse_total, rmse=metric_logger.rmse_total, pcc=metric_logger.pcc_total, ccc=metric_logger.ccc_total))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, args, best=False):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt') if not best else os.path.join(eval_path, str(x) + '_best.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = np.fromstring(line.split(']')[1].split('[')[1], dtype=np.float, sep=',')
            chunk_nb = line.split(']')[2].split(' ')[1]
            split_nb = line.split(']')[2].split(' ')[2]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    # more metrics and save preds
    pred_dict = {'id': [], 'label': [], 'pred': []}
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
        pred = np.mean(dict_feats[item], axis=0)
        label = dict_label[item]
        pred_dict['pred'].append(pred)
        pred_dict['label'].append(label)
        pred_dict['id'].append(item.strip())

    # calculate overall metrics on the test set
    total_preds, total_labels = np.stack(pred_dict['pred']), np.stack(pred_dict['label'])
    metrics_dict = cal_regression_metrics(total_preds, total_labels, return_finegrained=True)

    return metrics_dict, pred_dict

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
