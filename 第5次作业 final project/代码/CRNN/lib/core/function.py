from  __future__ import  absolute_import
import time
import lib.utils.utils as utils
import torch
import sys
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    debug_path = os.path.join(config.DEBUG_DIR, "debug_{}_ep{}.txt".format(config.DATASET.DATASET, epoch))
    if os.path.exists(debug_path):
        os.remove(debug_path)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        labels = utils.get_batch_label(dataset, idx)
        inp = inp.to(device)

        # inference
        preds = model(inp).cpu()                                    # preds = Log_probs: Tensor of size (T, N, C)

        # compute loss
        batch_size = inp.size(0)
        text, length = converter.encode(labels)                     # text = Targets: Tensor of size (N, S), N=batch size and S=max target length
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)  # preds_size = Input_lengths: Tuple or tensor of size (N)        
        loss = criterion(preds, text, preds_size, length)           # length = Target_lengths: Tuple or tensor of size (N). It represent lengths of the targets.
        
        # for debug
        if "{}".format(loss.item()) == 'nan' or "{}".format(loss.item()) == "inf":
            info = "ep{} {} {}\t".format(epoch, loss.item(), labels)
            for tid in idx: info += list(dataset.labels[tid.item()].keys())[0] + " "
            info += "\n"
            # print(info, end="")
            with open(debug_path, "a+") as debug_f:
                debug_f.write(info)
            
            bug_flag = True
            # print(preds.shape, text.shape, preds_size.shape, length.shape)
            # print(preds)
            # print(text)
            # print(preds_size)
            # print(length)
            # sys.exit(0)
        else:
            bug_flag = False
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg + " [Bug : {}]".format(bug_flag)) # 原本是end='\r'，这样不方便保存log

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
    
        end = time.time()
    print()


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict, mode="test"):

    losses = AverageMeter()
    model.eval()

    n_correct, n_trials = 0, 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):
            
            n_trials += idx.shape[0]

            labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)

            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1

            if i % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            if i == config.TEST.NUM_TEST:
                break
    print()

    # raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    # for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
    #     print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(n_trials)
    print('Test loss: {:.4f}, n_correct: {}, total_trials: {}, accuray: {:.4f}'.format(losses.avg, n_correct, n_trials, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy