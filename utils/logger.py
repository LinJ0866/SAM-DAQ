r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch

from utils.tools import to_cpu

class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self):
        self.loss_buf = []

    def update(self, preds, masks, loss):
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute(self):
        loss_buf = torch.stack(self.loss_buf)

        return loss_buf.mean()
    
    def write_result(self, split, epoch):
        avg_loss = self.compute()
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % avg_loss

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            avg_loss = self.compute()
            if epoch != -1:
                msg += 'L: %6.5f  ' % self.loss_buf[-1]
                msg += 'Avg L: %6.5f' % avg_loss
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training \
            else os.path.join(args.logpath, 'test/fold_' + args.load.split('/')[-2].split('.')[0] + logtime)
        if logpath == '': logpath = logtime

        cls.logpath = logpath
        if not os.path.exists(cls.logpath): os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:==================== Start =====================')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, metric_name, value):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. %s: %5.2f.\n' % (epoch, metric_name, value))
