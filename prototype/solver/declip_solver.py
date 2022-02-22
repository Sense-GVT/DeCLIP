import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import datetime
import torch
import json
import math
import linklink as link
import torch.nn.functional as F

from .base_solver import BaseSolver
from prototype.utils.dist import link_dist, DistModule, broadcast_object
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    param_group_all, AverageMeter, accuracy, load_state_model, load_state_optimizer, mixup_data, \
    mix_criterion, modify_state, cutmix_data, parse_config
from prototype.utils.ema import EMA
from prototype.model import model_entry
from prototype.optimizer import optim_entry, FP16RMSprop, FP16SGD, FusedFP16SGD, FP16AdamW
from prototype.lr_scheduler import scheduler_entry
from prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.data import build_clip_dataloader
from prototype.loss_functions import LabelSmoothCELoss, ClipInfoCELoss, SimsiamLoss, NTXentLoss
# from prototype.utils.user_analysis_helper import send_info
from prototype.utils.grad_clip import clip_grad_norm_, clip_grad_value_, clip_param_grad_value_


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        # with torch.cuda.stream(self.stream):
        #     for k in self.batch:
        #         if k != 'meta':
        #             self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


class EMA_logit_scale():
    def __init__(self, param, threshold):
        self.param = param
        self.buffer = 3.125
        self.momentum = 0.9
        self.threshold = threshold
        self.clip_number = 0

    def update(self):
        self.buffer = self.momentum*self.buffer + \
            (1-self.momentum)*self.param.data

    def clamp(self):
        if (self.param-self.buffer) > self.threshold:
            self.param.data = torch.as_tensor(
                self.buffer+self.threshold, dtype=self.param.dtype, device=self.param.device)
            self.clip_number += 1
        elif (self.buffer-self.param) > self.threshold:
            self.param.data = torch.as_tensor(
                self.buffer-self.threshold, dtype=self.param.dtype, device=self.param.device)
            self.clip_number += 1
        # self.param.data = torch.as_tensor(
        #     3.125, dtype=self.param.dtype, device=self.param.device)


class ClsSolver(BaseSolver):

    def __init__(self, config_file):
        self.config_file = config_file
        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)
        self.setup_env()
        # import ipdb
        # ipdb.set_trace()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()
        # send_info(self.prototype_info)

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.critical(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.critical(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if hasattr(self.config.saver, 'pretrain'):
            if self.config.saver.pretrain.auto_resume:
                last_checkpoint = self.find_last_checkpoint()
                if last_checkpoint:
                    self.config.saver.pretrain.path = last_checkpoint
            if self.config.saver.pretrain.get("path", None):
                self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                self.logger.info(
                    f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            else:
                self.state = {}
                self.state['last_iter'] = 0
            #pretrain from moco
            if self.config.saver.pretrain.get('pretrain_from', None)  == 'moco':
                encoder_state = {}
                for key, value in self.state['model'].items():
                    if 'encoder_q' in key and 'fc' not in key and 'attnpool' not in key:
                        new_key = key.replace('encoder_q', 'visual')
                        encoder_state[new_key] = value
                self.state = {'model': encoder_state, 'last_iter': 0, 'optimizer': None}
            #pretrain from supervised
            if self.config.saver.pretrain.get('pretrain_from', None)  == 'supervised':
                encoder_state = {}
                for key, value in self.state['model'].items():
                    if 'fc' not in key:
                        new_key = key.replace('module', 'module.visual')
                        encoder_state[new_key] = value
                self.state = {'model': encoder_state, 'last_iter': 0, 'optimizer': None}

            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(
                    self.state, self.config.saver.pretrain.ignore)

        else:
            self.state = {}
            self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def find_last_checkpoint(self):
        ckpt_list = os.listdir('checkpoints')
        if 'ckpt.pth.tar' in ckpt_list:
            return 'checkpoints/ckpt.pth.tar'
        elif len(ckpt_list) == 0:
            return None
        num = [int(ckpt.split('.')[0][5:]) for ckpt in ckpt_list]
        num.sort()
        last_checkpoint_path = 'checkpoints/ckpt_' + str(num[-1])+'.pth.tar'
        return last_checkpoint_path

    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.critical('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)
        # count_flops(self.model, input_shape=[
        #             1, 3, self.config.data.input_size, self.config.data.input_size])

        # handle fp16
        if self.config.optimizer.type == 'FP16SGD' or \
           self.config.optimizer.type == 'FP16RMSprop' or \
           self.config.optimizer.type == 'FP16AdamW'or \
           self.config.optimizer.type == 'FP16AdamW_SGD':
            self.fp16 = True
            self.fused_fp16 = False
        elif self.config.optimizer.type == 'FusedFP16SGD' or \
             self.config.optimizer.type == 'FusedFP16AdamW':
            self.fp16 = True
            self.fused_fp16 = True

        else:
            self.fp16 = False
            self.fused_fp16 = False

        if self.fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            self.logger.critical('using fp16 when training')
            if self.config.optimizer.get('fp16_normal_bn', False):
                self.logger.critical('using normal bn for fp16')
                link.fp16.register_float_module(
                    link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(
                    torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get('fp16_normal_fc', False):
                self.logger.critical('using normal fc for fp16')
                link.fp16.register_float_module(
                    torch.nn.Linear, cast_args=True)
            if self.config.optimizer.get('fp16_normal_ln', False):
                self.logger.critical('using normal ln for fp16')
                link.fp16.register_float_module(
                    torch.nn.LayerNorm, cast_args=False)
            link.fp16.init()
            # Note: The module is converted to fp16!
            self.model.half()
        self.model = DistModule(self.model, self.config.dist.sync)

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def build_optimizer(self):

        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        #if opt_config['type'] == 'AdamW_SGD':
        if opt_config['type'] in ['FP16AdamW_SGD', 'AdamW_SGD']:
            text_config = opt_config['text_config']
            visual_config = opt_config['visual_config']
            text_parameters = self.model.module.text_parameters()
            visual_parameters = self.model.module.visual_parameters()
            param_group = []
            if len(text_parameters) > 0:
                param_group.append({'params': text_parameters, **text_config})
            if len(visual_parameters) > 0:
                param_group.append(
                    {'params': visual_parameters, **visual_config})
            #for text_module in self.model.module.text_modules():
            self.logger.critical('[Info] param group: Text-module')
            for idx, text_module in enumerate(self.model.module.text_modules()):
                self.logger.info(f'[Info] text param group {idx}')
                param_group_text, type2num = param_group_all(
                    text_module, pconfig, text_config)
                param_group += param_group_text
            #for visual_module in self.model.module.visual_modules():
            self.logger.critical('[Info] param group: Visual-module')
            for idx, visual_module in enumerate(self.model.module.visual_modules()):
                self.logger.info(f'[Info] visual param group {idx}')
                param_group_visual, type2num = param_group_all(
                    visual_module, pconfig, visual_config)
                param_group += param_group_visual
        else:
            param_group, type2num = param_group_all(self.model, pconfig)

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)

        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

        # EMA
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
        else:
            self.ema = None

        if 'ema' in self.state and self.config.ema.enable:
            self.ema.load_state_dict(self.state['ema'])

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer.optimizer if isinstance(self.optimizer, FP16SGD) or \
            isinstance(self.optimizer, FP16RMSprop) or isinstance(self.optimizer, FP16AdamW) else self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        test_config = {}
        self.config.data.last_iter = self.state['last_iter']
        test_config['last_iter'] = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
            test_config['max_iter'] = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch
            test_config['max_epoch'] = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        elif self.config.data.get('type') == 'clip':
            self.train_data = build_clip_dataloader('train', self.config.data)

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        elif self.config.data.get('type') == 'clip':
            self.val_data = []
            for config in self.config.data.test:
                config.update(test_config)
                self.val_data.append(build_clip_dataloader('test', config))

        self.prefetch = self.config.data.train.get('prefetch', False)
        if self.prefetch:
            self.prefetcher = DataPrefetcher(self.train_data['loader'])
        elif self.train_data['loader']:
            self.train_data['loader'] = iter(self.train_data['loader'])

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.clip_losses = AverageMeter(self.config.saver.print_freq)                         
        self.meters.simsiam_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.text_simsiam_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.clip_nn_text_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.text_mlm_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.nt_xent_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        label_smooth = self.config.get('label_smooth', 0.0)
        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        if label_smooth > 0:
            self.logger.critical('using label_smooth: {}'.format(label_smooth))
            self.criterion = LabelSmoothCELoss(label_smooth, self.num_classes)
        else:
            # self.criterion = ClipInfoCELoss(self.config.loss.partition_num)
            self.criterion = ClipInfoCELoss()
        self.simsiam_criterion = SimsiamLoss()
        self.nt_xent_criterion = NTXentLoss(self.config.data.batch_size)

        self.mixup = self.config.get('mixup', 1.0)
        self.cutmix = self.config.get('cutmix', 0.0)
        if self.mixup < 1.0:
            self.logger.critical(
                'using mixup with alpha of: {}'.format(self.mixup))
        if self.cutmix > 0.0:
            self.logger.critical(
                'using cutmix with alpha of: {}'.format(self.cutmix))

    def train(self):

        self.pre_train()
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()
        last_logit_scale = 0
        logit_scale = EMA_logit_scale(self.model.module.logit_scale,
                          self.config.grad_clip.value)
        #test 
        train_loss = 1000.0
        #test_top1_prec = 0.0
        #for id, val_data in enumerate(self.val_data):
        #    metrics = self.evaluate(val_data)
        #    test_top1_prec = metrics.metric['top1']


        # import ipdb;ipdb.set_trace()
        for i in range(len(self.train_data['loader'])):
            if self.prefetch:
                batch = self.prefetcher.next()
            else:
                batch = next(self.train_data['loader'])
        # for i, batch in enumerate(self.train_data['loader']):
            # images = batch['images']
            # texts = [caption[0] fro caption in batch['captions']] # use first caption of each image
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            # target = target.squeeze().cuda().long()
            if self.fp16:
                if type(batch['images']) == list:
                    batch['images'] = [image.cuda().half() for image in batch['images']]
                else:
                    batch['images'] = batch['images'].cuda().half()
                # batch['captions'] = batch['captions'].cuda().half()
            else:
                if type(batch['images']) == list:
                    batch['images'] = [image.cuda() for image in batch['images']]
                else:
                    batch['images'] = batch['images'].cuda()
                # batch.cuda()
            # mixup
            # if self.mixup < 1.0:
            #     batch, target_a, target_b, lam = mixup_data(
            #         batch, target, self.mixup)
            # cutmix
            # if self.cutmix > 0.0:
            #     input, target_a, target_b, lam = cutmix_data(
            #         input, target, self.cutmix)
            # forward
            if 'CLSA' not in self.config.data.train.transforms.type:
                output_dict = self.model(batch, return_dict=True)
                logits_per_image, logits_per_image_2, logits_per_text, logits_per_text_2 = output_dict['logits']
                logits_per_image_1_aug, logits_per_image_2_aug, logits_per_text_1_aug, logits_per_text_2_aug = output_dict['logits_aug']
                p1, p2, z1, z2 = output_dict['simsiam_features']
                text_features, image_features_1, image_features_2 = output_dict['features']
                # loss
                clip_loss_1, target = self.criterion(logits_per_image, logits_per_text)
                clip_loss_2, target_2 = self.criterion(logits_per_image_2, logits_per_text_2)
                clip_loss_1_aug, target_1_aug = self.criterion(logits_per_image_1_aug, logits_per_text_1_aug)
                clip_loss_2_aug, target_2_aug = self.criterion(logits_per_image_2_aug, logits_per_text_2_aug)

                if self.config.data.train.get('only_image_two_view', False):
                    clip_loss = (clip_loss_1 + clip_loss_2) / 2
                elif self.config.data.train.get('image_text_two_view', False):
                    clip_loss = (clip_loss_1 + clip_loss_2 + clip_loss_1_aug + clip_loss_2_aug) / 4
                else:
                    raise NotImplementedError()
                clip_loss /= self.dist.world_size

                if 'text_self_supervised' in output_dict.keys():
                    text_mlm_loss = output_dict['text_self_supervised'] / self.dist.world_size
                else:
                    text_mlm_loss = torch.zeros_like(clip_loss)

                if 'nn_text_simsiam' in output_dict.keys():
                    p_text, z_text_nn = output_dict['nn_text_simsiam']
                    z_text_nn = z_text_nn[0]
                    simsiam_nn_text_loss = self.simsiam_criterion(p_text,z_text_nn,p_text,z_text_nn, return_KPI=True)
                    simsiam_nn_text_loss = simsiam_nn_text_loss / self.dist.world_size  # z1d; z2d constructed
                else:
                    simsiam_nn_text_loss = torch.zeros_like(clip_loss)

                if 'text_simsiam' in output_dict.keys():
                    p1t, p2t, z1t, z2t = output_dict['text_simsiam']
                    text_simsiam_loss = self.simsiam_criterion(p1t,z1t,p2t,z2t) / self.dist.world_size
                else:
                    text_simsiam_loss = torch.zeros_like(clip_loss)

                if 'nn_text_logits' in output_dict.keys():
                    logits_per_image_1_nn, logits_per_image_2_nn, logits_per_image_1_nn_aug, logits_per_image_2_nn_aug = output_dict['nn_text_logits']
                    clip_loss_i1_nn, _ = self.criterion(logits_per_image_1_nn, logits_per_image_1_nn_aug)
                    clip_loss_i2_nn, _ = self.criterion(logits_per_image_2_nn, logits_per_image_2_nn_aug)
                    clip_nn_text_loss = (clip_loss_i1_nn+clip_loss_i2_nn)/2
                    clip_nn_text_loss = clip_nn_text_loss / self.dist.world_size
                    # print(clip_loss_i_nn, '<< clip loss i nn', flush=True)
                else:
                    clip_nn_text_loss = torch.zeros_like(clip_loss)

                simsiam_loss = self.simsiam_criterion(p1,z1,p2,z2) / self.dist.world_size

                nt_xent_loss_1 = self.nt_xent_criterion(image_features_1, text_features)
                nt_xent_loss_2 = self.nt_xent_criterion(image_features_2, text_features)
                nt_xent_loss = (nt_xent_loss_1 + nt_xent_loss_2) / self.dist.world_size
                # print(clip_loss, simsiam_loss, '<< losses', nt_xent_loss, flush=True)
            else:
                raise NotImplementedError('Solver Codes Not Implemented! please check it')
                logits_per_image, logits_per_image_2, logits_per_image_3, logits_per_image_4, logits_per_text, logits_per_text_2, logits_per_text_3, logits_per_text_4, \
                z1, z2, z3, z4, p1, p2, p3, p4, text_features, image_features_1 = self.model(batch)
                clip_loss_1, target = self.criterion(logits_per_image, logits_per_text)
                clip_loss_2, target_2 = self.criterion(logits_per_image_2, logits_per_text_2) 
                clip_loss_3, target_3 = self.criterion(logits_per_image_3, logits_per_text_3)
                #clip_loss_4, target_4 = self.criterion(logits_per_image_4, logits_per_text_4)
                clip_loss = (clip_loss_1 + clip_loss_2 + clip_loss_3) / 3
                #clip_loss = (clip_loss_1 + clip_loss_2) / 2
                clip_loss /= self.dist.world_size
                #simsiam_loss = (self.simsiam_criterion(p1,z1,p2,z2) + self.simsiam_criterion(p1,z1,p3,z3) + self.simsiam_criterion(p1,z1,p4,z4) + self.simsiam_criterion(p2,z2,p3,z3) + self.simsiam_criterion(p2,z2,p4,z4)) / 5
                simsiam_loss = (self.simsiam_criterion(p1,z1,p2,z2) + self.simsiam_criterion(p1,z1,p3,z3) + self.simsiam_criterion(p1,z1,p4,z4)) / 3
                simsiam_loss /= self.dist.world_size

                nt_xent_loss = self.nt_xent_criterion(image_features_1, text_features) / self.dist.world_size

            if not self.config.clip_simsiam_loss_weight.get('type', None):
                loss = clip_loss * self.config.clip_simsiam_loss_weight.clip_loss
                if self.config.clip_simsiam_loss_weight.get('simsiam_loss', 0):
                    loss = loss + simsiam_loss * self.config.clip_simsiam_loss_weight.simsiam_loss
                if self.config.clip_simsiam_loss_weight.get('masking_language', 0):
                    loss = loss + text_mlm_loss * self.config.clip_simsiam_loss_weight.masking_language
                    # print('New Loss, text_mlm_loss', text_mlm_loss, flush=True)
                if self.config.clip_simsiam_loss_weight.get('text_simsiam_loss', 0):
                    loss = loss + text_simsiam_loss * self.config.clip_simsiam_loss_weight.text_simsiam_loss
                if self.config.clip_simsiam_loss_weight.get('nn_text', 0):
                    loss = loss + clip_nn_text_loss * self.config.clip_simsiam_loss_weight.nn_text
                    # print(clip_nn_text_loss, '<< nn text loss', flush=True)
            elif self.config.clip_simsiam_loss_weight.get('type', None) == 'convirt':
                loss = (clip_loss + nt_xent_loss) / 2 * self.config.clip_simsiam_loss_weight.clip_loss + simsiam_loss * self.config.clip_simsiam_loss_weight.simsiam_loss
            elif self.config.clip_simsiam_loss_weight.get('type', None) == 'linear':
                base_clip_loss_weight = 0.2
                update_clip_loss_weight = 0.8 * curr_step / total_step
                clip_loss_weight = base_clip_loss_weight + update_clip_loss_weight
                simsiam_loss_weight = 1.0 - clip_loss_weight
                loss = clip_loss  * clip_loss_weight + simsiam_loss * simsiam_loss_weight

            elif self.config.clip_simsiam_loss_weight.get('type', None) == 'shift':
                if curr_step%2==0:
                    clip_loss_weight, simsiam_loss_weight = 1.0, 0.0
                else:
                    clip_loss_weight, simsiam_loss_weight = 0.0, 1.0
                loss = clip_loss * clip_loss_weight + simsiam_loss * simsiam_loss_weight
            # measure accuracy and record loss
            prec1, prec5 = accuracy(
                logits_per_image, target, topk=(1, self.topk))

            reduced_loss = loss.clone()
            reduced_clip_loss = clip_loss.clone()
            reduced_simsiam_loss = simsiam_loss.clone()
            reduced_text_simsiam_losses = text_simsiam_loss.clone()
            reduced_clip_nn_text_losses = clip_nn_text_loss.clone()
            reduced_text_mlm_loss = text_mlm_loss.clone()
            reduced_nt_xent_loss = nt_xent_loss.clone()
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            resume = False
            # if math.isnan(self.meters.losses.avg) or self.meters.losses.avg > train_loss + 0.5:
            if math.isnan(self.meters.losses.avg) or math.isnan(reduced_loss) or reduced_loss > self.meters.losses.avg + 10:
                resume = True
                self.logger.info(f'[ERROR] Training Loss Crashed,lr:{current_lr},prec1:{prec1},curr_step:{curr_step}, meters.losses.avg:{self.meters.losses.avg}')
            else:
                self.meters.losses.reduce_update(reduced_loss)
                self.meters.clip_losses.reduce_update(reduced_clip_loss)
                self.meters.simsiam_losses.reduce_update(reduced_simsiam_loss)
                self.meters.text_simsiam_losses.reduce_update(reduced_text_simsiam_losses)
                self.meters.clip_nn_text_losses.reduce_update(reduced_clip_nn_text_losses)
                self.meters.text_mlm_losses.reduce_update(reduced_text_mlm_loss)
                self.meters.nt_xent_losses.reduce_update(reduced_nt_xent_loss)
                self.meters.top1.reduce_update(reduced_prec1)
                self.meters.top5.reduce_update(reduced_prec5)

            # print(f'[Rank {self.dist.rank}] self.meters.losses.avg:{self.meters.losses.avg}, current_lr:{current_lr}', flush=True)

            if curr_step % 20 == 0:
                self.logger.info(f'[Rank {self.dist.rank}] self.meters.losses.avg:{self.meters.losses.avg}, current_lr:{current_lr}')
                train_loss = self.meters.losses.avg

            if curr_step % self.config.saver.print_freq == 0:
                last_checkpoint_step = ((curr_step -1) // 200 -5) * 200
                pretrain_path = f'{self.path.save_path}/ckpt_{last_checkpoint_step}.pth.tar'
                if self.dist.rank == 0:
                    if os.path.exists(pretrain_path):  # remove
                        cmd_ckpt = f'rm {pretrain_path}'
                        os.system(cmd_ckpt)

            if resume:
                link.barrier()
                #loss = clip_loss.mean()
                self.optimizer.zero_grad()
                
                self.optimizer.backward(loss)
                self.optimizer.step()
                torch.cuda.empty_cache()

                #last_checkpoint = self.find_last_checkpoint()
                #self.config.saver.pretrain.path = last_checkpoint
                #last_checkpoint_step = curr_step - 11
                for t in range(-3, 1):
                    last_checkpoint_step = ((curr_step -1) // 200 + t) * 200
                    self.config.saver.pretrain.path = f'{self.path.save_path}/ckpt_{last_checkpoint_step}.pth.tar'
                    if os.path.exists(self.config.saver.pretrain.path):
                        break
                if self.config.saver.pretrain.path is not None:  # RE-INIT
                    self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                    # self.logger.info(
                    #     f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
                    if 'model' in self.state:
                        load_state_model(self.model, self.state['model'])
                    self.config.lr_scheduler.kwargs.warmup_lr *= 0.997
                    self.build_optimizer()
                    #self.build_data()
                    self.build_lr_scheduler()
                    #start_step = last_checkpoint_step -i + 11
                    start_step = last_checkpoint_step - i
                    if self.dist.rank == 0:
                        self.logger.info(f'[ERROR] Training Loss Crashed,lr:{current_lr},start_step:{start_step}')
                        self.logger.info(f'[ERROR] recover from {self.config.saver.pretrain.path}')

                        #save curr_step ckpt & resave curr_step-1-2-3 ckpt
                        self.logger.info(f'save ckpt {last_checkpoint_step}')
                        curr_ckpt = f'{self.path.save_path}_k_times/ckpt_{last_checkpoint_step}.pth.tar'
                        cmd_ckpt = f'cp {self.config.saver.pretrain.path} {curr_ckpt}'
                        self.logger.info(f'save ckpt {curr_ckpt}')
                        os.system(cmd_ckpt)
                    continue
                else:
                    raise Exception('The training process crashed! And we cannot find a suitable ckpt')


            # compute and update gradient
            self.optimizer.zero_grad()

            def param_clip_before():
                if self.config.grad_clip.type == 'constant':
                    self.model.module.logit_scale.requires_grad = False
                elif self.config.grad_clip.type == 'logit_scale_param':
                    before = self.model.module.logit_scale.data.item()
                elif self.config.grad_clip.type == 'logit_scale_param_abs_min':
                    self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'logit_scale_param_value':  # clip param
                    self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value, max=self.config.grad_clip.max_value)
            def param_clip_after():
                if self.config.grad_clip.type == 'logit_scale_param':
                    after = self.model.module.logit_scale.data.item()
                    tem = self.model.module.logit_scale.data
                    if (after-before) > self.config.grad_clip.value:
                        self.model.module.logit_scale.data = torch.as_tensor(
                            before+self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
                    elif (before-after) > self.config.grad_clip.value:
                        self.model.module.logit_scale.data = torch.as_tensor(
                            before-self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
                elif self.config.grad_clip.type == 'logit_scale_param_abs_min':
                    self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'logit_scale_param_value':  # clip param
                    self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value, max=self.config.grad_clip.max_value)

            def grad_clip_before():  # before update(optimizer.step)
                if self.config.grad_clip.type == 'norm':
                    clip_grad_norm_(self.model.parameters(),
                                    self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'value':
                    clip_grad_value_(self.model.parameters(),
                                     self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'logit_scale_grad':
                    clip_param_grad_value_(
                        self.model.module.logit_scale, self.config.grad_clip.value)
            def grad_clip_after():
                pass

            param_clip_before()
            #print(f'Before link [Rank {self.dist.rank}] self.meters.losses.avg:{self.meters.losses.avg}, current_lr:{current_lr}, logit_scale:{self.model.module.logit_scale}', flush=True)
            if self.fused_fp16:
                self.optimizer.backward(loss)
                self.model.sync_gradients()
                grad_clip_before()
                self.optimizer.step()
                if self.config.get('check_grad', False):
                    self.check_model_and_grad(10)
                grad_clip_after()

            elif self.fp16 and 'FP16' in self.config.optimizer.type:
                def closure():
                    self.optimizer.backward(loss, False)
                    self.model.sync_gradients()
                    grad_clip_before()
                    # check overflow, convert to fp32 grads, downscale
                    self.optimizer.update_master_grads()
                    if self.config.get('check_grad', False):
                        self.check_model_and_grad(10)
                    grad_clip_after()
                    return loss
                self.optimizer.step(closure)
            else:
                loss.backward()
                self.model.sync_gradients()
                grad_clip_before()
                if self.config.get('check_grad', False):
                    self.check_model_and_grad(10)
                self.optimizer.step()
                grad_clip_after()

            # print(f'After link [Rank {self.dist.rank}] self.meters.losses.avg:{self.meters.losses.avg}, current_lr:{current_lr}', flush=True)
            param_clip_after()

            # clamp
            if self.config.grad_clip.type == 'logit_scale_param_ema':
                # if self.dist.rank == 0:
                #     print('*****************before_buffer', logit_scale.buffer)
                #     print('*****************before_param', logit_scale.param)
                logit_scale.clamp()
                logit_scale.update()
            # if self.dist.rank == 0:
                # print('*****************after_buffer', logit_scale.buffer)
                # print('*****************after_param', logit_scale.param)
                # EMA
            if self.ema is not None:
                self.ema.step(self.model, curr_step=curr_step)
            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)
            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar(
                    'loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar(
                    'clip_loss_train', self.meters.clip_losses.avg, curr_step)
                self.tb_logger.add_scalar(
                    'simsiam_loss_train', self.meters.simsiam_losses.avg, curr_step)
                self.tb_logger.add_scalar(
                    'nt_xent_loss_train', self.meters.nt_xent_losses.avg, curr_step)                
                self.tb_logger.add_scalar(
                    'text_mlm_loss', self.meters.text_mlm_losses.avg, curr_step)
                self.tb_logger.add_scalar(
                    'text_simsiam_loss_train', self.meters.text_simsiam_losses.avg, curr_step)
                # self.tb_logger.add_scalar(
                #     'simsiam_nn_text', self.meters.simsiam_nn_text_losses.avg, curr_step)
                self.tb_logger.add_scalar(
                    'clip_nn_text_loss', self.meters.clip_nn_text_losses.avg, curr_step)
                self.tb_logger.add_scalar(
                    'acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar(
                    'acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                self.tb_logger.add_scalar(
                    'logit_scale_exp', self.model.module.logit_scale.exp(), curr_step)
                self.tb_logger.add_scalar(
                    'logit_scale', self.model.module.logit_scale, curr_step)
                self.tb_logger.add_scalar(
                    'delta_logit_scale', self.model.module.logit_scale-last_logit_scale, curr_step)
                self.tb_logger.add_scalar(
                    'logit_scale_grad', self.model.module.logit_scale.grad, curr_step)
                self.tb_logger.add_scalar(
                    'clip_number', logit_scale.clip_number, curr_step)

                remain_secs = (total_step - curr_step) * \
                    self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                    # f'CLIP-NN-Text Loss {self.meters.clip_nn_text_losses.val:.4f} ({self.meters.clip_nn_text_losses.avg:.4f})\t' \
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Clip Loss {self.meters.clip_losses.val:.4f} ({self.meters.clip_losses.avg:.4f})\t' \
                    f'Simsiam Loss {self.meters.simsiam_losses.val:.4f} ({self.meters.simsiam_losses.avg:.4f})\t' \
                    f'Text Simsiam Loss {self.meters.text_simsiam_losses.val:.4f} ({self.meters.text_simsiam_losses.avg:.4f})\t' \
                    f'Clip NN Text Loss {self.meters.clip_nn_text_losses.val:.4f} ({self.meters.clip_nn_text_losses.avg:.4f})\t' \
                    f'Text MLM Loss {self.meters.text_mlm_losses.val:.4f} ({self.meters.text_mlm_losses.avg:.4f})\t' \
                    f'Nt_xent Loss {self.meters.nt_xent_losses.val:.4f} ({self.meters.nt_xent_losses.avg:.4f})\t' \
                    f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                    f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'logit_scale_exp {float(self.model.module.logit_scale.exp()):.4f}\t' \
                    f'logit_scale {float(self.model.module.logit_scale):.4f}\t' \
                    f'delta_logit_scale {float(self.model.module.logit_scale-last_logit_scale):.4f}\t' \
                    f'logit_scale_grad {float(self.model.module.logit_scale.grad):.4f}\t' \
                    f'clip_number {logit_scale.clip_number:.1f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})'
                self.logger.critical(log_msg)
            last_logit_scale = self.model.module.logit_scale.clone()

            # testing during training
            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                for id, val_data in enumerate(self.val_data):
                    metrics = self.evaluate(val_data)
                    if self.ema is not None:
                        self.ema.load_ema(self.model)
                        ema_metrics = self.evaluate(val_data)
                        self.ema.recover(self.model)
                        if self.dist.rank == 0 and self.config.data.test[id].test.evaluator.type == 'imagenet':
                            metric_key = 'top{}'.format(self.topk)
                            self.tb_logger.add_scalars(
                                'dataset_{}_acc1_val'.format(id), {'ema': ema_metrics.metric['top1']}, curr_step)
                            self.tb_logger.add_scalars(
                                'dataset_{}_acc5_val'.format(id), {'ema': ema_metrics.metric[metric_key]}, curr_step)

                    # testing logger
                    if self.dist.rank == 0 and self.config.data.test[id].test.evaluator.type == 'imagenet':
                        metric_key = 'top{}'.format(self.topk)
                        self.tb_logger.add_scalar(
                            'dataset_{}_acc1_val'.format(id), metrics.metric['top1'], curr_step)
                        self.tb_logger.add_scalar(
                            'dataset_{}_acc5_val'.format(id), metrics.metric[metric_key], curr_step)
            if curr_step > 0 and curr_step % self.config.saver.save_freq == 0:
                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    if self.ema is not None:
                        self.state['ema'] = self.ema.state_dict()
                    torch.save(self.state, ckpt_name)
                    if curr_step % (self.config.saver.save_freq*20) == 0:
                        print('save model kth')
                        k_times_save_path = f'{self.path.save_path}_k_times'
                        if not os.path.exists(k_times_save_path):
                            os.makedirs(k_times_save_path)
                        ckpt_name = f'{k_times_save_path}/ckpt_{curr_step}.pth.tar'
                        torch.save(self.state, ckpt_name)
            # link.barrier()

            end = time.time()
            # import ipdb
            # ipdb.set_trace()

    @torch.no_grad()
    def evaluate(self, val_data):
        self.model.eval()
        res_file = os.path.join(self.path.result_path,
                                f'results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        # label_ensemble
        # label_text, label_text_ensemble_matrix = val_data['loader'].dataset.get_label_texts(
        # )
        # label_text_preds = self.model.module.encode_text(label_text)
        # label_text_preds = label_text_preds / \
        #     (label_text_preds.norm(dim=-1, keepdim=True))

        label_text, label_text_ensemble_matrix = val_data['loader'].dataset.get_label_texts()
        label_num = label_text_ensemble_matrix.shape[1]
        prompts_num = len(label_text) // label_num
        self.logger.info('Use {} prompts'.format(prompts_num))
        label_text_preds = []
        for i in range(label_num):
            label_text_pred = self.model.module.encode_text(label_text[i*prompts_num:(i+1)*prompts_num])
            label_text_pred /= (label_text_pred.norm(dim=-1, keepdim=True))
            label_text_pred = label_text_pred.mean(dim=0)
            label_text_pred /= label_text_pred.norm()
            label_text_preds.append(label_text_pred)

        label_text_preds = torch.stack(label_text_preds, dim=0)

        label_text_ensemble_matrix = label_text_ensemble_matrix.to(
            label_text_preds)

        for batch_idx, batch in enumerate(val_data['loader']):
            input = batch['images']
            input = input.cuda().half() if self.fp16 else input.cuda()
            # label = label.squeeze().view(-1).cuda().long()
            # compute output
            if self.config.get('return_dense', False):
                image_preds, _ = self.model.module.encode_image(input, return_dense=True)
            else:
                image_preds = self.model.module.encode_image(input)
            image_preds = image_preds / \
                (image_preds.norm(dim=-1, keepdim=True))
            logits = image_preds @ label_text_preds.t()
            scores = F.softmax(logits, dim=1) @ label_text_ensemble_matrix
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            val_data['loader'].dataset.dump(writer, batch)

        writer.close()
        link.barrier()
        if self.dist.rank == 0:
            metrics = val_data['loader'].dataset.evaluate(res_file)
            self.logger.critical(json.dumps(metrics.metric, indent=2))
        else:
            metrics = {}
        link.barrier()
        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        self.model.train()
        return metrics


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    import prototype.solver.crash_on_ipy
    solver = ClsSolver(args.config)
    # evaluate or train
    if args.evaluate:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn(
                'Evaluating without resuming any solver checkpoints.')
        for id, val_data in enumerate(solver.val_data):
            solver.evaluate(val_data)
            if solver.ema is not None:
                solver.ema.load_ema(solver.model)
                solver.evaluate(val_data)
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
