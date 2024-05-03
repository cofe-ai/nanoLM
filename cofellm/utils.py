import torch
import os

from accelerate.utils import set_seed
from omegaconf import open_dict
from hydra.utils import to_absolute_path
from collections import defaultdict

from accelerate.logging import get_logger
from omegaconf import OmegaConf, open_dict
import logging
import datasets
import transformers
import neptune
import os
import time

class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key] for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats


class Logger:
    def __init__(self, args, accelerator):
        self.logger = get_logger('Main')

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is {os.getcwd()}')

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        self.setup_neptune(args)

    def setup_neptune(self, args):
        if args.neptune:
            neptune_logger = neptune.init_run(
                project=args.neptune_creds.project,
                api_token=args.neptune_creds.api_token,
                tags=[str(item) for item in args.neptune_creds.tags.split(",")],
            )
        else:
            neptune_logger = None

        self.neptune_logger = neptune_logger

        # with open_dict(args):
        if neptune_logger is not None:
            args.neptune_id = neptune_logger["sys/id"].fetch()

    def log_args(self, args):
        if self.neptune_logger is not None:
            logging_args = OmegaConf.to_container(args, resolve=True)
            self.neptune_logger['args'] = logging_args

    def log_stats(self, stats, step, args, prefix=''):
        if self.neptune_logger is not None:
            for k, v in stats.items():
                self.neptune_logger[f'{prefix}{k}'].log(v, step=step)

        msg_start = f'[{prefix[:-1]}] Step {step} out of {args.total_steps}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.5f}' for k, v in stats.items()]) + ' | '
        argument = '| hp_tune_actual_width:' + f'{args.hp_tune_actual_width}'+ ' | bae_lr:' + f'{args.base_lr }'+ ' | model_klass:' + f'{args.class_name}'  + ' | '

        msg = msg_start + dict_msg + argument
        if args.current_train_step + args.logging_every_steps > args.total_steps:
            file_path = 'logs/' f'{args.class_name}' + '_train_logs.text'
            with open(file_path, 'a') as file:
                file.writelines(msg)
                file.writelines('\n')
                file.writelines(time.ctime())
                file.writelines('\n')

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)

    def finish(self):
        if self.neptune_logger is not None:
            self.neptune_logger.stop()


def check_args_and_env(args):
    assert args.batch_size % args.grad_acc == 0

    # Train log must happen before eval log
    assert args.eval_every_steps % args.logging_every_steps == 0

    if args.device == 'gpu':
        print(torch.cuda.is_available())
        assert torch.cuda.is_available(), 'We use GPU to train/eval the model'

    assert not (args.eval_only and args.predict_only)

    if args.predict_only:
        assert args.mode == 'ft'


def opti_flags(args):
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.precision == 'bf16' and args.device == 'gpu' and args.class_name == 't5-':
        args.is_bf16 = True


def update_args_with_env_info(args):
    # with open_dict(args):
    slurm_id = os.getenv('SLURM_JOB_ID')

    if slurm_id is not None:
        args.slurm_id = slurm_id
    else:
        args.slurm_id = 'none'

    args.working_dir = os.getcwd()


def update_paths(args):
    if args.mode == 'ft':
        args.data.exec_file_path = to_absolute_path(args.data.exec_file_path)
        args.data.data_dir = to_absolute_path(args.data.data_dir)
        args.data.task_dir = to_absolute_path(args.data.task_dir)


def setup_basics(accelerator, args):
    check_args_and_env(args)
    update_args_with_env_info(args)
    update_paths(args)
    opti_flags(args)

    if args.seed is not None:
        set_seed(args.seed)

    logger = Logger(args=args, accelerator=accelerator)

    return logger
