import argparse
from accelerate import Accelerator
from omegaconf import open_dict
import hydra
import torch
import time
import os
from cofellm.train_utils import (
    train, 
    predict,
    eval
) 
from cofellm.model.utils import (
    get_config,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
)
from cofellm.utils import setup_basics
from cofellm.arguments import get_argparse
import pdb

def main():
    args = get_argparse().parse_args()
    print("########" + str(args.use_mup))
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
    )
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    model = get_model(args, config)
    tokenizer = get_tokenizer(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    logger.log_args(args)

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )
    for n, p in model.named_parameters():
        print(n, p.size(), p.mean().item(), p.var().item())

    if args.compile:
        model = torch.compile(model)

    args.current_train_step = 1
    args.current_epoch = 1
    args.last_log = time.time()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval(model, test_dataloader, logger, args, tokenizer)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger,
                    args, tokenizer)
    else:
        train(model, train_dataloader, test_dataloader, accelerator,
              lr_scheduler, optimizer, logger, args, tokenizer)

    logger.finish()


if __name__ == "__main__":
    main()
    