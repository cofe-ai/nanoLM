import torch
import datasets
from torch.utils.data import DataLoader
from omegaconf import open_dict
from datasets.iterable_dataset import IterableDataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
)
from .copied_utils import (
    compute_input_and_target_lengths, 
    DataCollatorForT5MLM, 
    DataCollatorForLanguageModelingGPT,
    tokenize_function, 
    DataCollatorForNI,)

from .t5 import T5
from .bert import BertModel
from .gpt import GPT
from .llama import LlamaForCausalLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
import copy

def get_model(args, config):
    klass = {
        't5': T5,
        'bert': BertModel,
        'gpt': GPT,
        'llama': LlamaForCausalLM,
    }[args.class_name]

    if args.checkpoint_path:
        model = klass(config)
        model.load_state_dict(torch.load(args.model.checkpoint_path))
    elif args.random_init:
        model = klass(config)
    else:
        assert klass == T5ForConditionalGeneration, 'To load HFs weights you need to use HF model'
        model = klass.from_pretrained(
            args.model_name,
            config=config,
        )

    # with open_dict(args):
    args.n_all_param = sum([p.nelement() for p in model.parameters()])

    return model


def get_config(args):
    # 加载huggingface上的gpt参数
    config = AutoConfig.from_pretrained(
        args.model_name,
    )
    # 如果使用了mup，一些参数需要根据脚本重新设置
    
    # 如果有更改或者增加，在.config中更改或者增加
    # if hasattr(args.model, 'overwrite'):
    #     for k, v in args.model.overwrite.items():
    #         assert hasattr(config, k), f'config does not have attribute {k}'
    #         setattr(config, k, v)

    # if hasattr(args.model, 'add_config'):
    #     for k, v in args.items():
    #         assert not hasattr(config, k), f'config already has attribute {k}'
    #         setattr(config, k, v)
    
    # 加载 config 参数
    
    # config = copy.deepcopy(args)
    
    config.use_mup = args.use_mup
    config.output_mult = args.output_mult
    config.mup_base_width = args.mup_base_width
    config.zero_emb = args.zero_emb
    config.zero_query = args.zero_query
    config.d_model = args.hp_tune_actual_width
    config.n_embd = args.hp_tune_actual_width
    config.d_kv = 128
    config.num_heads = int(config.n_embd / config.d_kv)
    config.d_ff = 4 * config.d_model
    # if args.class_name == 'gpt':
    config.vocab_size = args.vocab_size
    config.block_size = args.block_size
    config.n_layer = args.n_layer
    config.bias = args.bias
    config.dropout = args.dropout
    config.initializer_range = args.initializer_range
    
    #llama
    config.rms_norm_eps=1e-06
    config.pretraining_tp = 1

    return config


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True
    )
    tokenizer.model_max_length = int(1e9)

    return tokenizer


def load_dataset_splits(args):
    if args.mode == 'pt':
        # dataset = datasets.load_dataset(
        #     'c4',
        #     'en',
        #     streaming= True,
        # )
        dataset = datasets.load_from_disk("data_mC4")
        # dataset = datasets.load_dataset(
        #     'Jackmin108/c4-en-validation',
        #     streaming= True,
        # ) # stas/openwebtext-10k
        # dataset = datasets.load_from_disk("/home/huangxiusheng/LLMTrainBench-dev-hxs/data")
        dataset = dataset.remove_columns(
            ['length']
        ) # ['timestamp', 'url']
        dataset_splits = {
            'train': dataset['train'],
            'test': dataset['train'],
        }
        # dataset_splits = {
        #     'train': dataset['train'],
        #     'test': dataset['validation'],
        # }

        # assert (
        #     dataset['train'].n_shards == 1024
        # ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"
    elif args.mode == 'ft':
        dataset_splits = datasets.load_dataset(
            args.data.exec_file_path,
            data_dir=args.data.data_dir,
            task_dir=args.data.task_dir,
            max_num_instances_per_task=args.data.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.data.max_num_instances_per_task
        )
    else:
        raise NotImplementedError

    return dataset_splits


def process_dataset(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            # We increase the input_length, because instead of masking tokens T5 replaces
            # masked spans with a single token, therefore to avoid padding we need to have
            # longer sequences at the start, before masking
            before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.input_length,
                noise_density=args.mlm_probability,
                mean_noise_span_length=args.mean_noise_span_length,
            )

            # with open_dict(args):
            args.before_mask_input_length = before_mask_input_length
            args.target_length = target_length

            dataset_split = dataset_split.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'in_length': before_mask_input_length,
                },
                remove_columns=['text'],
            )

            dataset_split = dataset_split.shuffle(seed=args.seed) #buffer_size=10,
            final_datasets[split] = dataset_split
    elif args.mode == 'ft':
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets


def get_data_collator(tokenizer, config, args):
    if  args.class_name == 't5':
        if args.mode == 'pt':
            data_collator = DataCollatorForT5MLM(
                tokenizer=tokenizer,
                noise_density=args.mlm_probability,
                mean_noise_span_length=args.mean_noise_span_length,
                input_length=args.input_length,
                target_length=args.target_length,
                pad_token_id=config.pad_token_id,
            )
        elif args.mode == 'ft':
            data_collator = DataCollatorForNI(
                tokenizer,
                padding="longest",
                max_source_length=args.data.max_seq_len,
                max_target_length=args.data.max_target_len,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
                add_task_name=args.data.add_task_name,
                add_task_definition=args.data.add_task_definition,
                num_pos_examples=args.data.num_pos_examples,
                num_neg_examples=args.data.num_neg_examples,
                add_explanation=args.data.add_explanation,
                tk_instruct=args.data.tk_instruct
            )
        else:
            raise NotImplementedError
    elif args.class_name == 'bert':
        if args.mode == 'pt':
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm = True
                # pad_to_multiple_of = 512
            )
        elif args.mode == 'ft':
            data_collator = DataCollatorForNI(
                tokenizer,
                padding="longest",
                max_source_length=args.data.max_seq_len,
                max_target_length=args.data.max_target_len,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
                add_task_name=args.data.add_task_name,
                add_task_definition=args.data.add_task_definition,
                num_pos_examples=args.data.num_pos_examples,
                num_neg_examples=args.data.num_neg_examples,
                add_explanation=args.data.add_explanation,
                tk_instruct=args.data.tk_instruct
            )
    elif args.class_name == 'gpt' or args.class_name == 'llama':
        if args.mode == 'pt':
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm = False,
                # pad_to_multiple_of = 512
            )
            # data_collator = DataCollatorForLanguageModelingGPT(
            #     tokenizer=tokenizer,
            #     mlm=False,
            #     # pad_to_multiple_of=8,
            # )

        elif args.mode == 'ft':
            data_collator = DataCollatorForNI(
                tokenizer,
                padding="longest",
                max_source_length=args.data.max_seq_len,
                max_target_length=args.data.max_target_len,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
                add_task_name=args.data.add_task_name,
                add_task_definition=args.data.add_task_definition,
                num_pos_examples=args.data.num_pos_examples,
                num_neg_examples=args.data.num_neg_examples,
                add_explanation=args.data.add_explanation,
                tk_instruct=args.data.tk_instruct
            )
        else:
            raise NotImplementedError
    return data_collator


def get_dataloaders(tokenizer, config, args):
    dataset_splits = load_dataset_splits(args)
    dataset = process_dataset(dataset_splits=dataset_splits, args=args, tokenizer=tokenizer)
    #dataset = dataset.with_format("torch")
    data_collator = get_data_collator(tokenizer=tokenizer, config=config,
                                      args=args)

    is_iterable = isinstance(dataset['train'], IterableDataset)

    dataloaders = {}

    for split in ['train', 'test']:
        batch_size = args.batch_size // args.grad_acc

        # shuffle = (split == 'train') and not is_iterable
        shuffle = False
        if args.mode == 'ft' and split == 'train':
            assert shuffle is True
        else:
            assert shuffle is False
        dataset[split] = dataset[split].with_format("torch")
        dataloaders[split] = DataLoader(
            dataset[split],
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # Add & Check args about data loaders
    # with open_dict(args):
    if not is_iterable:
        args.train_batches = len(dataloaders['train'])
        args.test_batches = len(dataloaders['test'])

    if args.epochs > 0:
        assert not is_iterable
        args.total_steps = (len(dataloaders['train']) // args.grad_acc) * args.epochs 

    args.corrected_steps = args.steps

    return dataloaders['train'], dataloaders['test']


def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    
    if not model.use_mup:
        # 维持原本分组
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
    else:
        # 在原有基础上，按照是否matrix-like重新分组
        width_mult = model.model_dim / model.mup_base_width
        matrix_like = ["SelfAttention.q.weight", "SelfAttention.k.weight", "SelfAttention.v.weight",
                       "SelfAttention.o.weight", "wi_0.weight", "wi_1.weight", "wo.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                           if any(kw in n for kw in matrix_like)],
                "weight_decay": args.weight_decay * width_mult,
                "width_mult": width_mult,
                "lr": args.base_lr / width_mult,
                "name": "matrix-like-decay"
            },
            {
                "params": [p for n, p in model.named_parameters() 
                           if not any(kw in n for kw in matrix_like) and not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
                "width_mult": 1.0,
                "lr": args.base_lr,
                "name": "others-decay"
            },
            {
                "params": [p for n, p in model.named_parameters() 
                           if not any(kw in n for kw in matrix_like) and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "width_mult": 1.0,
                "lr": args.base_lr,
                "name": "others-no-decay"
            }
            
        ]

    if args.name == 'adamw':
        from transformers import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.base_lr,
        )
    elif args.name == 'adamwscale':
        from .copied_utils import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.base_lr,
        )
    elif args.name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, args, logger):
    if args.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
        )
        if args.use_mup:
            from .utils_for_mup import CosineAnnealingLR
        else:
            from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.total_steps - args.warmup_steps,
            eta_min=args.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.warmup_steps]
        )
    elif args.lr_scheduler == 'legacy':
        assert not args.use_mup
        import math
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            LambdaLR,
        )

        msg = "You are using T5 legacy LR Schedule, it's independent from the base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.total_steps * 0.9)
        iters_left_for_optimizer2 = args.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / args.base_lr if step else 1e-2 / args.base_lr
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif args.lr_scheduler == 'constant':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError

    return lr_scheduler
