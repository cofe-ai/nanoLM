import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='pt', help="")
    parser.add_argument("--device", default='gpu', help="")
    parser.add_argument("--precision", default='bf16', help="")
    parser.add_argument("--eval_only", default= False, help="")
    parser.add_argument("--predict_only", default=False, help="")
    parser.add_argument("--seed", default=2137,  type=int, help="")
    parser.add_argument("--initializer_range", default=0.05,  type=float, help="")
    
    # model
    parser.add_argument("--class_name", default='llama', help="t5, bert, gpt, llama") # 
    parser.add_argument("--model_name", default='google/t5-v1_1-base', help="")
    parser.add_argument("--tokenizer_path", default='google/t5-v1_1-base',  help="google/t5-v1_1-base, bert-base-uncased")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="")
    parser.add_argument("--is_bf16", default=False, help="")
    parser.add_argument("--checkpoint_path", default="", help="")
    parser.add_argument("--random_init", default=True, help="")
    parser.add_argument("--compile", default=False, help="")
    parser.add_argument("--vocab_size", default=50304, type=int, help="")
    parser.add_argument("--block_size", default=1024, type=int, help="")
    parser.add_argument("--n_layer", default=32, type=int, help="")
    parser.add_argument("--num_heads", default=12, type=int, help="")
    parser.add_argument("--bias", default=True, type=bool, help="") 
    parser.add_argument("--dropout", default=0.0, type=float, help="")

    # mup
    parser.add_argument("--use_mup", action="store_true", help="")
    parser.add_argument("--output_mult", default=1.0, type=float, help="")
    parser.add_argument("--mup_base_width", default=256, type=int, help="")
    parser.add_argument("--hp_tune_actual_width", default=128, type=int, help="")
    parser.add_argument("--zero_query", default=True, help="")
    parser.add_argument("--zero_emb", default=True, help="")
    
    # data
    parser.add_argument("--input_length", default=512, type=int, help="")
    parser.add_argument("--mlm_probability", default=0.15, type=float, help="")
    parser.add_argument("--mean_noise_span_length", default=3.0, type=float, help="")
    parser.add_argument("--num_workers", default=2, help="")
    
    # optim
    parser.add_argument("--name", default='adamwscale', help="")
    parser.add_argument("--base_lr", default=2e-2, type=float, help="")
    parser.add_argument("--batch_size", default=16, type=int, help="")
    parser.add_argument("--total_steps", default=20000, type=int, help="")
    parser.add_argument("--epochs", default=-1, type=int, help="")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="")
    parser.add_argument("--lr_scheduler", default='cosine', help="")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="")
    parser.add_argument("--grad_acc", default=1, type=int, help="")
    parser.add_argument("--final_cosine", default=1e-5, type=float, help="")
    
    # eval
    parser.add_argument("--eval_every_steps", default=100000, type=int, help="")
    parser.add_argument("--steps", default=500, type=int, help="")
    
    # checkpoint
    parser.add_argument("--checkpoint_every_steps", default=1000000, type=int, help="")
    
    # logging
    parser.add_argument("--neptune", default=False, help="")
    parser.add_argument("--project", default='', help="")
    parser.add_argument("--api_token", default='', help="")
    parser.add_argument("--tags", default='', help="")
    parser.add_argument("--logging_every_steps", default=100, type=int, help="")
    parser.add_argument("--grad_l2", default=True, help="")
    parser.add_argument("--weights_l2", default=True, help="")
    
    return parser