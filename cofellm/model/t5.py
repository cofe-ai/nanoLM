# From: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

import copy
import math
from typing import Optional
from dataclasses import dataclass
from torch.nn import functional as F
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5DenseGatedActDense

from .transoformer import get_transformers
from .module.attention import Attention
from .utils_for_mup import _exact

@dataclass
class EncoderOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    encoder_outputs: EncoderOutput = None




class T5(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        config.is_encoder_decoder = False
        assert not config.tie_word_embeddings

        self.config = config
        self.model_dim = config.d_model
        # mup
        self.use_mup = config.use_mup
        self.output_mult = config.output_mult if self.use_mup else 1.0
        self.mup_base_width = config.mup_base_width
        self.zero_query = config.zero_query
        self.zero_emb = config.zero_emb
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = get_transformers(encoder_config, self.shared)


        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = get_transformers(decoder_config, self.shared)


        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.generation_config = None
        self.qa_outputs = nn.Linear(config.d_model, config.vocab_size)

        if self.use_mup:
            self.apply(self._init_weights_with_mup)
            self.encoder.emb_layer_norm.bias.data.normal_(0,0.1)
            self.decoder.emb_layer_norm.bias.data.normal_(0,0.1)
        else:
            self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,)) #128: 8.82M； 256: 22.36M
        a = 0


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params


    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_length = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
            input_ids: B x L_encoder, int64
            attention_mask: B x L_encoder, int64
                1 for tokens to attend to, 0 for tokens to ignore
            
            Generation:
                Starts with 0, ends with 1, padding is 0

            # For 20 input/outputs, the diff between my implementation and HF is 9.8s vs 11.4s
        """
        B, _ = input_ids.size()
        labels = torch.zeros(B, 1, dtype=torch.long, device=input_ids.device)
        encoder_outputs = None

        for _ in range(max_length):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=labels,
                encoder_outputs=encoder_outputs,
            )
            encoder_outputs = out.encoder_outputs
            top_labels = out.logits[:, -1].argmax(-1).unsqueeze(-1)
            labels = torch.cat([labels, top_labels], dim=-1)

            if (labels == 1).sum(-1).clamp(min=0, max=1).sum().item() == B:
                break
        
        labels[:, -1] = 1

        # Mask out the padding, i.e., all positions after the first 1 with 0
        B, L = labels.size()
        mask = torch.arange(L, device=labels.device).unsqueeze(0) <= (labels == 1).long().argmax(-1).unsqueeze(-1)
        labels = labels.masked_fill(~mask, 0)

        return labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_outputs = None,
    ) -> Seq2SeqLMOutput:
        """
            input_ids: B x L_encoder, int64
            attention_mask: B x L_encoder, int64
                1 for tokens to attend to, 0 for tokens to ignore
            labels: B x L_decoder, int64
        """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = encoder_outputs.hidden_states # 32, 512, 768

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
            
            
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        sequence_output = decoder_outputs[0]  # 32, 114, 768
        lm_logits = self.lm_head(sequence_output)
        # label 32, 114
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)) # 32, 114, 32128

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            encoder_outputs=encoder_outputs,
        )

    def _init_weights(self, module):
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseGatedActDense):
            d_ff, d_model = module.wi_0.weight.data.size()
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((d_ff) ** -0.5))
        elif isinstance(module, Attention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if hasattr(module, "relative_attention_bias"):
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    # mup
    def _init_weights_with_mup(self, module):
        
        factor = self.config.initializer_factor  # 初始分布超参数
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5)):
            # 没有设定初始化为全零的情况下，属于others
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            _exact(module.shared.weight, 0., factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
                _exact(module.lm_head.weight, 0., factor * 1.0)
            
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * 1.0)
                module.qa_outputs.bias.data.zero_()
                _exact(module.qa_outputs.weight, 0., factor * 1.0)
                
            if self.zero_emb:
                module.shared.weight.data.zero_()
                if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                    module.lm_head.weight.data.zero_()
                if hasattr(module, "qa_outputs"):
                    module.qa_outputs.weight.data.zero_()
                    
                
        elif isinstance(module, T5DenseGatedActDense):
            d_ff, d_model = module.wi_0.weight.data.size()
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((d_ff) ** -0.5))
            _exact(module.wi_0.weight, 0.0, factor * ((d_model) ** -0.5))
            _exact(module.wi_1.weight, 0.0, factor * ((d_model) ** -0.5))
            _exact(module.wo.weight, 0.0, factor * ((d_ff) ** -0.5))
            
        elif isinstance(module, Attention):
            d_model = self.config.d_model
            # 需确保d_kv对所有实验是个常数
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            if not self.zero_query:
                module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
                _exact(module.q.weight, 0.0, factor * ((d_model * key_value_proj_dim) ** -0.5))
            else:
                module.q.weight.data.zero_()
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            
            _exact(module.k.weight, 0.0, factor * ((d_model) ** -0.5))
            _exact(module.v.weight, 0.0, factor * ((d_model) ** -0.5))
            _exact(module.o.weight, 0.0, factor * ((n_heads * key_value_proj_dim) ** -0.5))
            
            if hasattr(module, "relative_attention_bias"):
                # scalar-like, 初始化全零即可
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=0.0)
                
                
                
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None and pad_token_id is not None
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids