from dataclasses import dataclass

import torch
from torch import nn

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5DenseGatedActDense,
)
from .module.attention import Attention

@dataclass
class EncoderOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None

class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = Attention(config, has_relative_attention_bias=False)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs

class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        assert config.is_gated_act
        self.DenseReluDense = T5DenseGatedActDense(config)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states).type_as(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states



class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
            )
            hidden_states = cross_attention_outputs[0]

            # Keep relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        outputs = (hidden_states,)
        outputs = outputs + attention_outputs

        return outputs  # hidden-states, (self-attention position bias), (cross-attention position bias)



class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states).type_as(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs



class get_transformers(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, embed_tokens):
        super().__init__()
        assert embed_tokens is not None

        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        # mup
        self.use_mup = config.use_mup
        
        if self.use_mup:
            self.emb_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ) -> EncoderOutput:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        inputs_embeds = self.embed_tokens(input_ids)

        if hasattr(self.config, 'is_bf16') and self.config.is_bf16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        # Masking
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        position_bias = None
        encoder_decoder_position_bias = None
        
        if self.use_mup:
            inputs_embeds = self.emb_layer_norm(inputs_embeds)

        hidden_states = self.dropout(inputs_embeds)

        for _, layer_module in enumerate(self.block):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
            )
            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[2]
        
        hidden_states = self.final_layer_norm(hidden_states).type_as(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return EncoderOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
