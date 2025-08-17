from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import PretrainedConfig
from transformers.utils.deprecation import deprecate_kwarg

from model.kv_caches.cache_utils import Cache, StaticCache, DynamicCache, RecursiveDynamicCache
from model.base_model.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
)
from model.mor_model.modeling_llama import (
    MoRBaseModelOutputWithPast,
    MoRCausalLMOutputWithPast,
)

logger = logging.get_logger(__name__)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class ViTConfig(PretrainedConfig):
    model_type = "vit_mor"
    
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=1000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Required by MOR layers
        self.num_key_value_heads = num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        
        # Computed properties
        self.num_patches = (image_size // patch_size) ** 2
        self.max_position_embeddings = self.num_patches + 1  # +1 for CLS token
        
        # Set properties needed by MOR
        self.vocab_size = num_labels
        self.rms_norm_eps = layer_norm_eps
        self.attention_dropout = attention_probs_dropout_prob
        self.mlp_bias = False
        self.attention_bias = False


class ViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embeddings
        self.patch_embeddings = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_size)
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        
        # Create patch embeddings
        embeddings = self.patch_embeddings(pixel_values)  # (B, hidden_size, H/P, W/P)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # (B, num_patches+1, hidden_size)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ViTAttention(nn.Module):
    """Multi-headed attention for ViT - removes RoPE and causal masking"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_probs_dropout_prob
        self.is_causal = False  # ViT uses bidirectional attention
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) 
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # No RoPE for ViT
        
        if past_key_value is not None:
            # For MOR caching
            cache_kwargs = {"cache_position": cache_position}
            if "selected_tokens" in kwargs:
                cache_kwargs["selected_tokens"] = kwargs["selected_tokens"]
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Use eager attention for simplicity
        from model.base_model.modeling_llama import eager_attention_forward
        
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ViTDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = ViTAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            
        return outputs



class MoRViTModel(nn.Module):
    """ViT model adapted for MOR"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        
        self.layers = nn.ModuleList([
            ViTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Use single-hook checkpoint to avoid double-marking
        from torch.utils.checkpoint import checkpoint
        self._gradient_checkpointing_func = lambda func, *args: checkpoint(func, *args, use_reentrant=False)
        
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
        num_items_in_batch: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
     ) -> Union[Tuple, MoRBaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, 'use_cache') else False
        return_dict = return_dict if return_dict is not None else True
        
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(pixel_values)
            
        if use_cache and past_key_values is None:
            if hasattr(self.config, "kv_sharing") and self.config.kv_sharing is not None:
                kwargs = self.config.kv_sharing
                past_key_values = RecursiveDynamicCache(kwargs["base_depth"], kwargs["num_recursion"], kwargs["sharing"], kwargs["update_cache"])
            else:
                past_key_values = DynamicCache()
                
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        # Simple attention mask for ViT (no causal masking)
        if attention_mask is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
        hidden_states = inputs_embeds
        
        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # MOR tracking variables
        prev_selected_tokens = None
        sampling_loss = torch.tensor(0.0, device=hidden_states.device)
        sampling_acc_list = []
        sampling_topk_acc_list = []
        uniformity = None
        dead_token_seq = None
        balancing_loss = torch.tensor(0.0, device=hidden_states.device)
        balancing_ratio = torch.tensor(0.0, device=hidden_states.device)
        router_z_loss = torch.tensor(0.0, device=hidden_states.device)
        
        for decoder_layer in self.layers[:self.config.num_hidden_layers]: 
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Use checkpoint only for pure ViT layers (no MOR routing)
            if self.gradient_checkpointing and self.training and not getattr(decoder_layer, "mor", False):
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    None,  # no position embeddings in ViT
                )
            else:
                if hasattr(decoder_layer, "mor") and decoder_layer.mor:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=None,
                        prev_selected_tokens=prev_selected_tokens,
                        num_items_in_batch=num_items_in_batch,
                        **flash_attn_kwargs,
                    )
                    
                    # Handle MOR outputs
                    if hasattr(decoder_layer, 'mor_type'):
                        if decoder_layer.mor_type == "expert":
                            prev_selected_tokens = layer_outputs.selected_tokens
                            if layer_outputs.sampling_loss is not None:
                                sampling_loss += layer_outputs.sampling_loss
                            if layer_outputs.sampling_acc is not None:
                                sampling_acc_list.append(layer_outputs.sampling_acc)
                            if layer_outputs.sampling_topk_acc is not None:
                                sampling_topk_acc_list.append(layer_outputs.sampling_topk_acc)
                            if layer_outputs.router_z_loss is not None:
                                router_z_loss += layer_outputs.router_z_loss
                                
                        elif decoder_layer.mor_type == "token":
                            if layer_outputs.balancing_loss is not None:
                                balancing_loss = layer_outputs.balancing_loss
                            if layer_outputs.balancing_ratio is not None:
                                balancing_ratio = layer_outputs.balancing_ratio
                            if layer_outputs.router_z_loss is not None:
                                router_z_loss = layer_outputs.router_z_loss
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=None,
                        prev_selected_tokens=prev_selected_tokens,
                        num_items_in_batch=num_items_in_batch,
                        **flash_attn_kwargs,
                    )
                    
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        output = MoRBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            sampling_loss=sampling_loss,
            sampling_acc=sum(sampling_acc_list)/len(sampling_acc_list) if len(sampling_acc_list) > 0 else torch.tensor(0.0, device=hidden_states.device),
            sampling_topk_acc=sum(sampling_topk_acc_list)/len(sampling_topk_acc_list) if len(sampling_topk_acc_list) > 0 else torch.tensor(0.0, device=hidden_states.device),
            uniformity=uniformity,
            dead_token_seq=dead_token_seq,
            balancing_loss=balancing_loss,
            balancing_ratio=balancing_ratio,
            router_z_loss=router_z_loss,
        )
        return output if return_dict else output.to_tuple()



class MoRViTForImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MoRViTModel(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Indicate support for gradient checkpointing
        self.supports_gradient_checkpointing = True

    
    def gradient_checkpointing_enable(self, **kwargs):
        """Hook called by Trainer to enable gradient checkpointing."""
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Hook called by Trainer to disable gradient checkpointing."""
        self.model.gradient_checkpointing = False

    def forward(
         self,
         pixel_values: torch.FloatTensor = None,
         attention_mask: Optional[torch.Tensor] = None,
         position_ids: Optional[torch.LongTensor] = None,
         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
         inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
         use_cache: Optional[bool] = None,
         output_attentions: Optional[bool] = None,
         output_hidden_states: Optional[bool] = None,
         return_dict: Optional[bool] = None,
         cache_position: Optional[torch.LongTensor] = None,
        num_items_in_batch: Optional[int] = None,
         **kwargs,
     ) -> Union[Tuple, MoRCausalLMOutputWithPast]:
        
        outputs = self.model(
            pixel_values=pixel_values,
            num_items_in_batch=num_items_in_batch,
            **kwargs
        )

        cls_out = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_out)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return MoRCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sampling_loss=outputs.sampling_loss,
            sampling_acc=outputs.sampling_acc,
            sampling_topk_acc=outputs.sampling_topk_acc,
            uniformity=outputs.uniformity,
            dead_token_seq=outputs.dead_token_seq,
            balancing_loss=outputs.balancing_loss,
            balancing_ratio=outputs.balancing_ratio,
            router_z_loss=outputs.router_z_loss,
        )

        
    def transform_layer_to_mor_expert(self, cfg):
        from model.mor_model.expert_choice_router import MoRLlamaDecoderLayer
        
        capacity = [float(cap) for cap in cfg.mor.capacity.split(',')]
        if "cap_warmup_step" in cfg.mor.expert and cfg.mor.expert.cap_warmup_step is not None:
            cap_warmup_step = cfg.mor.expert.cap_warmup_step
        else:
            cap_warmup_step = cfg.num_warmup_steps * cfg.gradient_accumulation_steps
        
        sharing = cfg.recursive.sharing
        num_recursion = cfg.recursive.num_recursion 
        num_hidden_layers = len(self.model.layers)
        
        if sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            self.model.layers = nn.ModuleList([
                MoRLlamaDecoderLayer(self.config, nn.ModuleList([self.model.layers[layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]), 
                                   cfg, capacity[recur_idx], cap_warmup_step,) 
                for recur_idx in range(num_recursion)
            ])
        elif sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            self.model.layers = nn.ModuleList(
                [self.model.layers[0]] + 
                [MoRLlamaDecoderLayer(self.config, nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]), 
                                    cfg, capacity[recur_idx], cap_warmup_step,)
                 for recur_idx in range(num_recursion)] +
                [self.model.layers[-1]]
            )
            
    def transform_layer_to_mor_token(self, cfg):
        from model.mor_model.token_choice_router import MoRLlamaDecoderLayer
        
        bal_warmup_step = 0
        if "bal_warmup_step" in cfg.mor.token and cfg.mor.token.bal_warmup_step > 0:
            bal_warmup_step = cfg.mor.token.bal_warmup_step * cfg.gradient_accumulation_steps
        
        sharing = cfg.recursive.sharing
        num_recursion = cfg.recursive.num_recursion 
        num_hidden_layers = len(self.model.layers)
        
        if sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            self.model.layers = MoRLlamaDecoderLayer(
                self.config,
                nn.ModuleList([nn.ModuleList([self.model.layers[layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]) for recur_idx in range(num_recursion)]),
                cfg,
                bal_warmup_step,
            )
        elif sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            self.model.layers = nn.ModuleList(
                [self.model.layers[0]] + 
                [MoRLlamaDecoderLayer(
                    self.config,
                    nn.ModuleList([nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]) for recur_idx in range(num_recursion)]),
                    cfg,
                    bal_warmup_step,
                ),] + 
                [self.model.layers[-1]]
            )
            
    def set_kv_sharing_config(self, cfg):
        if cfg.kv_sharing.sharing in ["cycle", "sequence"]:
            base_depth = self.config.num_hidden_layers // cfg.kv_sharing.num_recursion
        elif cfg.kv_sharing.sharing in ["middle_cycle"]:
            base_depth = (self.config.num_hidden_layers - 2) // cfg.kv_sharing.num_recursion
        
        if "kv_sharing" in cfg: 
            kwargs = {
                "enable": cfg.kv_sharing.enable,
                "base_depth": base_depth,
                "num_recursion": cfg.kv_sharing.num_recursion,
                "sharing": cfg.kv_sharing.sharing,
                "update_cache": cfg.kv_sharing.update_cache if "update_cache" in cfg.kv_sharing else False,
            } 
            self.model.config.kv_sharing = kwargs
        else:
            self.model.config.kv_sharing = None
    
    # def forward(
    #     self,
    #     pixel_values: torch.FloatTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs,
    # ) -> Union[Tuple, MoRCausalLMOutputWithPast]:
        
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
    #     output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
    #     return_dict = return_dict if return_dict is not None else True
        
    #     # Forward through the model
    #     outputs = self.model(
    #         pixel_values=pixel_values,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #         cache_position=cache_position,
    #         **kwargs,
    #     )
        
    #     hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        
    #     # Use CLS token for classification (first token)
    #     cls_output = hidden_states[:, 0]  # (batch_size, hidden_size)
    #     logits = self.classifier(cls_output)
        
    #     loss = None
    #     if labels is not None:
    #         loss_fct = nn.CrossEntropyLoss()
    #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
    #     if not return_dict:
    #         output = (logits,) + outputs[1:]
    #         return (loss,) + output if loss is not None else output
        
    #     return MoRCausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #         sampling_loss=outputs.sampling_loss,
    #         sampling_acc=outputs.sampling_acc,
    #         sampling_topk_acc=outputs.sampling_topk_acc,
    #         uniformity=outputs.uniformity,
    #         dead_token_seq=outputs.dead_token_seq,
    #         balancing_loss=outputs.balancing_loss,
    #         balancing_ratio=outputs.balancing_ratio,
    #         router_z_loss=outputs.router_z_loss,
    #     )
