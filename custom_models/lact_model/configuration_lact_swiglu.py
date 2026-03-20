# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class LaCTSWIGLUConfig(PretrainedConfig):
    """
    Configuration for LaCT-SWIGLU model.
    It implements the LaCT-SWIGLU layer mixed with in-layer sliding window attention

    Args:
        hidden_size (int, optional): The hidden size of the model. Defaults to 2048.
        num_hidden_layers (int, optional): The number of hidden layers in the model. Defaults to 24.
        num_attn_heads (int, optional): The number of attention heads in the model. Defaults to 32.
        num_lact_heads (int, optional): The number of feed-forward heads in the model. Defaults to 4.
    """

    model_type = "lact_swiglu"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attn_heads: int = 32,
        num_lact_heads: int = 4,
        inter_multi: int = 1,
        qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        lact_chunk_size: int = 2048,
        use_muon: bool = False,
        lr_dim: int = 1,
        qkv_silu: bool = True,  # if True, apply silu to q, k, v.
        no_v_silu: bool = False,  # if True, don't apply silu to v, will overwrite qkv_silu.
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = True,
        use_momentum: bool = True,
        ttt_loss_type: str = "dot_product",  # "dot_product" or "delta_rule"
        ttt_prenorm: bool = True,  # pre-norm or post-norm for ttt.
        # prenorm ttt:  state = state + f(norm(state))
        # postnorm ttt:  state = norm(state + f(state)
        ttt_nope: bool = False,  # if True, no positional encoding for query and key used in ttt.
        w0_w2_low_rank: int = -1,  # -1 means fully learnable.  > 1 means low rank parameterization of the initial learnable weights.
        window_size: int = 2048,
        rope_theta: Optional[float] = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.006,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        last_layer_fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        fw_init_gain: float = 0.5,
        use_fused_kernel: bool = False,  # use triton kernel for ttt implementation
        fp32_states: bool = False,  # whether to keep the fast weights in fp32
        # The important ablation factors
        w_reg_lrs: list[float] = None,  # Activates weight decays. Learnable if [None, None, None]. Fixed if [float, float, float].
        clip_grad_norm: float = None,  # Clips the gradient norm of the fast weights.
        weight_norm: bool = True,  # Applies weight normalization to the fast weights.
        track_states: bool = False,  # Tracks the states of the fast weights.
        # The rest of the ablation factors used during exploration
        ablation: bool = False,  # True if we want to use the following ablations factors. Used during exploration.
        w_reg_mode: str = None,  # "init" or "zero" or None (default)
        fwd_mode: str = "gdn",  # Architecture of the fast weights. {gdn, gdn+, factorized_gdn, factorized_gdn+, lact, lact+}
        update_then_apply: bool = False,  # Whether to update the fast weights then apply them to the hidden states.    
        normalize: bool = False,  # Normalizes the hidden states in the fast weights.
        scale: bool = False,  # Scales the fast weights.
        mean_reg_lr: bool = False,  # Uses the mean of the fast weights for regularization.
        no_reg_lr: bool = False,  # Does not use regularization for the fast weights.
        rand_w1: bool = False,  # Randomly initializes the fast weights.
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_heads = num_attn_heads
        self.num_lact_heads = num_lact_heads
        self.inter_multi = inter_multi
        self.qkv_bias = qkv_bias
        self.attn_qk_norm = attn_qk_norm
        self.lact_chunk_size = lact_chunk_size
        self.use_muon = use_muon
        self.lr_dim = lr_dim
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.window_size = window_size
        self.lr_parameterization = lr_parameterization
        self.learnable_ttt_scale = learnable_ttt_scale
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.last_layer_fuse_norm = last_layer_fuse_norm  # seems that you need to set this to False to use activation checkpointing for every layer.
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        self.use_momentum = use_momentum
        self.ttt_loss_type = ttt_loss_type
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states

        # The important ablation factors
        self.w_reg_lrs = w_reg_lrs
        self.weight_norm = weight_norm
        self.clip_grad_norm = clip_grad_norm
        self.track_states = track_states

        # The rest of the ablation factors used during exploration
        self.w_reg_mode = w_reg_mode
        self.ablation = ablation
        self.fwd_mode = fwd_mode
        self.update_then_apply = update_then_apply
        self.normalize = normalize
        self.scale = scale
        self.mean_reg_lr = mean_reg_lr
        self.no_reg_lr = no_reg_lr
        self.rand_w1 = rand_w1


        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
