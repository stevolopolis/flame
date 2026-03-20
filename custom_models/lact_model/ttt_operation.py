import torch.nn.functional as F
import torch
from typing import Union


COMPILE_MODE = "default"

@torch.compile(mode=COMPILE_MODE)
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile(mode=COMPILE_MODE)
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


@torch.compile(mode=COMPILE_MODE)
def l2_norm_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Backprop for y = x / (||x|| + eps), returning J^T @ dy.
    Args:
        dy: [b, l, d], gradient of the outer loss wrt the y
        x: [b, l, d], input of the l2 norm
    outs:
        dx: [b, l, d], gradient of the outer loss wrt the x
    """
    x_type = x.dtype
    x_f = x.float()
    dy_f = dy.float()

    x_norm = x_f.norm(dim=-1, keepdim=True)
    denom = x_norm + 1e-5
    proj = (dy_f * x_f).sum(dim=-1, keepdim=True)
    dx = dy_f / denom - x_f * (proj / (x_norm * (denom ** 2) + 1e-5))
    return dx.type(x_type)


@torch.compile(mode=COMPILE_MODE)
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile(mode=COMPILE_MODE)
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
    ttt_loss_type: str = "dot_product",
    w_inits: list[torch.Tensor] = None,
    w_reg_lrs: list[Union[torch.Tensor, float]] = None,
    linearize: bool = False,
    w_reg_mode: str = None,  # "init" or "zero" or None (default)
    remove_norm: bool = False,
    return_states: bool = False,
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """
    print(w_reg_mode, linearize, remove_norm)

    assert ttt_loss_type in ["dot_product", "delta_rule"], f"Unsupported ttt_loss_type: {ttt_loss_type}"
    assert w_reg_mode == None or w_reg_lrs is not None, "w_reg_lrs must be provided if w_reg_mode is not None"
    assert w_reg_mode != "init" or w_inits is not None, "w_inits must be provided if w_reg_mode is 'init'"
    assert linearize == False or w_inits is not None, "w_inits must be provided if linearize is True"

    # Extract the initial weights from the list
    if w_inits is not None:
        w0_init = w_inits[0]
        w1_init = w_inits[1]
        w2_init = w_inits[2]

    # Extract the regularization scalars from the list
    if w_reg_lrs is not None:
        w0_reg_lr = w_reg_lrs[0]
        w1_reg_lr = w_reg_lrs[1]
        w2_reg_lr = w_reg_lrs[2]

    # Initialize arrays for storing the states of the fast weights
    w0_norms = []
    w1_norms = []
    w2_norms = []
    w0_dists = []
    w1_dists = []
    w2_dists = []

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # Get the regularization scalars for the current chunk
        # If the regularization scalar is float, we don't have to slice it.
        if w_reg_lrs is not None:
            w0_reg_lr_i = w0_reg_lr[:, s_index:e_index, :].sum(dim=1, keepdim=True) if isinstance(w0_reg_lr, torch.Tensor) else w0_reg_lr
            w1_reg_lr_i = w1_reg_lr[:, s_index:e_index, :].sum(dim=1, keepdim=True) if isinstance(w1_reg_lr, torch.Tensor) else w1_reg_lr
            w2_reg_lr_i = w2_reg_lr[:, s_index:e_index, :].sum(dim=1, keepdim=True) if isinstance(w2_reg_lr, torch.Tensor) else w2_reg_lr

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0_init if linearize else w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2_init if linearize else w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        if ttt_loss_type == "delta_rule":
            pred_vi = torch.bmm(w1_init if linearize else w1, hidden)
            update_signal = vi - pred_vi
        else:
            update_signal = vi

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1_init.transpose(1, 2) if linearize else w1.transpose(1, 2), update_signal)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(
            update_signal, (hidden.transpose(1, 2) * lr1i).type_as(update_signal)
        )  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        # W init regularization
        # if w0_init is not None:
        #     dw0 = dw0 - (w0 - w0_init) * w0_reg if w0_reg_lr is None else dw0 - (w0 - w0_init) * w0_reg_lr_i
        # if w1_init is not None:
        #     dw1 = dw1 - (w1 - w1_init) * w1_reg if w1_reg_lr is None else dw1 - (w1 - w1_init) * w1_reg_lr_i
        # if w2_init is not None:
        #     dw2 = dw2 - (w2 - w2_init) * w2_reg if w2_reg_lr is None else dw2 - (w2 - w2_init) * w2_reg_lr_i

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2

        # w init regularization (post update)
        if w_reg_mode == "init":  # pull towards initial weights
            w0 = w0 - (w0 - w0_init) * w0_reg_lr_i
            w1 = w1 - (w1 - w1_init) * w1_reg_lr_i
            w2 = w2 - (w2 - w2_init) * w2_reg_lr_i
        elif w_reg_mode == "zero":  # pull towards zero
            w0 = w0 - w0 * w0_reg_lr_i
            w1 = w1 - w1 * w1_reg_lr_i
            w2 = w2 - w2 * w2_reg_lr_i

        # Do channel-wise l2 norm.  conceptually like post-norm.
        if not remove_norm:
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        if return_states:
            # Track some states of the fast weights
            # 1. norm
            w0_norms.append(w0.norm(dim=(1, 2)).detach())
            w1_norms.append(w1.norm(dim=(1, 2)).detach())
            w2_norms.append(w2.norm(dim=(1, 2)).detach())

            # 2. distance from initial weights
            w0_dists.append((w0 - w0_init).norm(dim=(1, 2)).detach())
            w1_dists.append((w1 - w1_init).norm(dim=(1, 2)).detach())
            w2_dists.append((w2 - w2_init).norm(dim=(1, 2)).detach())

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2), w0_norms, w1_norms, w2_norms, w0_dists, w1_dists, w2_dists


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_lact_swiglu2(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
    w_reg_lrs: list[Union[torch.Tensor, float]] = None,
    clip_grad_norm: float = 1.0,
    return_states: bool = False,
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """
    w0_reg_lr = w_reg_lrs[0]
    w1_reg_lr = w_reg_lrs[1]
    w2_reg_lr = w_reg_lrs[2]

    # Store initial weights for potential regularization
    w0_init = w0.clone()
    w1_init = w1.clone()
    w2_init = w2.clone()

    # Initialize arrays for storing the states of the fast weights
    w0_norms = []
    w1_norms = []
    w2_norms = []
    w0_dists = []
    w1_dists = []
    w2_dists = []

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # get the regularization scalars for the current chunk  
        w0_reg_lr_i = w0_reg_lr[:, s_index:e_index, :].mean(dim=1, keepdim=True)
        w1_reg_lr_i = w1_reg_lr[:, s_index:e_index, :].mean(dim=1, keepdim=True)
        w2_reg_lr_i = w2_reg_lr[:, s_index:e_index, :].mean(dim=1, keepdim=True)

        # w init regularization (pre update)
        w1 = w1_init + w1_reg_lr_i * (w1 - w1_init)     
        w0 = w0_init + w0_reg_lr_i * (w0 - w0_init)
        w2 = w2_init + w2_reg_lr_i * (w2 - w2_init)

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        pred_vi = torch.bmm(w1, hidden)
        update_signal = vi - pred_vi

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), update_signal)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(update_signal, (hidden.transpose(1, 2) * lr1i).type_as(update_signal))  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w0 = w0 + dw0 if clip_grad_norm is None else w0 + dw0 * torch.clamp(clip_grad_norm / (dw0.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)
        w1 = w1 + dw1 if clip_grad_norm is None else w1 + dw1 * torch.clamp(clip_grad_norm / (dw1.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)
        w2 = w2 + dw2 if clip_grad_norm is None else w2 + dw2 * torch.clamp(clip_grad_norm / (dw2.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)

        # # Do channel-wise l2 norm.  conceptually like post-norm.
        # w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        # w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        # w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        if return_states:   
            # Track some states of the fast weights
            # 1. norm
            w0_norms.append(w0.norm(dim=(1, 2)).detach())
            w1_norms.append(w1.norm(dim=(1, 2)).detach())
            w2_norms.append(w2.norm(dim=(1, 2)).detach())

            # 2. distance from initial weights
            w0_dists.append((w0 - w0_init).norm(dim=(1, 2)).detach())
            w1_dists.append((w1 - w1_init).norm(dim=(1, 2)).detach())
            w2_dists.append((w2 - w2_init).norm(dim=(1, 2)).detach())

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2), w0_norms, w1_norms, w2_norms, w0_dists, w1_dists, w2_dists


@torch.compile(mode=COMPILE_MODE)
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
    ttt_loss_type: str = "dot_product",
    w_inits: list[torch.Tensor] = None,
    w_reg_lrs: list[Union[torch.Tensor, float]] = None,
    w_reg_mode: str = None,  # "init" or "zero" or None (default)
    linearize: bool = False,
    remove_norm: bool = False,
    return_states: bool = False,
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """
    assert ttt_loss_type in ["dot_product", "delta_rule"], f"Unsupported ttt_loss_type: {ttt_loss_type}"
    assert w_reg_mode == None or w_reg_lrs is not None, "w_reg_lrs must be provided if w_reg_mode is not None"
    assert linearize == False or w_inits is not None, "w_inits must be provided if linearize is True"

    # Extract the initial weights from the list
    if w_inits is not None:
        w0_init = w_inits[0]
        w1_init = w_inits[1]
        w2_init = w_inits[2]

    # Extract the regularization scalars from the list
    if w_reg_lrs is not None:
        w0_reg_lr = w_reg_lrs[0]
        w1_reg_lr = w_reg_lrs[1]
        w2_reg_lr = w_reg_lrs[2]

    # Initialize arrays for storing the states of the fast weights
    w0_norms = []
    w1_norms = []
    w2_norms = []
    w0_dists = []
    w1_dists = []
    w2_dists = []

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # Get the regularization scalars for the current chunk
        # If the regularization scalar is float, we don't have to slice it.
        if w_reg_lrs is not None:
            w0_reg_lr_i = w0_reg_lr[:, s_index:e_index, :] if isinstance(w0_reg_lr, torch.Tensor) else w0_reg_lr
            w1_reg_lr_i = w1_reg_lr[:, s_index:e_index, :] if isinstance(w1_reg_lr, torch.Tensor) else w1_reg_lr
            w2_reg_lr_i = w2_reg_lr[:, s_index:e_index, :] if isinstance(w2_reg_lr, torch.Tensor) else w2_reg_lr

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        if ttt_loss_type == "delta_rule":
            pred_vi = torch.bmm(w1, hidden)
            update_signal = vi - pred_vi
        else:
            update_signal = vi

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), update_signal)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(
            update_signal, (hidden.transpose(1, 2) * lr1i).type_as(update_signal)
        )  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        # W init regularization
        if w_reg_mode == "init":  # pull towards initial weights
            dw0 = dw0 - (w0 - w0_init) * w0_reg_lr_i
            dw1 = dw1 - (w1 - w1_init) * w1_reg_lr_i
            dw2 = dw2 - (w2 - w2_init) * w2_reg_lr_i
        elif w_reg_mode == "zero":  # pull towards zero
            dw0 = dw0 - w0 * w0_reg_lr_i
            dw1 = dw1 - w1 * w1_reg_lr_i
            dw2 = dw2 - w2 * w2_reg_lr_i
            
        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2

        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        if return_states:   
            # Track some states of the fast weights
            # 1. norm
            w0_norms.append(w0.norm(dim=(1, 2)).detach())
            w1_norms.append(w1.norm(dim=(1, 2)).detach())
            w2_norms.append(w2.norm(dim=(1, 2)).detach())

            # 2. distance from initial weights
            w0_dists.append((w0 - w0_init).norm(dim=(1, 2)).detach())
            w1_dists.append((w1 - w1_init).norm(dim=(1, 2)).detach())
            w2_dists.append((w2 - w2_init).norm(dim=(1, 2)).detach())

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2), w0_norms, w1_norms, w2_norms, w0_dists, w1_dists, w2_dists


@torch.compile(mode=COMPILE_MODE)
# @torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_lact_swiglu_ablate(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    ttt_loss_type: str = "dot_product",
    w_reg_lrs: list[Union[torch.Tensor, float]] = None,
    fwd_mode: str = "gdn",  # "gdn" or "factorized_gdn" or "lact"
    update_then_apply: bool = False,
    normalize: bool = False,
    scale: bool = False,
    weight_norm: bool = False,
    mean_reg_lr: bool = False,
    no_reg_lr: bool = False,
    clip_grad_norm: float = None,
    rand_w1: bool = False,
    return_states: bool = False,
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """
    assert ttt_loss_type in ["delta_rule"], f"Unsupported ttt_loss_type: {ttt_loss_type}"
    assert chunk_size == 1 or update_then_apply == False, "chunk_size > 1 and update_then_apply = True are not supported"
    assert w_reg_lrs is not None, "w_reg_lrs must be provided"

    # Extract the regularization scalars from the list
    if w_reg_lrs is not None:
        w0_reg_lr = w_reg_lrs[0]
        w1_reg_lr = w_reg_lrs[1]
        w2_reg_lr = w_reg_lrs[2]

    # Store initial weights for potential regularization
    w0_init = w0.clone()
    w1_init = w1.clone()
    w2_init = w2.clone()

    # Initialize the norms of the fast weights
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    # Initialize arrays for storing the states of the fast weights
    w0_norms = []
    w1_norms = []
    w2_norms = []
    w0_dists = []
    w1_dists = []
    w2_dists = []

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size if not update_then_apply else seq_len, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # Get the regularization scalars for the current chunk
        # If the regularization scalar is float, we don't have to slice it.
        if w_reg_lrs is not None:
            if mean_reg_lr:
                w0_reg_lr_i = w0_reg_lr[:, s_index:e_index, :].mean(dim=1, keepdim=True) if isinstance(w0_reg_lr, torch.Tensor) else w0_reg_lr
                w1_reg_lr_i = w1_reg_lr[:, s_index:e_index, :].mean(dim=1, keepdim=True) if isinstance(w1_reg_lr, torch.Tensor) else w1_reg_lr
                w2_reg_lr_i = w2_reg_lr[:, s_index:e_index, :].mean(dim=1, keepdim=True) if isinstance(w2_reg_lr, torch.Tensor) else w2_reg_lr
            else:
                w0_reg_lr_i = w0_reg_lr[:, s_index:e_index, :].sum(dim=1, keepdim=True) if isinstance(w0_reg_lr, torch.Tensor) else w0_reg_lr
                w1_reg_lr_i = w1_reg_lr[:, s_index:e_index, :].sum(dim=1, keepdim=True) if isinstance(w1_reg_lr, torch.Tensor) else w1_reg_lr
                w2_reg_lr_i = w2_reg_lr[:, s_index:e_index, :].sum(dim=1, keepdim=True) if isinstance(w2_reg_lr, torch.Tensor) else w2_reg_lr

        if not no_reg_lr:
            if rand_w1 and fwd_mode == "lact+":
                w1 = w1_init + w1_reg_lr_i * (w1 - w1_init)
            else:
                w1 = w1 * w1_reg_lr_i
            if fwd_mode == "factorized_gdn":
                w2 = w2 * w2_reg_lr_i
                pass
            elif fwd_mode == "factorized_gdn+":
                w2 = w2_init + w2_reg_lr_i * (w2 - w2_init)  # equivalent to w2 = w2 * w2_reg_lr_i + (1-w2_reg_lr_i) * w2_init
            elif fwd_mode == "lact":
                w0 = w0 * w0_reg_lr_i
                w2 = w2 * w2_reg_lr_i
            elif fwd_mode == "lact+":
                w0 = w0_init + w0_reg_lr_i * (w0 - w0_init)
                w2 = w2_init + w2_reg_lr_i * (w2 - w2_init)

        def apply(w0, w1, w2, qi, s_index, e_index):
            # use previous w0 and w1 to get the final output
            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            if fwd_mode == "gdn":
                z = qi
            elif fwd_mode == "gdn+" or fwd_mode == "lact" or fwd_mode == "lact+":
                h = torch.bmm(w2, qi)
                gate = F.silu(torch.bmm(w0, qi), inplace=True)
                z = gate * h
            elif fwd_mode == "factorized_gdn" or fwd_mode == "factorized_gdn+":
                z = torch.bmm(w2, qi)
            else:
                raise ValueError(f"Unsupported fwd_mode: {fwd_mode}")

            if normalize:
                z = l2_norm(z.transpose(1, 2)).transpose(1, 2)
            if scale: 
                z = z * z.shape[-2] ** -0.5

            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
            output[:, :, s_index:e_index] = torch.bmm(w1, z)

        if not update_then_apply:
            apply(w0, w1, w2, qi, s_index, e_index)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        if fwd_mode == "gdn":
            hidden = ki.transpose(1, 2)
        elif fwd_mode == "gdn+" or fwd_mode == "lact" or fwd_mode == "lact+": 
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        elif fwd_mode == "factorized_gdn" or fwd_mode == "factorized_gdn+":
            hidden = hidden_before_mul
        else:
            raise ValueError(f"Unsupported fwd_mode: {fwd_mode}")

        hidden_before_transform = hidden
        if normalize:
            hidden = l2_norm(hidden.transpose(1, 2)).transpose(1, 2)
        if scale:
            hidden = hidden * hidden.shape[-2] ** -0.5

        pred_vi = torch.bmm(w1, hidden)
        update_signal = vi - pred_vi

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), update_signal)
        if scale:
            dhidden = dhidden * hidden.shape[-2] ** -0.5
        if normalize:
            dhidden = l2_norm_backprop(
                dy=dhidden.transpose(1, 2),
                x=hidden_before_transform.transpose(1, 2),
            ).transpose(1, 2)

        if fwd_mode == "lact" or fwd_mode == "lact+":
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)
        else:
            dhidden_before_mul = None
            dgate_before_act = None

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(
            update_signal, (hidden.transpose(1, 2) * lr1i).type_as(update_signal)
        )  # [b, d, d]

        if fwd_mode == "gdn" or fwd_mode == "gdn+":
            dw0 = None
            dw2 = None
        elif fwd_mode == "factorized_gdn" or fwd_mode == "factorized_gdn+":
            dw0 = None
            dw2 = torch.bmm(dhidden, (ki * lr2i).type_as(dhidden))
        elif fwd_mode == "lact" or fwd_mode == "lact+":
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if fwd_mode == "gdn" or fwd_mode == "gdn+": 
            w1 = w1 + dw1 if clip_grad_norm is None else w1 + dw1 * torch.clamp(clip_grad_norm / (dw1.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)
        elif fwd_mode == "factorized_gdn" or fwd_mode == "factorized_gdn+":  
            w1 = w1 + dw1 if clip_grad_norm is None else w1 + dw1 * torch.clamp(clip_grad_norm / (dw1.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)
            w2 = w2 + dw2 if clip_grad_norm is None else w2 + dw2 * torch.clamp(clip_grad_norm / (dw2.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)
        elif fwd_mode == "lact" or fwd_mode == "lact+":
            w0 = w0 + dw0 if clip_grad_norm is None else w0 + dw0 * torch.clamp(clip_grad_norm / (dw0.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)
            w1 = w1 + dw1 if clip_grad_norm is None else w1 + dw1 * torch.clamp(clip_grad_norm / (dw1.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)
            w2 = w2 + dw2 if clip_grad_norm is None else w2 + dw2 * torch.clamp(clip_grad_norm / (dw2.norm(dim=(1, 2), keepdim=True) + 1e-5), max=1.0)
        else:
            raise ValueError(f"Unsupported fwd_mode: {fwd_mode}")

        # Do channel-wise l2 norm.  conceptually like post-norm.
        if weight_norm:
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        if update_then_apply:
            apply(w0, w1, w2, qi, s_index, e_index)
        
        if return_states:   
            # Track some states of the fast weights
            # 1. norm
            w0_norms.append(w0.norm(dim=(1, 2)).detach())
            w1_norms.append(w1.norm(dim=(1, 2)).detach())
            w2_norms.append(w2.norm(dim=(1, 2)).detach())

            # 2. distance from initial weights
            w0_dists.append((w0 - w0_init).norm(dim=(1, 2)).detach())
            w1_dists.append((w1 - w1_init).norm(dim=(1, 2)).detach())
            w2_dists.append((w2 - w2_init).norm(dim=(1, 2)).detach())

    if not update_then_apply:
        # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
        s_index = e_index
        e_index = seq_len

        qi = q[:, :, s_index:e_index]
        apply(w0, w1, w2, qi, s_index, e_index)

    return output.transpose(1, 2), w0_norms, w1_norms, w2_norms, w0_dists, w1_dists, w2_dists


def _test_l2_norm_backprop_matches_autograd():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 7, dtype=torch.float64, requires_grad=True)
    dy = torch.randn_like(x)

    y = l2_norm(x)
    loss = (y * dy).sum()
    (dx_autograd,) = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)
    dx_manual = l2_norm_backprop(dy=dy.detach(), x=x.detach()).to(dx_autograd.dtype)

    max_abs_err = (dx_manual - dx_autograd).abs().max().item()
    assert torch.allclose(dx_manual, dx_autograd, atol=1e-10, rtol=1e-7), (
        f"l2_norm_backprop mismatch with autograd. max_abs_err={max_abs_err:.3e}"
    )
    print(f"[PASS] l2_norm_backprop matches autograd (max_abs_err={max_abs_err:.3e})")


if __name__ == "__main__":
    _test_l2_norm_backprop_matches_autograd()
    
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from tqdm import tqdm
    # from fla.ops.gated_delta_rule import chunk_gated_delta_rule, naive_recurrent_gated_delta_rule
    
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    # REG = 0.1
    # LINEARIZE = True
    # REMOVE_NORM = True

    # options = [(reg, linearize, remove_norm) for reg in [0.1, 0.01, 0.001] for linearize in [True, False] for remove_norm in [True, False]]

    # b = 2
    # h = 8
    # d = 64
    # dh = 64
    # l = 128    
    # device = "cuda"

    # torch.manual_seed(42)

    # w0 = torch.randn(b * h, dh, d, device=device, dtype=torch.float32)
    # w1 = torch.zeros(b * h, d, dh, device=device, dtype=torch.float32)
    # w2 = torch.randn(b * h, dh, d, device=device, dtype=torch.float32)
    # q = l2_norm(torch.randn(b * h, l, d, device=device, dtype=torch.float32))
    # k = l2_norm(torch.randn(b * h, l, d, device=device, dtype=torch.float32))
    # v = l2_norm(torch.randn(b * h, l, d, device=device, dtype=torch.float32))
    # beta0 = torch.rand(b * h, l, 1, device=device, dtype=torch.float32)
    # beta1 = torch.rand(b * h, l, 1, device=device, dtype=torch.float32)
    # beta2 = torch.rand(b * h, l, 1, device=device, dtype=torch.float32)
    # g0 = torch.rand(b * h, l, 1, device=device, dtype=torch.float32)
    # g1 = torch.rand(b * h, l, 1, device=device, dtype=torch.float32)
    # g2 = torch.rand(b * h, l, 1, device=device, dtype=torch.float32)

    # fwd_mode = "gdn+"

    # print("Running LaCT...")
    # o_lact, w0_norms, w1_norms, w2_norms, w0_dists, w1_dists, w2_dists = block_causal_lact_swiglu_ablate(
    #     w0,
    #     w1,
    #     w2,
    #     q,
    #     k,
    #     v,
    #     beta0,
    #     beta1,
    #     beta2,
    #     chunk_size=1,
    #     ttt_loss_type="delta_rule",
    #     w_reg_lrs=[g0, torch.exp(torch.log(g1)), g2],
    #     fwd_mode=fwd_mode,
    #     normalize=True, 
    #     scale=False,
    #     update_then_apply=True,
    # )
    # o_lact = o_lact.reshape(b, h, l, dh).transpose(1, 2)
    # # o_lact = o_lact.reshape(b, l, h, dh)

    # print("Running GDN...")

    # if fwd_mode == "gdn+":
    #     q_gdn = F.silu(torch.bmm(w0, q.transpose(1, 2))) * torch.bmm(w2, q.transpose(1, 2))  # [b, k, l]
    #     k_gdn = F.silu(torch.bmm(w0, k.transpose(1, 2))) * torch.bmm(w2, k.transpose(1, 2))  # [b, k, l]
    # elif fwd_mode == "gdn":
    #     q_gdn = q.transpose(1, 2)
    #     k_gdn = k.transpose(1, 2)
        
    # o_gdn, final_state = chunk_gated_delta_rule(
    #     q_gdn.transpose(1, 2).reshape(b, h, l, dh).transpose(1, 2),  # [b, l, h, k]
    #     k_gdn.transpose(1, 2).reshape(b, h, l, dh).transpose(1, 2),  # [b, l, h, k]
    #     v.reshape(b, h, l, dh).transpose(1, 2),  # [b, l, h, v]
    #     torch.log(g1.squeeze(-1).reshape(b, h, l).transpose(1, 2)),  # [b, l, h]
    #     beta1.squeeze(-1).reshape(b, h, l).transpose(1, 2),  # [b, l, h]
    #     scale=1.0,
    #     use_qk_l2norm_in_kernel=True
    # )

    # print(torch.allclose(o_lact[:, 0, :, :], o_gdn[:, 0, :, :]))
    # print(torch.mean(torch.abs(o_lact[:, 0, :, :] - o_gdn[:, 0, :, :])))
    # print(torch.allclose(o_lact, o_gdn))
    # print(torch.mean(torch.abs(o_lact - o_gdn)))


    # for REG, LINEARIZE, REMOVE_NORM in tqdm(options):
    #     output, w0_norms, w1_norms, w2_norms, w0_dists, w1_dists, w2_dists = block_causal_lact_swiglu(
    #         w0, 
    #         w1, 
    #         w2, 
    #         q, 
    #         k, 
    #         v, 
    #         lr0, 
    #         lr1, 
    #         lr2,
    #         1,
    #         use_muon=False,
    #         momentum=None,
    #         ttt_loss_type="dot_product",
    #         w_inits=[w0, w1, w2],
    #         w_reg_lrs=[REG, REG, REG],
    #         linearize=LINEARIZE,
    #         w_reg_mode="init",
    #         remove_norm=REMOVE_NORM,
    #         return_states=True,
    #     )

    #     w0_norms = torch.stack(w0_norms, dim=1)
    #     w1_norms = torch.stack(w1_norms, dim=1)
    #     w2_norms = torch.stack(w2_norms, dim=1)
    #     w0_dists = torch.stack(w0_dists, dim=1)
    #     w1_dists = torch.stack(w1_dists, dim=1)
    #     w2_dists = torch.stack(w2_dists, dim=1)

    #     plt.plot(w0_norms[0], label="w0_norms")
    #     plt.plot(w1_norms[0], label="w1_norms")
    #     plt.plot(w2_norms[0], label="w2_norms")
    #     plt.legend()
    #     plt.savefig(f"w_norms_{REG}_{"linear" if LINEARIZE else "nonlinear"}_{"nonorm" if REMOVE_NORM else "postnorm"}.png")
    #     plt.close()
    #     plt.plot(w0_dists[0], label="w0_dists")
    #     plt.plot(w1_dists[0], label="w1_dists")
    #     plt.plot(w2_dists[0], label="w2_dists")
    #     plt.legend()
    #     plt.savefig(f"w_dists_{REG}_{"linear" if LINEARIZE else "nonlinear"}_{"nonorm" if REMOVE_NORM else "postnorm"}.png")
    #     plt.close()