"""
kernels.py — Triton kernels + pipeline drivers for F1..F7.
"""

import math
import torch
import triton
import triton.language as tl

from twiddles import (
    make_dft_matrix,
    make_dft_R_padded,
    make_radix2_twiddles,
    make_radix16_twiddles,
    make_bailey_cross_twiddles,
    bit_reversal_perm,
)


# ===========================================================================
# F1 — Dense DFT as complex matmul
# ===========================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,
    w_re_ptr, w_im_ptr,
    y_re_ptr, y_im_ptr,
    B, N,
    stride_xb, stride_xn,
    stride_wb, stride_wn,
    stride_yb, stride_yn,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    b_mask = b_offs < B
    k_mask = k_offs < N
    acc_re = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        x_re = tl.load(x_re_ptr + b_offs[:, None] * stride_xb + n_offs[None, :] * stride_xn,
                       mask=b_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        x_im = tl.load(x_im_ptr + b_offs[:, None] * stride_xb + n_offs[None, :] * stride_xn,
                       mask=b_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        w_re = tl.load(w_re_ptr + n_offs[:, None] * stride_wb + k_offs[None, :] * stride_wn,
                       mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        w_im = tl.load(w_im_ptr + n_offs[:, None] * stride_wb + k_offs[None, :] * stride_wn,
                       mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc_re += tl.dot(x_re, w_re) - tl.dot(x_im, w_im)
        acc_im += tl.dot(x_re, w_im) + tl.dot(x_im, w_re)
    tl.store(y_re_ptr + b_offs[:, None] * stride_yb + k_offs[None, :] * stride_yn,
             acc_re, mask=b_mask[:, None] & k_mask[None, :])
    tl.store(y_im_ptr + b_offs[:, None] * stride_yb + k_offs[None, :] * stride_yn,
             acc_im, mask=b_mask[:, None] & k_mask[None, :])


def f1_launch(x_re, x_im, w_re, w_im, y_re, y_im):
    B, N = x_re.shape
    BLOCK_B = 16
    BLOCK_K = 16
    BLOCK_N = 16
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_K))
    f1_kernel[grid](
        x_re, x_im, w_re, w_im, y_re, y_im,
        B, N,
        x_re.stride(0), x_re.stride(1),
        w_re.stride(0), w_re.stride(1),
        y_re.stride(0), y_re.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )


# ===========================================================================
# F2 — Stockham FFT (no scratchpad, no bit-reversal, race-free)
# Reads from buffer A, writes to buffer B, ping-pongs each stage.
# ===========================================================================

@triton.jit
def stockham_stage_kernel(
    a_re_ptr, a_im_ptr,
    b_re_ptr, b_im_ptr,
    N, m,          # m = 2^s = half butterfly size at this stage
    BLOCK_N: tl.constexpr,
):
    """
    One Stockham stage. Each program handles one batch element.
    Input in A (stride N per batch), output to B (stride N per batch).
    For each output position n:
      i0 = (n % m) + (n // (2*m)) * m        upper source
      i1 = i0 + N//2                           lower source
      tw = exp(-2pi*i * (n%m) / (2*m*2)) -- wait:
      tw = exp(-2pi*i * (n%m) / (2^(s+1))) where 2^s = m
         = exp(-2pi*i * (n%m) / (2*m))
    """
    pid = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)

    # Source indices in A
    k = n % m                    # position within butterfly (0..m-1)
    j = n // (2 * m)             # which group
    i0 = j * m + k              # upper source
    i1 = i0 + N // 2            # lower source

    u_re = tl.load(a_re_ptr + pid * N + i0, mask=n < N, other=0.0)
    u_im = tl.load(a_im_ptr + pid * N + i0, mask=n < N, other=0.0)
    l_re = tl.load(a_re_ptr + pid * N + i1, mask=n < N, other=0.0)
    l_im = tl.load(a_im_ptr + pid * N + i1, mask=n < N, other=0.0)

    # Twiddle: exp(-2pi*i * k / (2*m))
    angle = -2.0 * 3.141592653589793 * k.to(tl.float32) / (2.0 * m)
    tw_re = tl.cos(angle)
    tw_im = tl.sin(angle)

    tv_re = tw_re * l_re - tw_im * l_im
    tv_im = tw_re * l_im + tw_im * l_re

    # Output index (natural order): upper at n if n%2m < m, lower otherwise
    # Upper goes to: j*(2m) + k, lower goes to j*(2m) + k + m
    is_upper = (n % (2 * m)) < m
    out_re = tl.where(is_upper, u_re + tv_re, u_re - tv_re)
    out_im = tl.where(is_upper, u_im + tv_im, u_im - tv_im)

    tl.store(b_re_ptr + pid * N + n, out_re, mask=n < N)
    tl.store(b_im_ptr + pid * N + n, out_im, mask=n < N)


@triton.jit
def stockham_epilogue_kernel(
    # final buffer (result after L stages)
    a_re_ptr, a_im_ptr,
    # output
    y_re_ptr, y_im_ptr,
    # optional Bailey cross-twiddle
    ct_re_ptr, ct_im_ptr,
    N, N2,
    stride_yb,
    BLOCK_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)
    v_re = tl.load(a_re_ptr + pid * N + n, mask=n < N)
    v_im = tl.load(a_im_ptr + pid * N + n, mask=n < N)

    if BAILEY_EPILOGUE:
        ct_re = tl.load(ct_re_ptr + pid * N2 + n, mask=n < N, other=1.0)
        ct_im = tl.load(ct_im_ptr + pid * N2 + n, mask=n < N, other=0.0)
        tmp_re = v_re * ct_re - v_im * ct_im
        tmp_im = v_re * ct_im + v_im * ct_re
        v_re = tmp_re
        v_im = tmp_im

    if STRIDED_STORE:
        out_idx = (n % N2) * (N // N2) + (n // N2)
        tl.store(y_re_ptr + pid * stride_yb + out_idx, v_re, mask=n < N)
        tl.store(y_im_ptr + pid * stride_yb + out_idx, v_im, mask=n < N)
    else:
        tl.store(y_re_ptr + pid * stride_yb + n, v_re, mask=n < N)
        tl.store(y_im_ptr + pid * stride_yb + n, v_im, mask=n < N)


@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,
    tw_re_ptr, tw_im_ptr,
    brp_ptr,
    y_re_ptr, y_im_ptr,
    ct_re_ptr, ct_im_ptr,
    B, N,
    stride_xb, stride_yb, N2, row_offset,
    BLOCK_N: tl.constexpr,
    LOG2N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Stub for harness compatibility."""
    pid = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)
    re = tl.load(x_re_ptr + (pid + row_offset) * stride_xb + n, mask=n < N)
    im = tl.load(x_im_ptr + (pid + row_offset) * stride_xb + n, mask=n < N)
    tl.store(y_re_ptr + pid * stride_yb + n, re, mask=n < N)
    tl.store(y_im_ptr + pid * stride_yb + n, im, mask=n < N)


def _f2_triton(x_re, x_im, tw_re, tw_im, brp, y_re, y_im,
               ct_re=None, ct_im=None, N2=1, row_offset=0,
               BAILEY_EPILOGUE=False, STRIDED_STORE=False):
    """Stockham FFT: race-free, no scratchpad, no bit-reversal needed."""
    B, N = x_re.shape
    LOG2N = int(math.log2(N))
    assert 1 << LOG2N == N

    # Allocate two ping-pong buffers (flat B*N)
    buf_re = [
        x_re.reshape(-1).contiguous().clone(),
        torch.empty(B * N, dtype=x_re.dtype, device=x_re.device),
    ]
    buf_im = [
        x_im.reshape(-1).contiguous().clone(),
        torch.empty(B * N, dtype=x_im.dtype, device=x_im.device),
    ]

    cur = 0  # which buffer is current input
    m = 1
    for s in range(LOG2N):
        nxt = 1 - cur
        stockham_stage_kernel[(B,)](
            buf_re[cur], buf_im[cur],
            buf_re[nxt], buf_im[nxt],
            N, m, BLOCK_N=N,
        )
        cur = nxt
        m *= 2

    # Epilogue: copy result to y, apply Bailey cross-twiddle if needed
    if ct_re is None:
        ct_re_arg = x_re
        ct_im_arg = x_im
    else:
        ct_re_arg = ct_re
        ct_im_arg = ct_im

    stockham_epilogue_kernel[(B,)](
        buf_re[cur].reshape(B, N),
        buf_im[cur].reshape(B, N),
        y_re, y_im,
        ct_re_arg, ct_im_arg,
        N, int(N2),
        y_re.stride(0),
        BLOCK_N=N, LOG2N=LOG2N,
        BAILEY_EPILOGUE=BAILEY_EPILOGUE,
        STRIDED_STORE=STRIDED_STORE,
    )


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Harness: f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm)"""
    _f2_triton(x_re, x_im, tw_re, tw_im, perm, y_re, y_im)


# ===========================================================================
# Transpose kernel — (B, R, C) → (B, C, R)
# ===========================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,
    y_re_ptr, y_im_ptr,
    B, R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b     = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = r_offs < R
    c_mask = c_offs < C
    src = b * R * C + r_offs[:, None] * C + c_offs[None, :]
    re = tl.load(x_re_ptr + src, mask=r_mask[:, None] & c_mask[None, :], other=0.0)
    im = tl.load(x_im_ptr + src, mask=r_mask[:, None] & c_mask[None, :], other=0.0)
    dst = b * C * R + c_offs[None, :] * R + r_offs[:, None]
    tl.store(y_re_ptr + dst, re, mask=r_mask[:, None] & c_mask[None, :])
    tl.store(y_im_ptr + dst, im, mask=r_mask[:, None] & c_mask[None, :])


def _transpose(x_re, x_im, B, R, C):
    B, R, C = int(B), int(R), int(C)
    xr = x_re.reshape(B, R, C).contiguous()
    xi = x_im.reshape(B, R, C).contiguous()
    yr = torch.empty(B, C, R, dtype=x_re.dtype, device=x_re.device)
    yi = torch.empty(B, C, R, dtype=x_im.dtype, device=x_im.device)
    BLOCK = 32
    grid = (B, triton.cdiv(R, BLOCK), triton.cdiv(C, BLOCK))
    transpose_kernel[grid](xr, xi, yr, yi, B, R, C, BLOCK_R=BLOCK, BLOCK_C=BLOCK)
    return yr.reshape(B, C * R), yi.reshape(B, C * R)


# ===========================================================================
# F3 — Bailey six-step
# ===========================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    N  = plan['N']
    N1 = plan['N1']
    N2 = plan['N2']
    tw_re_n2 = plan['tw_re_n2']
    tw_im_n2 = plan['tw_im_n2']
    tw_re_n1 = plan['tw_re_n1']
    tw_im_n1 = plan['tw_im_n1']
    perm_n2  = plan['perm_n2']
    perm_n1  = plan['perm_n1']
    bt_re    = plan['bt_re'].float()
    bt_im    = plan['bt_im'].float()
    B = int(B)

    t1_re, t1_im = _transpose(in_re, in_im, B, N2, N1)

    a_re = t1_re.reshape(B * N1, N2).contiguous()
    a_im = t1_im.reshape(B * N1, N2).contiguous()
    fa_re = mid_re[:B * N1 * N2].reshape(B * N1, N2)
    fa_im = mid_im[:B * N1 * N2].reshape(B * N1, N2)
    ct_re = bt_re.unsqueeze(0).expand(B, N1, N2).reshape(B * N1, N2).contiguous()
    ct_im = bt_im.unsqueeze(0).expand(B, N1, N2).reshape(B * N1, N2).contiguous()
    _f2_triton(a_re, a_im, tw_re_n2, tw_im_n2, perm_n2, fa_re, fa_im,
               ct_re=ct_re, ct_im=ct_im, N2=N2, BAILEY_EPILOGUE=True)

    t2_re, t2_im = _transpose(fa_re.reshape(B, N), fa_im.reshape(B, N), B, N1, N2)

    b_re = t2_re.reshape(B * N2, N1).contiguous()
    b_im = t2_im.reshape(B * N2, N1).contiguous()
    fb_re = mid_re[:B * N2 * N1].reshape(B * N2, N1)
    fb_im = mid_im[:B * N2 * N1].reshape(B * N2, N1)
    _f2_triton(b_re, b_im, tw_re_n1, tw_im_n1, perm_n1, fb_re, fb_im)

    result_re = fb_re.reshape(B, N2, N1).permute(0, 2, 1).contiguous().reshape(-1)
    result_im = fb_im.reshape(B, N2, N1).permute(0, 2, 1).contiguous().reshape(-1)
    out_re.copy_(result_re)
    out_im.copy_(result_im)


# ===========================================================================
# F4 — tcFFT radix-16, N=256
# ===========================================================================

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,
    y_re_ptr, y_im_ptr,
    dft_re_ptr, dft_im_ptr,
    tw_re_ptr, tw_im_ptr,
    B, stride_b,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    b = tl.program_id(0)
    offs_256 = tl.arange(0, 256)
    r_idx = tl.arange(0, 16)
    c_idx = tl.arange(0, 16)

    x_re = tl.load(x_re_ptr + b * 256 + offs_256).to(tl.float32)
    x_im = tl.load(x_im_ptr + b * 256 + offs_256).to(tl.float32)
    tile_re = tl.reshape(x_re, (16, 16))
    tile_im = tl.reshape(x_im, (16, 16))

    dft_re = tl.load(dft_re_ptr + r_idx[:, None] * 16 + c_idx[None, :]).to(tl.float32)
    dft_im = tl.load(dft_im_ptr + r_idx[:, None] * 16 + c_idx[None, :]).to(tl.float32)

    # Stage 0: A = F @ tile -> [k1, n1]
    a_re = tl.dot(dft_re, tile_re) - tl.dot(dft_im, tile_im)
    a_im = tl.dot(dft_re, tile_im) + tl.dot(dft_im, tile_re)

    if STAGE_STOP == 1:
        tl.store(y_re_ptr + b * 256 + offs_256, tl.reshape(a_re.to(tl.float16), (256,)))
        tl.store(y_im_ptr + b * 256 + offs_256, tl.reshape(a_im.to(tl.float16), (256,)))
    else:
        tw_re = tl.load(tw_re_ptr + 256 + c_idx[:, None] * 16 + r_idx[None, :]).to(tl.float32)
        tw_im = tl.load(tw_im_ptr + 256 + c_idx[:, None] * 16 + r_idx[None, :]).to(tl.float32)
        b_re = a_re * tw_re - a_im * tw_im
        b_im = a_re * tw_im + a_im * tw_re

        c_re = tl.dot(b_re, dft_re) - tl.dot(b_im, dft_im)
        c_im = tl.dot(b_re, dft_im) + tl.dot(b_im, dft_re)

        out_idx = c_idx[None, :] * 16 + r_idx[:, None]
        out_re = c_re.to(tl.float16)
        out_im = c_im.to(tl.float16)

        if STORE_T:
            tl.store(y_re_ptr + b * 256 + offs_256, tl.reshape(out_re, (256,)))
            tl.store(y_im_ptr + b * 256 + offs_256, tl.reshape(out_im, (256,)))
        else:
            tl.store(y_re_ptr + b * 256 + tl.reshape(out_idx, (256,)),
                     tl.reshape(out_re, (256,)))
            tl.store(y_im_ptr + b * 256 + tl.reshape(out_idx, (256,)),
                     tl.reshape(out_im, (256,)))


F4_L2_BLOCK_B = 1


def _run_f4(x_re, x_im, y_re, y_im, plan):
    B = x_re.shape[0]
    f4_kernel_L2[(triton.cdiv(B, F4_L2_BLOCK_B),)](
        x_re, x_im, y_re, y_im,
        plan['F_re'], plan['F_im'],
        plan['tw_re'], plan['tw_im'],
        B, 1,
        BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=plan['L'], STORE_T=False,
        num_warps=4, num_stages=1,
    )


def f4_launch(x_re, x_im, y_re=None, y_im=None, plan=None):
    if plan is None:
        tw_re, tw_im = make_radix16_twiddles(256, device=x_re.device)
        dft_re, dft_im = make_dft_matrix(16, device=x_re.device)
        plan = {'F_re': dft_re.half(), 'F_im': dft_im.half(),
                'tw_re': tw_re, 'tw_im': tw_im, 'L': 2}
    if y_re is None:
        y_re = torch.empty_like(x_re)
        y_im = torch.empty_like(x_im)
    _run_f4(x_re.half(), x_im.half(), y_re, y_im, plan)
    return y_re, y_im


# ===========================================================================
# Bailey scale kernel
# ===========================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,
    ct_re_ptr, ct_im_ptr,
    y_re_ptr, y_im_ptr,
    m0, M,
    STORE_T: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = tl.num_programs(0) * BLOCK
    row = offs // (m0 * M)
    rem = offs % (m0 * M)
    n1  = rem // M
    k2  = rem % M
    mask = offs < total
    x_re = tl.load(x_re_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_im = tl.load(x_im_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ct_re = tl.load(ct_re_ptr + n1 * M + k2, mask=mask, other=1.0).to(tl.float32)
    ct_im = tl.load(ct_im_ptr + n1 * M + k2, mask=mask, other=0.0).to(tl.float32)
    y_re = (x_re * ct_re - x_im * ct_im).to(tl.float16)
    y_im = (x_re * ct_im + x_im * ct_re).to(tl.float16)
    if STORE_T:
        t_offs = row * M * m0 + k2 * m0 + n1
        tl.store(y_re_ptr + t_offs, y_re, mask=mask)
        tl.store(y_im_ptr + t_offs, y_im, mask=mask)
    else:
        tl.store(y_re_ptr + offs, y_re, mask=mask)
        tl.store(y_im_ptr + offs, y_im, mask=mask)


def _scale(x_re, x_im, tw_re, tw_im, m0, M, store_t=False):
    m0, M = int(m0), int(M)
    x_re_flat = x_re.reshape(-1)
    x_im_flat = x_im.reshape(-1)
    total = x_re_flat.numel()
    rows = total // (m0 * M)
    tw_re_flat = tw_re.reshape(-1)
    tw_im_flat = tw_im.reshape(-1)
    BLOCK = 256
    y_re = torch.empty(total, dtype=torch.float16, device=x_re.device)
    y_im = torch.empty_like(y_re)
    grid = (triton.cdiv(total, BLOCK),)
    bailey_scale_kernel[grid](
        x_re_flat, x_im_flat,
        tw_re_flat, tw_im_flat, y_re, y_im,
        m0, M, STORE_T=store_t, BLOCK=BLOCK,
    )
    if store_t:
        return y_re.reshape(rows, M, m0), y_im.reshape(rows, M, m0)
    return y_re.reshape(rows, m0, M), y_im.reshape(rows, m0, M)


# ===========================================================================
# dft_kernel — small padded DFT
# ===========================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,
    dft_re_ptr, dft_im_ptr,
    y_re_ptr, y_im_ptr,
    B, R, M,
    STORE_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    r_offs = tl.arange(0, 16)
    k_offs = tl.arange(0, 16)
    x_re = tl.load(x_re_ptr + pid * R + r_offs, mask=r_offs < R, other=0.0).to(tl.float32)
    x_im = tl.load(x_im_ptr + pid * R + r_offs, mask=r_offs < R, other=0.0).to(tl.float32)
    dft_re = tl.load(dft_re_ptr + r_offs[:, None] * 16 + k_offs[None, :]).to(tl.float32)
    dft_im = tl.load(dft_im_ptr + r_offs[:, None] * 16 + k_offs[None, :]).to(tl.float32)
    xr = tl.reshape(x_re, (1, 16))
    xi = tl.reshape(x_im, (1, 16))
    yr = tl.dot(xr, dft_re) - tl.dot(xi, dft_im)
    yi = tl.dot(xr, dft_im) + tl.dot(xi, dft_re)
    tl.store(y_re_ptr + pid * R + k_offs,
             tl.reshape(yr.to(tl.float16), (16,)), mask=k_offs < R)
    tl.store(y_im_ptr + pid * R + k_offs,
             tl.reshape(yi.to(tl.float16), (16,)), mask=k_offs < R)


def _run_dft(x_re, x_im, R, dft_re, dft_im, store_t=False):
    B = x_re.shape[0]
    y_re = torch.empty_like(x_re)
    y_im = torch.empty_like(x_im)
    dft_kernel[(B,)](x_re, x_im, dft_re, dft_im, y_re, y_im,
                     B, R, 1, STORE_T=store_t, BLOCK_M=1)
    return y_re, y_im


# ===========================================================================
# F5
# ===========================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    N  = plan['N']
    N1 = plan['N1']
    N2 = plan['N2']
    f4 = plan['f4_plan']
    bt_re = plan['bt_re']
    bt_im = plan['bt_im']
    B = int(B)

    t1_re, t1_im = _transpose(in_re, in_im, B, N2, N1)
    a_re = t1_re.reshape(B * N1, N2).contiguous()
    a_im = t1_im.reshape(B * N1, N2).contiguous()
    fa_re = b1_re[:B * N1 * N2].reshape(B * N1, N2)
    fa_im = b1_im[:B * N1 * N2].reshape(B * N1, N2)
    _run_f4(a_re, a_im, fa_re, fa_im, f4)

    sc_re, sc_im = _scale(fa_re, fa_im, bt_re, bt_im, N1, N2)

    t2_re, t2_im = _transpose(sc_re.reshape(B, N), sc_im.reshape(B, N), B, N1, N2)
    b_re = t2_re.reshape(B * N2, N1).contiguous()
    b_im = t2_im.reshape(B * N2, N1).contiguous()
    fb_re = b2_re[:B * N2 * N1].reshape(B * N2, N1)
    fb_im = b2_im[:B * N2 * N1].reshape(B * N2, N1)
    _run_f4(b_re, b_im, fb_re, fb_im, f4)

    t3_re, t3_im = _transpose(fb_re.reshape(B, N), fb_im.reshape(B, N), B, N2, N1)
    b0_re.copy_(t3_re.reshape(B, N))
    b0_im.copy_(t3_im.reshape(B, N))


# ===========================================================================
# f6_factor
# ===========================================================================

def f6_factor(N):
    N = int(N)
    chunks = []
    remaining = N
    while remaining > 1:
        if remaining % 256 == 0:
            chunks.append(256)
            remaining //= 256
        elif remaining % 16 == 0:
            chunks.append(16)
            remaining //= 16
        else:
            assert remaining in (2, 4, 8), f"Cannot factor N={N}"
            chunks.append(remaining)
            remaining = 1
    return chunks


# ===========================================================================
# F6/F7
# ===========================================================================

def _leaf_fft(x_re, x_im, chunk, plan):
    chunk = int(chunk)
    if chunk == 256:
        y_re = torch.empty_like(x_re)
        y_im = torch.empty_like(x_im)
        _run_f4(x_re, x_im, y_re, y_im, plan['f4_plan'])
        return y_re, y_im
    else:
        dft_re, dft_im = plan['dft_mats'][chunk]
        return _run_dft(x_re, x_im, chunk, dft_re, dft_im)


def _f6_rec(in_re, in_im, B, chunks, plan, cyc):
    B = int(B)
    chunks = list(chunks)
    N = 1
    for c in chunks: N *= int(c)

    if len(chunks) == 1:
        m0 = int(chunks[0])
        return _leaf_fft(in_re.reshape(B, m0), in_im.reshape(B, m0), m0, plan)

    m0 = int(chunks[0])
    rest = [int(c) for c in chunks[1:]]
    M = 1
    for c in rest: M *= c
    N_i = m0 * M

    tw_level = len(plan['chunks']) - len(chunks)
    _, _, _, tw_re, tw_im = plan['tw'][tw_level]

    t1_re, t1_im = _transpose(in_re, in_im, B, M, m0)
    rec_re, rec_im = _f6_rec(
        t1_re.reshape(B * m0, M), t1_im.reshape(B * m0, M),
        B * m0, rest, plan, cyc)

    sc_re, sc_im = _scale(
        rec_re.reshape(B * m0, M), rec_im.reshape(B * m0, M),
        tw_re, tw_im, m0, M)

    t2_re, t2_im = _transpose(
        sc_re.reshape(B, N_i), sc_im.reshape(B, N_i), B, m0, M)

    f_re, f_im = _leaf_fft(
        t2_re.reshape(B * M, m0), t2_im.reshape(B * M, m0), m0, plan)

    t3_re, t3_im = _transpose(
        f_re.reshape(B, M * m0), f_im.reshape(B, M * m0), B, M, m0)

    return t3_re.reshape(B * N_i), t3_im.reshape(B * N_i)


def f6_launch(x_re, x_im, plan, B):
    return _f6_rec(x_re, x_im, B, plan['chunks'], plan, None)


def _f7_rec(in_re, in_im, B, chunks, plan, cyc):
    B = int(B)
    chunks = list(chunks)
    N = 1
    for c in chunks: N *= int(c)

    if len(chunks) == 1:
        m0 = int(chunks[0])
        return _leaf_fft(in_re.reshape(B, m0), in_im.reshape(B, m0), m0, plan)

    m0 = int(chunks[0])
    rest = [int(c) for c in chunks[1:]]
    M = 1
    for c in rest: M *= c
    N_i = m0 * M

    tw_level = len(plan['chunks']) - len(chunks)
    _, _, _, tw_re, tw_im = plan['tw'][tw_level]

    t1_re, t1_im = _transpose(in_re, in_im, B, M, m0)
    rec_re, rec_im = _f7_rec(
        t1_re.reshape(B * m0, M), t1_im.reshape(B * m0, M),
        B * m0, rest, plan, cyc)

    # Fused Scale+T2
    t2_re, t2_im = _scale(
        rec_re.reshape(B * m0, M), rec_im.reshape(B * m0, M),
        tw_re, tw_im, m0, M, store_t=True)

    f_re, f_im = _leaf_fft(
        t2_re.reshape(B * M, m0), t2_im.reshape(B * M, m0), m0, plan)

    t3_re, t3_im = _transpose(
        f_re.reshape(B, M * m0), f_im.reshape(B, M * m0), B, M, m0)

    return t3_re.reshape(B * N_i), t3_im.reshape(B * N_i)


def f7_launch(x_re, x_im, plan, B):
    return _f7_rec(x_re, x_im, B, plan['chunks'], plan, None)
