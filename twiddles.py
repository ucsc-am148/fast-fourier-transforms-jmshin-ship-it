"""
twiddles.py — host-side twiddle / DFT-matrix helpers.

Convention: forward FFT, exp(-2*pi*i * ...).
All helpers return (re, im) tuples of separate real-valued float32 tensors
(or float16 where noted).  No complex dtype is used anywhere.
"""

import math
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _roots(N: int, numerator, dtype=torch.float32, device="cuda"):
    """
    Return (re, im) for exp(-2*pi*i * numerator / N).
    numerator is a 1-D integer tensor or Python int.
    """
    if isinstance(numerator, int):
        angle = -2.0 * math.pi * numerator / N
        return (torch.tensor(math.cos(angle), dtype=dtype, device=device),
                torch.tensor(math.sin(angle), dtype=dtype, device=device))
    angle = -2.0 * math.pi * numerator.to(dtype=torch.float64) / N
    re = torch.cos(angle).to(dtype=dtype).to(device=device)
    im = torch.sin(angle).to(dtype=dtype).to(device=device)
    return re, im


# ---------------------------------------------------------------------------
# Scaffolding (used internally and by F1 / F6 / F7)
# ---------------------------------------------------------------------------

def make_dft_matrix(N: int, device="cuda", dtype=torch.float32):
    """
    Return (re, im) each (N, N) float32.
    W[j, k] = exp(-2*pi*i * j*k / N).
    """
    j = torch.arange(N, device=device)          # (N,)
    k = torch.arange(N, device=device)          # (N,)
    jk = j.unsqueeze(1) * k.unsqueeze(0)        # (N, N)  outer product
    re, im = _roots(N, jk, dtype=dtype, device=device)
    return re, im


def make_dft_R_padded(R: int, device="cuda", dtype=torch.float32):
    """
    Return (re, im) each (16, 16) float32.
    Top-left (R, R) block holds the DFT matrix for length R;
    the rest is identity-padded so a (16, 16) tl.dot still works.

    R must be in {2, 4, 8, 16}.
    """
    assert R in (2, 4, 8, 16), f"make_dft_R_padded: R={R} not in {{2,4,8,16}}"
    re_full = torch.eye(16, dtype=dtype, device=device)
    im_full = torch.zeros(16, 16, dtype=dtype, device=device)
    re_sub, im_sub = make_dft_matrix(R, device=device, dtype=dtype)   # (R, R)
    re_full[:R, :R] = re_sub
    im_full[:R, :R] = im_sub
    return re_full, im_full


def bit_reversal_perm(N: int, device="cuda", dtype=torch.int32):
    """
    Return int32 tensor of shape (N,) with bit-reversal permutation for N = 2^L.
    rev[i] is the integer whose L-bit binary rep is i's bits reversed.
    """
    L = int(math.log2(N))
    assert 1 << L == N, "N must be a power of 2"
    idx = torch.arange(N, dtype=torch.int32, device=device)
    rev = torch.zeros(N, dtype=torch.int32, device=device)
    for _ in range(L):
        rev = (rev << 1) | (idx & 1)
        idx = idx >> 1
    return rev


# ---------------------------------------------------------------------------
# Pattern 1 — radix-2 twiddle table  (F2, F3)
# ---------------------------------------------------------------------------

def make_radix2_twiddles(N: int, device="cuda", dtype=torch.float32):
    """
    Return (re, im) each (N//2,) float32.
    Entry t holds w_N^t = exp(-2*pi*i*t / N) for t in [0, N//2).

    Usage in a butterfly stage s:
        twiddle index = (j & ((1 << s) - 1)) * (N >> (s + 1))
    which ranges over [0, N//2), so the full table covers every stage.
    """
    t = torch.arange(N // 2, device=device)
    return _roots(N, t, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Pattern 2 — radix-16 per-stage twiddle table  (F4, F5, F6, F7)
# ---------------------------------------------------------------------------

def _column_axis_labeling(L: int):
    """
    Returns a list of length L.  Entry s is a list of the digit labels
    that appear at column-axis positions 1..L-1 at the start of stage s,
    ordered by their axis position (position 1 first).

    These are the *output* digit labels e_{L-1-j} for j in 0..s-1.
    At stage s, axes 0..s-1 already hold output digits; axis 0 is
    being transformed this step, axes 1..s-1 hold already-transformed
    output digits.  After the stage-s permute, the order is:
        axis 0: d_s   (current input digit, about to be transformed)
        axis 1..s-1: e_{L-1}, e_{L-2}, ..., e_{L-s+1}   (already done)
        axis s..L-1: d_{s+1}, ..., d_{L-1}               (not yet done)
    So "column" positions 1..s-1 carry output digit labels
    e_{L-1-(0)} = e_{L-1}, e_{L-1-(1)} = e_{L-2}, ..., up to e_{L-s+1}.
    """
    col_labels = []
    for s in range(L):
        # output digits already placed at axes 1..s-1 after the permute
        labels = [L - 1 - j for j in range(s)]  # e_{L-1}, ..., e_{L-s+1}
        col_labels.append(labels)
    return col_labels


def make_radix16_twiddles(N: int, device="cuda", dtype=torch.float16):
    """
    Return (re, im) each (L, 16, N//16) float16,
    where L = log16(N).

    Entry [s, m, c] = exp(-2*pi*i * m * t(s,c) / 16^(s+1))

    where t(s, c) is reconstructed from the already-transformed output
    digits whose column-axis labels appear at axis positions 1..s-1 in
    the stage-s tile layout:

        t = sum_{j=0}^{s-1}  digit_e_{L-1-j}(c)  *  16^j

    digit_e_{L-1-j}(c) is the (L-1-j)-th base-16 digit of c
    when c is viewed as an index into the sub-table of size 16^s.

    At s=0 there are no preceding digits, so t=0 for all c → twiddle = 1.
    """
    L = int(round(math.log(N, 16)))
    assert 16 ** L == N, f"N must be a power of 16 (got N={N})"
    M = N // 16   # = 16^(L-1)

    re_all = torch.ones(L, 16, M, dtype=torch.float32, device=device)
    im_all = torch.zeros(L, 16, M, dtype=torch.float32, device=device)

    c = torch.arange(M, device=device)  # column indices 0..M-1

    for s in range(1, L):     # s=0 → twiddle = 1, skip
        # Reconstruct t from the s already-processed output digits.
        # After the stage-s permute, the column sub-table has size 16^s.
        # The digit at position j (0-indexed) corresponds to output digit
        # e_{L-1-j}, which lives in the j-th base-16 digit of c
        # (c ranges over [0, 16^s) for each independent group;
        #  here c ranges over [0, M) = [0, 16^(L-1)) — we use c mod 16^s).
        sub_size = 16 ** s
        c_mod = c % sub_size   # shape (M,)
        t = torch.zeros(M, dtype=torch.int64, device=device)
        for j in range(s):
            # j-th base-16 digit of c_mod
            digit = (c_mod // (16 ** j)) % 16
            t = t + digit * (16 ** j)
        # t has shape (M,); m has shape (16,)
        m = torch.arange(16, device=device)         # (16,)
        # twiddle[m, c] = exp(-2pi*i * m * t[c] / 16^(s+1))
        mt = m.unsqueeze(1) * t.unsqueeze(0)        # (16, M)
        denom = 16 ** (s + 1)
        re_s, im_s = _roots(denom, mt, dtype=torch.float32, device=device)
        re_all[s] = re_s
        im_all[s] = im_s

    return re_all.to(torch.float16), im_all.to(torch.float16)


# ---------------------------------------------------------------------------
# Pattern 3 — Bailey cross-twiddle  (F3, F5, F6, F7)
# ---------------------------------------------------------------------------

def make_bailey_cross_twiddles(m0: int, M: int, N: int = None, device="cuda", dtype=torch.float16):
    """
    Return (re, im) each (m0, M) float16.
    Entry [n1, k2] = exp(-2*pi*i * n1 * k2 / (m0 * M))
    i.e. w_{N}^{n1 * k2}  where N = m0 * M.

    N parameter is accepted for harness compatibility but ignored (always m0*M).
    n1 in [0, m0), k2 in [0, M).
    """
    _ = N  # harness may pass N=m0*M explicitly; we recompute it anyway
    N = m0 * M
    n1 = torch.arange(m0, device=device)    # (m0,)
    k2 = torch.arange(M, device=device)    # (M,)
    nk = n1.unsqueeze(1) * k2.unsqueeze(0) # (m0, M)
    re, im = _roots(N, nk, dtype=torch.float32, device=device)
    return re.to(dtype), im.to(dtype)
