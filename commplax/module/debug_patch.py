import types, functools, numpy as np, jax.numpy as jnp, jax
from flax.core import Scope
from commplax.module import core

def _is_init(s: Scope):
    f = getattr(s, 'is_initializing', None)
    return f() if callable(f) else False

def _next_pow2(n): return 1 << (n-1).bit_length()

def conv1d_fft_debug(scope: Scope, signal: core.Signal, *,
                     taps=261, seglen=None, kernel_init=core.delta,
                     debug=True, max_fftlen=None):
    x, t_in = signal
    h = scope.param('kernel', kernel_init, (taps,), jnp.complex64)
    rtap   = (taps-1)//2
    Nout   = x.shape[0] - 2*rtap
    fftlen = seglen or _next_pow2(x.shape[0] + taps - 1)
    if max_fftlen: fftlen = min(fftlen, max_fftlen)

    Xk = jnp.fft.fft(x, fftlen, axis=0)
    Hk = jnp.fft.fft(h, fftlen);  Hk = Hk[:,None] if x.ndim==2 else Hk
    yf = jnp.fft.ifft(Xk*Hk, fftlen, axis=0)
    yv = yf[rtap: rtap+Nout]
    tout = core.SigTime(t_in.start+rtap, t_in.stop-rtap, t_in.sps)

    if debug and _is_init(scope):
        print(f"[conv1d_fft] N={x.shape[0]} taps={taps} "
              f"rtap={rtap} fftlen={fftlen} out_len={Nout}")

    # -------- nan / inf guard (only when not tracing) ---------------------
    from jax import core as jcore
    if debug and (not _is_init(scope)) and not isinstance(yv, jcore.Tracer):
        y_np = np.asarray(yv)
        if not np.isfinite(y_np).all():
            bad = int(np.where(~np.isfinite(y_np))[0][0])
            raise FloatingPointError(
                f"⚠️ conv1d_fft nan/inf @ out[{bad}] value={y_np[bad]}")

    return core.Signal(yv, tout)

def dconv_pair_debug(scope: Scope, sig: core.Signal, *, taps, kinit, debug=True):
    x, t = sig
    outs = []
    for p in range(x.shape[1]):
        y, tp = scope.child(
            functools.partial(conv1d_fft_debug, taps=taps,
                              kernel_init=kinit, debug=debug),
            name=f"Pol{p}"
        )(core.Signal(x[:, p], t))
        outs.append(y[:, None])
    return core.Signal(jnp.concatenate(outs, axis=1), tp)

core.conv1d_fft = conv1d_fft_debug
core.dconv_pair = dconv_pair_debug
print("✓ debug_patch loaded — conv1d_fft & dconv_pair now guarded.")
