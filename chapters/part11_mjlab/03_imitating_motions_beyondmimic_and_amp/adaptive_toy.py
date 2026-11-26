import numpy as np

BINS, N, ALPHA, UNI = 20, 1024, 0.05, 0.1

def run(adaptive, seed, max_it=400):
    rng = np.random.default_rng(seed)
    p = np.full(BINS, 0.03)
    p[12:14] = 0.5                       # a hard part of the motion
    fail_ema = np.zeros(BINS)
    for it in range(max_it):
        if p.max() < 0.05:
            return it, p.max()
        if adaptive:
            w = fail_ema + UNI / BINS
            w = w / w.sum()
        else:
            w = np.full(BINS, 1.0 / BINS)
        n = rng.multinomial(N, w)        # starts per bin
        f = rng.binomial(n, p)           # failures per bin
        fail_ema = ALPHA * f + (1 - ALPHA) * fail_ema
        p = p * np.exp(-0.002 * n)       # practice makes better
    return max_it, p.max()

for mode in (False, True):
    its = [run(mode, s)[0] for s in range(100)]
    print("adaptive" if mode else "uniform ", "median iterations:",
          int(np.median(its)), " 90th percentile:",
          int(np.percentile(its, 90)))
