import numpy as np

rng = np.random.default_rng(0)
G, MC, MP, L = 9.81, 1.0, 0.1, 0.5      # gravity, masses, pole length
DT, DEC = 0.01, 5                       # physics step, decimation

def step(s, u):
    """s: [num_envs, 4] = (x, xdot, theta, thetadot), theta=0 upright."""
    x, xd, th, thd = s.T
    for _ in range(DEC):
        f = 10.0 * np.clip(u, -1, 1)
        c, sn = np.cos(th), np.sin(th)
        tmp = (f + MP * L * thd**2 * sn) / (MC + MP)
        thdd = (G * sn - c * tmp) / (L * (4/3 - MP * c**2 / (MC + MP)))
        xdd = tmp - MP * L * thdd * c / (MC + MP)
        x += DT * xd; xd += DT * xdd
        th += DT * thd; thd += DT * thdd
    return np.stack([x, xd, th, thd], 1)

def reward(s, u):
    x, _, th, _ = s.T
    upright = (np.cos(th) + 1) / 2
    centred = np.exp(-x**2)
    small_u = (4 + np.exp(-(np.clip(u, -1, 1) / 0.5)**2)) / 5
    return upright * centred * small_u

def rollout(K, n_steps=200):
    """One linear policy u = K.s per environment; returns the returns."""
    n = K.shape[0]
    s = np.stack([rng.uniform(-.1, .1, n), rng.uniform(-.01, .01, n),
                  rng.uniform(-.1, .1, n), rng.uniform(-.01, .01, n)], 1)
    ret, alive = np.zeros(n), np.ones(n, bool)
    for _ in range(n_steps):
        u = np.einsum('ij,ij->i', K, s)
        s = step(s, u)
        alive &= (np.abs(s[:, 0]) < 1.8) & (np.abs(s[:, 2]) < 0.8)
        ret += reward(s, u) * alive
    return ret

K, sigma, lr = np.zeros(4), 1.0, 0.5
for it in range(61):
    eps = rng.standard_normal((4096, 4))        # 4096 perturbed policies
    R = rollout(K + sigma * eps)                # 4096 environments at once
    adv = (R - R.mean()) / (R.std() + 1e-8)
    K = K + lr * sigma * (adv[:, None] * eps).mean(0)
    if it % 20 == 0:
        print(it, round(R.mean(), 1), K.round(2))
print("mean policy:", rollout(np.tile(K, (1000, 1))).mean().round(1))
