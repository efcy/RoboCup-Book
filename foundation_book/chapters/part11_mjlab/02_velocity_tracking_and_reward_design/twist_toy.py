import numpy as np

rng = np.random.default_rng(6)
DT, T = 0.1, 60.0
t = 0.0; heading = 0.0; next_t = 0.0
mode, target = "stand", 0.0
vx = vy = 0.0
rows = []
log = []
modes = []
while t < T:
    if t >= next_t:                          # resample the command
        next_t = t + rng.uniform(3.0, 8.0)
        m = rng.uniform()
        log.append(round(t, 1))
        if m < 0.1:                          # standing
            mode, vx, vy = "stand", 0.0, 0.0
        elif m < 0.3:                        # lateral
            mode, vx, vy = "side", 0.0, rng.uniform(-0.3, 0.3)
        else:                                # forward + heading
            mode, vy = "fwd", 0.0
            vx = rng.uniform(-1.0, 1.0)
            target = rng.uniform(-np.pi, np.pi)
    if mode == "fwd":
        err = (target - heading + np.pi) % (2 * np.pi) - np.pi
        wz = float(np.clip(0.5 * err, -0.5, 0.5))
    else:
        wz = 0.0
    heading += wz * DT                       # robot follows perfectly
    rows.append((t, vx, vy, wz))
    modes.append(mode)
    t += DT
rows = np.array(rows)
np.savetxt("twist.dat", rows, header="t vx vy wz", comments="", fmt="%.3f")
print(len(log), "commands at", log)
print({k: modes.count(k) * DT for k in set(modes)})
