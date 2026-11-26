import numpy as np

# --- Physics of one kick (all numbers are ASSUMPTIONS for illustration) ---
M_BALL, M_EFF, E = 0.30, 1.5, 0.6   # ball mass, foot mass, restitution
A_ROLL = 0.5                        # rolling deceleration (m/s^2)

def ball_speed(v_foot):
    """1-D collision of a foot of effective mass M_EFF with a ball."""
    return (1 + E) * M_EFF / (M_EFF + M_BALL) * v_foot

def roll_distance(v_ball, a=A_ROLL):
    return v_ball**2 / (2 * a)

def foot_speed_for(distance, a=A_ROLL):
    v_ball = np.sqrt(2 * a * distance)
    return v_ball * (M_EFF + M_BALL) / ((1 + E) * M_EFF)

# --- Reward terms for a kick task, written for a batch of environments ---
def r_approach(foot_pos, ball_pos, contact_done, sigma=0.3):
    """Before contact: bring the kicking foot to the ball."""
    d2 = np.sum((foot_pos - ball_pos)**2, axis=-1)
    return np.exp(-d2 / sigma**2) * (~contact_done)

def r_ball_velocity(ball_vel, cmd_vel, contact_done, sigma=0.5):
    """After contact: the ball must leave with the commanded velocity."""
    e2 = np.sum((ball_vel - cmd_vel)**2, axis=-1)
    return np.exp(-e2 / sigma**2) * contact_done

if __name__ == "__main__":
    for v in (1.0, 2.0, 3.0):
        vb = ball_speed(v)
        print(f"foot {v:.1f} -> ball {vb:.2f} m/s,",
              f"rolls {roll_distance(vb):.1f} m")
    for d in (2.0, 5.0, 8.0):
        print(f"{d:.0f} m needs {foot_speed_for(d):.2f} m/s of foot")
    cmd = np.array([[2.6, 0.0]])
    print(r_ball_velocity(np.array([[2.4, 0.0]]), cmd, np.array([True])))
    print(r_ball_velocity(np.array([[2.4, 0.5]]), cmd, np.array([True])))
    foot, ball = np.array([[0.1, 0, 0]]), np.array([[0.2, 0, 0]])
    print(r_approach(foot, ball, np.array([False])))
