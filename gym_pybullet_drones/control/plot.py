import numpy as np
import matplotlib.pyplot as plt

import gym_pybullet_drones.trajectory.circle as C
from gym_pybullet_drones.trajectory.diamond import diamond

m, g = 0.027, 9.8
mg = m * g
E3 = np.array([0.0, 0.0, 1.0])
LOW, HIGH = 1.5, 2.25          # allowed peak window in units of mg


# ---------------- trajectory wrappers ----------------
def set_circle_duration(T):
    """Rebuild circle.py's globals and polynomials for total duration T."""
    C.climb_up_time_duration = T / 3
    C.circle_time_duration = 2 * T / 3
    C.climb_down_time_duration = T
    C.climb_up_traj = C.PolynomialTrajectory(
        to=0.0, tf=C.climb_up_time_duration,
        initial_state=C.climb_up_initial, final_state=C.climb_up_final)
    C.theta_traj = C.PolynomialTrajectory(
        to=C.climb_up_time_duration, tf=C.circle_time_duration,
        initial_state=C.theta_initial, final_state=C.theta_final)
    C.climb_down_traj = C.PolynomialTrajectory(
        to=C.circle_time_duration, tf=C.climb_down_time_duration,
        initial_state=C.climb_down_initial, final_state=C.climb_down_final)


def circle_acc(t, T):
    return np.asarray(C.circle(t, T)['acc'], dtype=float)


def diamond_acc(t, T):
    return np.asarray(diamond(t, T)['acc'], dtype=float)


# ---------------- f_d(t) = m ||a_d + g e3|| ----------------
def fd_ratio(acc_fn, T, setup=None, n=1500):
    """Return t and f_d/(mg) for duration T."""
    if setup:
        setup(T)
    t = np.linspace(0, T, n)
    a = np.array([acc_fn(ti, T) for ti in t])
    return t, np.linalg.norm(a / g + E3, axis=1)


def peak(acc_fn, T, setup=None):
    return fd_ratio(acc_fn, T, setup)[1].max()


def solve_T(acc_fn, target, setup=None, lo=1.0, hi=60.0):
    """Duration whose peak f_d equals target*mg (peak decreases as T grows)."""
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if peak(acc_fn, mid, setup) > target:
            lo = mid
        else:
            hi = mid
    return hi


def pick_durations(acc_fn, T0, setup=None):
    """T0 plus two shorter durations, the shortest peaking just under HIGH."""
    # T_mid = np.ceil(solve_T(acc_fn, 1.7, setup) * 10) / 10
    # T_short = np.ceil(solve_T(acc_fn, 2.1, setup) * 10) / 10
    T_mid = 3.5
    T_short = 3
    return [T0, T_mid, T_short]


# ---------------- plotting ----------------
def plot_fd(acc_fn, durations, name, setup=None):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for T in durations:
        t, r = fd_ratio(acc_fn, T, setup)
        print(f'{name} T={T}: peak f_d = {r.max():.2f} mg = {r.max()*mg:.3f} N')
        ax.plot(t, r, label=f'T = {T:g} s (peak {r.max():.2f} mg)')
    ax.axhline(HIGH, color='r', ls='--', label=r'$f_{max}=2.25\,mg$')
    ax.axhline(LOW, color='gray', ls=':', label=r'$1.5\,mg$')
    ax.set_ylim(0, HIGH + 0.6)
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$f_d / mg$')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7, loc='upper right')
    sec = ax.secondary_yaxis('right', functions=(lambda r: r * mg, lambda f: f / mg))
    sec.set_ylabel(r'$f_d$ [N]')
    fig.tight_layout()
    fig.savefig(f'{name}_fd.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    plot_fd(circle_acc,  [15, 9, 10], 'circle', setup=set_circle_duration)
    plot_fd(diamond_acc, [8, 3.5, 3.0],  'diamond')