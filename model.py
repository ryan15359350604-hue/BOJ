"""Illustrative, linearized forward-looking 3-equation NK model.

Deviations from a zero-gap steady state; inflation and nominal interest rate
are expressed in percentage-point deviations from their steady-state values.
All parameters are illustrative, not estimates for Japan.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NKParameters:
    beta: float = 0.99
    sigma: float = 1.0
    kappa: float = 0.15
    phi_pi: float = 1.5
    phi_x: float = 0.5
    persistence: float = 0.55
    horizon: int = 24
    inflation_target: float = 2.0
    steady_nominal_rate: float = 2.5


def solve_nk(params: NKParameters, shock_type: str, magnitude: float, duration: int = 1):
    """Solve an anticipated, perfect-foresight linear rational-expectations path.

    x_t = x_{t+1} - (i_gap_t - pi_gap_{t+1} - rn_gap_t)/sigma + d_t
    pi_gap_t = beta*pi_gap_{t+1} + kappa*x_t + u_t
    i_gap_t = phi_pi*pi_gap_t + phi_x*x_t + m_t

    Terminal conditions: x_T = 0 and pi_gap_T = 0.
    Shocks: demand d_t, supply/cost-push u_t, monetary policy m_t,
    natural real interest rate rn_gap_t. A single shock is switched on.
    """
    T = params.horizon
    if not 1 <= duration <= T:
        raise ValueError('duration must be between 1 and horizon')
    if params.sigma <= 0:
        raise ValueError('sigma must be positive')
    shocks = {k: np.zeros(T) for k in ('demand', 'supply', 'monetary', 'natural_rate')}
    if shock_type not in shocks:
        raise ValueError(f'Unknown shock type: {shock_type}')
    shocks[shock_type][:duration] = magnitude * (params.persistence ** np.arange(duration))

    # Unknown z = [x_0,...,x_(T-1), pi_0,...,pi_(T-1)]
    A = np.zeros((2*T, 2*T))
    b = np.zeros(2*T)
    for t in range(T):
        # Substituting the Taylor rule into the Dynamic IS equation:
        # (1+phi_x/sigma)x_t+(phi_pi/sigma)pi_t
        #   -x_(t+1)-(1/sigma)pi_(t+1) = d_t+(rn_t-m_t)/sigma
        A[t, t] = 1 + params.phi_x / params.sigma
        A[t, T+t] = params.phi_pi / params.sigma
        if t < T-1:
            A[t, t+1] = -1
            A[t, T+t+1] = -1 / params.sigma
        b[t] = shocks['demand'][t] + (shocks['natural_rate'][t]-shocks['monetary'][t]) / params.sigma

        # NKPC: -kappa*x_t + pi_t - beta*pi_(t+1) = cost-push shock.
        A[T+t, t] = -params.kappa
        A[T+t, T+t] = 1
        if t < T-1:
            A[T+t, T+t+1] = -params.beta
        b[T+t] = shocks['supply'][t]

    z = np.linalg.solve(A,b)
    x, pi = z[:T], z[T:]
    i_gap = params.phi_pi*pi + params.phi_x*x + shocks['monetary']
    pi_next = np.r_[pi[1:], 0.0]
    x_next = np.r_[x[1:], 0.0]
    real_gap = i_gap - pi_next
    return {
        'period': np.arange(T), 'x': x, 'pi_gap': pi, 'i_gap': i_gap,
        'real_gap': real_gap, 'pi_level': params.inflation_target+pi,
        'i_level': params.steady_nominal_rate+i_gap,
        'x_next': x_next, 'pi_next': pi_next, 'shocks': shocks,
    }


def curves_for_period(params: NKParameters, result, t: int = 0):
    """Plot IS and monetary-policy schedules in (x, ex ante real-rate gap).

    At the selected period, next-period expectations are held fixed along
    each schedule. This is an illustrative conditional slice of the NK model,
    not a classical money-supply LM curve.
    """
    x_grid = np.linspace(-6, 6, 301)
    exp_x_next, exp_pi_next = result['x_next'][t], result['pi_next'][t]
    s = result['shocks']
    demand, rn, monetary, supply = (s[k][t] for k in ('demand', 'natural_rate','monetary','supply'))
    # From IS: r_gap = rn + sigma*(E x_next + d - x)
    is_rate = rn + params.sigma * (exp_x_next + demand - x_grid)
    # With NKPC holding E pi_next constant, pi = beta E pi_next+kappa*x+u.
    pc_pi = params.beta*exp_pi_next+params.kappa*x_grid+supply
    # From Taylor rule: r_gap = phi_pi*pi + phi_x*x +m - E pi_next
    mp_rate = params.phi_pi*pc_pi+params.phi_x*x_grid+monetary-exp_pi_next
    return {'x_grid': x_grid, 'is_rate': is_rate, 'mp_rate': mp_rate, 'pc_pi': pc_pi}


def test_residuals(params, result):
    x, pi, i_gap = result['x'], result['pi_gap'], result['i_gap']
    x_next, pi_next, s = result['x_next'], result['pi_next'], result['shocks']
    r1 = x-x_next+(i_gap-pi_next-s['natural_rate'])/params.sigma-s['demand']
    r2 = pi-params.beta*pi_next-params.kappa*x-s['supply']
    r3 = i_gap-params.phi_pi*pi-params.phi_x*x-s['monetary']
    return float(max(np.max(np.abs(z)) for z in (r1,r2,r3)))
