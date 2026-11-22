"""IS–LM and Taylor Rule model equations.

Uses the functional-form specialization derived in Kazakova's Lecture 8
and the simplified expectations-augmented Phillips curve / Taylor rule
in Kurlat Ch.15. See README for model assumptions and parameter definitions.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import brentq

@dataclass(frozen=True)
class CourseParameters:
    beta: float = 0.5            # Lecture 8: log-utility household discount factor
    expected_future_pi: float = 0.0  # Lecture 8 Fisher approximation; decimal
    lam: float = 1.0
    p1: float = 1.0
    money_supply: float = 1.0
    g: float = 0.0          # government purchases term
    y_n: float = 0.68      # natural output; editable baseline value
    r_n: float = 0.0       # natural real interest rate; decimal
    pi_expected: float = 0.02  # current inflation expectations; decimal
    pi_target: float = 0.02
    pc_a: float = 0.05     # PC slope (inflation fraction per unit of Y)
    taylor_a: float = 0.5 # textbook coefficients a,b
    taylor_b: float = 0.5
    cost_push: float = 0.0 # cost-push term
    policy_wedge: float = 0.0 # optional interest-rate shock in Taylor rule
    price_pc_slope: float = 0.25  # reduced-form price adjustment slope (log price per unit output)


def is_output(i, p: CourseParameters):
    """Lecture 8 p24: Y1=(2+beta)/(16 beta (1+i-pi)^2); +G extension.
    Valid where 1+i-pi>0. Rates are represented as decimal fractions.
    """
    rate = 1 + np.asarray(i) - p.expected_future_pi
    return np.where(rate > 1.e-8, (2+p.beta)/(16*p.beta*np.maximum(rate,1.e-8)**2)+p.g, np.nan)


def is_nominal_rate(y, p: CourseParameters):
    y = np.asarray(y)
    return np.where(y>p.g, p.expected_future_pi-1+np.sqrt((2+p.beta)/(16*p.beta*np.maximum(y-p.g,1.e-10))), np.nan)


def pc_price(y, p: CourseParameters):
    """Reduced-form upward-sloping price-level PC for the static chapter-14 view.

    This is a specified reduced-form price-adjustment equation.
    At Y=Y_n the price level equals p1 (the adjustable anchor).
    """
    return p.p1*np.exp(p.price_pc_slope*(np.asarray(y)-p.y_n))

def lm_rate(y, p: CourseParameters):
    # With partial price flexibility, the LM relation incorporates p1(Y).
    return p.lam*pc_price(y,p)*np.asarray(y)/p.money_supply-1


def pc_inflation(y, p: CourseParameters):
    # Kurlat equation (15.2.10) including a cost-push term
    return p.pi_expected+p.pc_a*(np.asarray(y)-p.y_n)+p.cost_push


def taylor_rate(y, p: CourseParameters):
    pi = pc_inflation(y,p)
    # textbook Eq 15.3.1, with an optional additive shock
    return p.r_n+pi+p.taylor_a*(pi-p.pi_target)+p.taylor_b*(np.asarray(y)-p.y_n)+p.policy_wedge


def _find_root(fun, low=1e-6, high=100.0):
    xs=np.geomspace(low,high,300)
    vals=[float(fun(x)) for x in xs]
    for x0,x1,f0,f1 in zip(xs[:-1],xs[1:],vals[:-1],vals[1:]):
        if np.isfinite(f0) and np.isfinite(f1) and f0*f1<=0:
            return float(brentq(fun,x0,x1))
    raise ValueError('No valid equilibrium found for these parameters; reduce shock magnitude or change settings.')


def solve_course(p:CourseParameters, regime='Money Supply / IS-LM'):
    if not (p.beta>0 and p.money_supply>0 and p.lam>0 and p.p1>0):
        raise ValueError('beta, money supply, lambda and price level must be positive')
    rate_fn=lm_rate if regime=='Money Supply / IS-LM' else taylor_rate
    y=_find_root(lambda y: float(is_nominal_rate(y,p)-rate_fn(y,p)),low=max(1e-5,p.g+1e-5))
    i=float(rate_fn(y,p))
    pi=float(pc_inflation(y,p))
    return {'Y1':y,'i':i,'r':i-p.expected_future_pi,'pi1':pi,
            'output_gap':y-p.y_n,'price_level':float(pc_price(y,p)),
            'implied_Ms':float(pc_price(y,p)*p.lam*y/(1+i)) if i>-1 else np.nan,
            'is_residual':float(y-is_output(i,p)),
            'policy_residual':float(i-rate_fn(y,p))}


def apply_shock(p:CourseParameters, kind, magnitude):
    mapping={
       'Increase money supply': ('money_supply', lambda v:v*(1+magnitude/100)),
       'Higher expected future inflation': ('expected_future_pi', lambda v:v+magnitude/100),
       'Lower discount factor (impatience)': ('beta',lambda v:v*(1-magnitude/100)),
       'Higher government spending': ('g',lambda v:v+magnitude/100),
       'Cost-push shock': ('cost_push',lambda v:v+magnitude/100),
       'Policy-rate shock': ('policy_wedge',lambda v:v+magnitude/100),
       'Change inflation expectations (PC)': ('pi_expected',lambda v:v+magnitude/100),
    }
    if kind=='No shock':return p
    field, fn=mapping[kind]
    new_value=fn(getattr(p,field))
    if field=='money_supply' and new_value<=0:raise ValueError('Money supply must be positive.')
    if field=='beta' and new_value<=0:raise ValueError('Discount factor must be positive.')
    return replace(p,**{field:new_value})
