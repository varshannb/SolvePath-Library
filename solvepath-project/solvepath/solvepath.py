"""
SolvePath — A self-contained math library that mirrors many functions from Python's math module
plus number-theoretic helpers. Functions default to returning the plain result, but if
`show_steps=True` is passed the function will return a tuple `(result, steps)` where
`steps` is a human-readable list of strings describing the calculation step-by-step.

Usage examples:
    from solvepath import comb, gcd
    comb(5, 2)                      # -> 10
    comb(5, 2, show_steps=True)     # -> (10, ["...step strings..."])

Notes:
- This file is meant to be saved as `solvepath.py` and imported.
- For performance-critical operations use the plain call (without steps).
- Many functions simply wrap Python's math module; step-by-step details are provided
  for combinatorics and elementary number-theoretic operations.
"""

from __future__ import annotations

import math
import functools
import operator
from typing import Iterable, Tuple, List, Union, Optional
from fractions import Fraction

Number = Union[int, float]

# --- Internal utilities -----------------------------------------------------

def _with_steps(result, steps: Optional[List[str]], show_steps: bool):
    return (result, steps) if show_steps else result

# helper to build step lists
def _fmt_steps(*lines: str) -> List[str]:
    return [str(l) for l in lines]

# --- Number-theoretic functions --------------------------------------------

def factorial(n: int, show_steps: bool = False) -> Union[int, Tuple[int, List[str]]]:
    """Return n! (n factorial). If show_steps=True, return (result, steps).
    Steps show the multiplicative chain.
    """
    if n < 0:
        raise ValueError("factorial() not defined for negative values")
    res = 1
    steps = [f"Start: result = 1"]
    for i in range(1, n + 1):
        res *= i
        steps.append(f"Multiply by {i}: result = {res}")
    steps.append(f"Final: {n}! = {res}")
    return _with_steps(res, steps, show_steps)


def comb(n: int, k: int, show_steps: bool = False) -> Union[int, Tuple[int, List[str]]]:
    """Number of ways to choose k items from n without order: C(n, k).
    Uses multiplicative formula to avoid huge intermediate factorials.
    """
    if not (0 <= k <= n):
        return _with_steps(0, [f"Invalid: k={k} not in [0, {n}]"], show_steps)
    k = min(k, n - k)
    if k == 0:
        return _with_steps(1, ["C(n,0)=1"], show_steps)
    num = 1
    den = 1
    steps = [f"Compute C({n},{k}) using multiplicative formula"]
    for i in range(1, k + 1):
        num *= n - (k - i)
        den *= i
        steps.append(f"Step {i}: multiply numerator by {n - (k - i)} -> num={num}; multiply denominator by {i} -> den={den}")
    res = num // den
    steps.append(f"Result = num/den = {num}/{den} = {res}")
    return _with_steps(res, steps, show_steps)


def perm(n: int, k: int, show_steps: bool = False) -> Union[int, Tuple[int, List[str]]]:
    """Number of permutations: P(n, k) = n*(n-1)*...*(n-k+1)"""
    if not (0 <= k <= n):
        return _with_steps(0, [f"Invalid: k={k} not in [0, {n}]"], show_steps)
    res = 1
    steps = [f"Compute P({n},{k}) by multiplying k terms starting at {n}"]
    for i in range(k):
        res *= (n - i)
        steps.append(f"Multiply by {n - i} -> result={res}")
    steps.append(f"Final: P({n},{k}) = {res}")
    return _with_steps(res, steps, show_steps)


def gcd(*integers: int, show_steps: bool = False) -> Union[int, Tuple[int, List[str]]]:
    """Greatest common divisor using Euclid's algorithm. Accepts multiple args."""
    if len(integers) == 0:
        raise TypeError("gcd() requires at least one integer")
    steps: List[str] = [f"Compute gcd of {integers}"]
    def _gcd(a: int, b: int) -> int:
        steps_local = []
        a0, b0 = abs(a), abs(b)
        steps_local.append(f"Start gcd({a0}, {b0})")
        while b0:
            q = a0 // b0
            r = a0 % b0
            steps_local.append(f"{a0} = {b0} * {q} + {r} (remainder)")
            a0, b0 = b0, r
        steps_local.append(f"Result: {a0}")
        steps.extend(steps_local)
        return a0
    result = functools.reduce(_gcd, integers)
    return _with_steps(result, steps, show_steps)


def isqrt(n: int, show_steps: bool = False) -> Union[int, Tuple[int, List[str]]]:
    """Integer square root: floor(sqrt(n)). Uses math.isqrt if available."""
    if n < 0:
        raise ValueError("isqrt() argument must be nonnegative")
    try:
        res = math.isqrt(n)
        steps = [f"Using built-in math.isqrt: isqrt({n}) = {res}"]
    except AttributeError:
        # fallback binary search
        steps = [f"Binary search integer sqrt of {n}"]
        lo, hi = 0, n
        while lo <= hi:
            mid = (lo + hi) // 2
            sq = mid * mid
            steps.append(f"Test mid={mid}: mid^2={sq}")
            if sq <= n:
                lo = mid + 1
                res = mid
            else:
                hi = mid - 1
        steps.append(f"Result: isqrt({n}) = {res}")
    return _with_steps(res, steps, show_steps)


def lcm(*integers: int, show_steps: bool = False) -> Union[int, Tuple[int, List[str]]]:
    """Least common multiple of multiple integers."""
    if len(integers) == 0:
        raise TypeError("lcm() requires at least one integer")
    steps = [f"Compute lcm of {integers}"]
    def _l(a, b):
        g = math.gcd(a, b)
        l = abs(a // g * b)
        steps.append(f"lcm({a},{b}) = abs({a} * {b})/gcd({a},{b}) = {l}")
        return l
    result = functools.reduce(_l, integers)
    return _with_steps(result, steps, show_steps)

# --- Floating point arithmetic & manipulation -------------------------------

def ceil(x: Number, show_steps: bool = False):
    return _with_steps(math.ceil(x), [f"ceil({x}) = {math.ceil(x)}"], show_steps)

def fabs(x: Number, show_steps: bool = False):
    return _with_steps(math.fabs(x), [f"fabs({x}) = {math.fabs(x)}"], show_steps)

def floor(x: Number, show_steps: bool = False):
    return _with_steps(math.floor(x), [f"floor({x}) = {math.floor(x)}"], show_steps)

def fma(x: Number, y: Number, z: Number, show_steps: bool = False):
    # Python 3.9+ math.fma
    try:
        val = math.fma(x, y, z)
        steps = [f"Compute fma: ({x} * {y}) + {z} = {val}"]
    except AttributeError:
        val = x * y + z
        steps = [f"Fallback: multiply then add -> ({x} * {y}) + {z} = {val}"]
    return _with_steps(val, steps, show_steps)

def fmod(x: Number, y: Number, show_steps: bool = False):
    val = math.fmod(x, y)
    return _with_steps(val, [f"fmod({x},{y}) = {val}"], show_steps)

def modf(x: Number, show_steps: bool = False):
    frac, inte = math.modf(x)
    steps = [f"modf({x}) -> fractional={frac}, integer={inte}"]
    return _with_steps((frac, inte), steps, show_steps)

def remainder(x: Number, y: Number, show_steps: bool = False):
    val = math.remainder(x, y) if hasattr(math, 'remainder') else (x % y)
    return _with_steps(val, [f"remainder({x},{y}) = {val}"], show_steps)

def trunc(x: Number, show_steps: bool = False):
    val = math.trunc(x)
    return _with_steps(val, [f"trunc({x}) = {val}"], show_steps)

# floating point manipulation

def copysign(x: Number, y: Number, show_steps: bool = False):
    val = math.copysign(x, y)
    return _with_steps(val, [f"copysign({x},{y}) = {val}"], show_steps)

def frexp(x: Number, show_steps: bool = False):
    m, e = math.frexp(x)
    steps = [f"frexp({x}) -> mantissa={m}, exponent={e}"]
    return _with_steps((m, e), steps, show_steps)

def isclose(a: Number, b: Number, rel_tol: float = 1e-9, abs_tol: float = 0.0, show_steps: bool = False):
    ok = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    steps = [f"isclose({a},{b}, rel_tol={rel_tol}, abs_tol={abs_tol}) -> {ok}"]
    return _with_steps(ok, steps, show_steps)

def isfinite(x: Number, show_steps: bool = False):
    return _with_steps(math.isfinite(x), [f"isfinite({x}) = {math.isfinite(x)}"], show_steps)

def isinf(x: Number, show_steps: bool = False):
    return _with_steps(math.isinf(x), [f"isinf({x}) = {math.isinf(x)}"], show_steps)

def isnan(x: Number, show_steps: bool = False):
    return _with_steps(math.isnan(x), [f"isnan({x}) = {math.isnan(x)}"], show_steps)

def ldexp(x: Number, i: int, show_steps: bool = False):
    val = math.ldexp(x, i)
    return _with_steps(val, [f"ldexp({x},{i}) -> {x} * (2**{i}) = {val}"], show_steps)

def nextafter(x: Number, y: Number, steps: int = 1, show_steps: bool = False):
    # steps param allows multiple single-step moves; math.nextafter available in new Python
    if hasattr(math, 'nextafter'):
        val = x
        log_steps = [f"Start nextafter from {x} towards {y}, steps={steps}"]
        for i in range(steps):
            val = math.nextafter(val, y)
            log_steps.append(f" step {i+1}: {val}")
        return _with_steps(val, log_steps, show_steps)
    else:
        # best-effort fallback: move a tiny amount toward y
        tiny = math.ldexp(1.0, -52)  # approx machine epsilon scaled
        direction = 1 if y > x else -1
        val = x + direction * tiny
        return _with_steps(val, [f"Fallback single move: {x} -> {val} toward {y}"], show_steps)


def ulp(x: Number, show_steps: bool = False):
    if hasattr(math, 'ulp'):
        val = math.ulp(x)
        return _with_steps(val, [f"ulp({x}) = {val}"], show_steps)
    # fallback: distance to nextafter
    if math.isnan(x) or math.isinf(x):
        return _with_steps(float('nan'), [f"ulp({x}) undefined (NaN or Inf)"], show_steps)
    next_up = math.nextafter(x, math.inf) if hasattr(math, 'nextafter') else x + math.ldexp(1.0, -52)
    val = abs(next_up - x)
    return _with_steps(val, [f"approx ulp({x}) = |next_up - x| = {val}"], show_steps)

# --- Power, exponential and logarithmic functions --------------------------

def cbrt(x: Number, show_steps: bool = False):
    val = math.copysign(abs(x) ** (1.0 / 3.0), x)
    return _with_steps(val, [f"cbrt({x}) = {val}"], show_steps)

def exp(x: Number, show_steps: bool = False):
    val = math.exp(x)
    return _with_steps(val, [f"exp({x}) = {val}"], show_steps)

def exp2(x: Number, show_steps: bool = False):
    val = 2 ** x
    return _with_steps(val, [f"exp2({x}) = 2**{x} = {val}"], show_steps)

def expm1(x: Number, show_steps: bool = False):
    val = math.expm1(x)
    return _with_steps(val, [f"expm1({x}) = {val}"], show_steps)

def log(x: Number, base: Optional[Number] = None, show_steps: bool = False):
    if base is None:
        val = math.log(x)
        steps = [f"ln({x}) = {val}"]
    else:
        val = math.log(x, base)
        steps = [f"log base {base} of {x} = {val}"]
    return _with_steps(val, steps, show_steps)

def log1p(x: Number, show_steps: bool = False):
    val = math.log1p(x)
    return _with_steps(val, [f"log1p({x}) = {val}"], show_steps)

def log2(x: Number, show_steps: bool = False):
    val = math.log2(x)
    return _with_steps(val, [f"log2({x}) = {val}"], show_steps)

def log10(x: Number, show_steps: bool = False):
    val = math.log10(x)
    return _with_steps(val, [f"log10({x}) = {val}"], show_steps)

def pow(x: Number, y: Number, show_steps: bool = False):
    val = math.pow(x, y)
    return _with_steps(val, [f"pow({x},{y}) = {val}"], show_steps)

def sqrt(x: Number, show_steps: bool = False):
    val = math.sqrt(x)
    return _with_steps(val, [f"sqrt({x}) = {val}"], show_steps)

# --- Summation and product functions --------------------------------------

def dist(p: Iterable[Number], q: Iterable[Number], show_steps: bool = False):
    p_list = list(p)
    q_list = list(q)
    if len(p_list) != len(q_list):
        raise ValueError("dist: p and q must have same length")
    steps = [f"Compute Euclidean distance between {p_list} and {q_list}"]
    s = 0.0
    for i, (a, b) in enumerate(zip(p_list, q_list)):
        d = (a - b) ** 2
        s += d
        steps.append(f"coord {i}: ({a} - {b})^2 = {d}, partial sum = {s}")
    val = math.sqrt(s)
    steps.append(f"sqrt({s}) = {val}")
    return _with_steps(val, steps, show_steps)


def fsum(iterable: Iterable[Number], show_steps: bool = False):
    vals = list(iterable)
    s = math.fsum(vals)
    steps = [f"Accurately sum {vals} -> {s}"]
    return _with_steps(s, steps, show_steps)


def hypot(*coordinates: Number, show_steps: bool = False):
    val = math.hypot(*coordinates)
    steps = [f"hypot{coordinates} = {val}"]
    return _with_steps(val, steps, show_steps)


def prod(iterable: Iterable[Number], start: Number = 1, show_steps: bool = False):
    res = start
    steps = [f"Start product with start={start}"]
    for i, v in enumerate(iterable):
        res *= v
        steps.append(f"Multiply by {v} -> {res}")
    steps.append(f"Final product = {res}")
    return _with_steps(res, steps, show_steps)


def sumprod(p: Iterable[Number], q: Iterable[Number], show_steps: bool = False):
    p_list = list(p)
    q_list = list(q)
    if len(p_list) != len(q_list):
        raise ValueError("sumprod: iterables must have same length")
    s = 0
    steps = [f"Compute sum of products for pairs"]
    for i, (a, b) in enumerate(zip(p_list, q_list)):
        s += a * b
        steps.append(f"pair {i}: {a}*{b} = {a*b}, partial sum = {s}")
    steps.append(f"Final sumprod = {s}")
    return _with_steps(s, steps, show_steps)

# --- Angular conversion ---------------------------------------------------

def degrees(x: Number, show_steps: bool = False):
    val = math.degrees(x)
    return _with_steps(val, [f"degrees({x}) = {val}"], show_steps)

def radians(x: Number, show_steps: bool = False):
    val = math.radians(x)
    return _with_steps(val, [f"radians({x}) = {val}"], show_steps)

# --- Trigonometric functions ----------------------------------------------

def acos(x: Number, show_steps: bool = False):
    val = math.acos(x)
    return _with_steps(val, [f"acos({x}) = {val}"], show_steps)

def asin(x: Number, show_steps: bool = False):
    val = math.asin(x)
    return _with_steps(val, [f"asin({x}) = {val}"], show_steps)

def atan(x: Number, show_steps: bool = False):
    val = math.atan(x)
    return _with_steps(val, [f"atan({x}) = {val}"], show_steps)

def atan2(y: Number, x: Number, show_steps: bool = False):
    val = math.atan2(y, x)
    return _with_steps(val, [f"atan2({y},{x}) = {val}"], show_steps)

def cos(x: Number, show_steps: bool = False):
    val = math.cos(x)
    return _with_steps(val, [f"cos({x}) = {val}"], show_steps)

def sin(x: Number, show_steps: bool = False):
    val = math.sin(x)
    return _with_steps(val, [f"sin({x}) = {val}"], show_steps)

def tan(x: Number, show_steps: bool = False):
    val = math.tan(x)
    return _with_steps(val, [f"tan({x}) = {val}"], show_steps)

# --- Hyperbolic functions -------------------------------------------------

def acosh(x: Number, show_steps: bool = False):
    val = math.acosh(x)
    return _with_steps(val, [f"acosh({x}) = {val}"], show_steps)

def asinh(x: Number, show_steps: bool = False):
    val = math.asinh(x)
    return _with_steps(val, [f"asinh({x}) = {val}"], show_steps)

def atanh(x: Number, show_steps: bool = False):
    val = math.atanh(x)
    return _with_steps(val, [f"atanh({x}) = {val}"], show_steps)

def cosh(x: Number, show_steps: bool = False):
    val = math.cosh(x)
    return _with_steps(val, [f"cosh({x}) = {val}"], show_steps)

def sinh(x: Number, show_steps: bool = False):
    val = math.sinh(x)
    return _with_steps(val, [f"sinh({x}) = {val}"], show_steps)

def tanh(x: Number, show_steps: bool = False):
    val = math.tanh(x)
    return _with_steps(val, [f"tanh({x}) = {val}"], show_steps)

# --- Special functions ----------------------------------------------------

def erf(x: Number, show_steps: bool = False):
    val = math.erf(x)
    return _with_steps(val, [f"erf({x}) = {val}"], show_steps)

def erfc(x: Number, show_steps: bool = False):
    val = math.erfc(x)
    return _with_steps(val, [f"erfc({x}) = {val}"], show_steps)

def gamma(x: Number, show_steps: bool = False):
    val = math.gamma(x)
    return _with_steps(val, [f"gamma({x}) = {val}"], show_steps)

def lgamma(x: Number, show_steps: bool = False):
    val = math.lgamma(x)
    return _with_steps(val, [f"lgamma({x}) = {val}"], show_steps)

# --- Constants ------------------------------------------------------------
pi = math.pi
"""Pi constant"""

e = math.e
"""Euler's number"""

tau = math.tau if hasattr(math, 'tau') else 2 * math.pi
"""Tau constant (2*pi)"""

inf = math.inf
"""Positive infinity"""

nan = math.nan
"""Not a Number (NaN)"""
# --- Sigmoid ------------------------------------------------------------
def sigmoid(x, show_steps=False):
    steps = []
    exp_val = math.exp(-x)
    result = 1 / (1 + exp_val)

    steps.append(f"Compute exp(-{x}) = {exp_val}")
    steps.append(f"Denominator = 1 + {exp_val}")
    steps.append(f"Sigmoid = {result}")

    return _with_steps(result, steps, show_steps)



def relu(x, show_steps=False):
    result = x if x > 0 else 0
    steps = [f"ReLU({x}) = max(0, {x}) = {result}"]
    return _with_steps(result, steps, show_steps)


def leaky_relu(x, alpha=0.01, show_steps=False):
    result = x if x > 0 else alpha * x
    steps = [f"LeakyReLU({x}) = {result} (alpha={alpha})"]
    return _with_steps(result, steps, show_steps)


def tanh_activation(x, show_steps=False):
    result = math.tanh(x)
    steps = [f"tanh({x}) = {result}"]
    return _with_steps(result, steps, show_steps)



def softplus(x, show_steps=False):
    exp_val = math.exp(x)
    result = math.log(1 + exp_val)
    steps = [
        f"exp({x}) = {exp_val}",
        f"Softplus = ln(1 + {exp_val}) = {result}"
    ]
    return _with_steps(result, steps, show_steps)



def swish(x, show_steps=False):
    sig = 1 / (1 + math.exp(-x))
    result = x * sig
    steps = [
        f"Sigmoid({x}) = {sig}",
        f"Swish = {x} * {sig} = {result}"
    ]
    return _with_steps(result, steps, show_steps)



def gelu(x, show_steps=False):
    inner = math.sqrt(2/math.pi) * (x + 0.044715 * x**3)
    tanh_val = math.tanh(inner)
    result = 0.5 * x * (1 + tanh_val)
    steps = [
        f"Inner = {inner}",
        f"tanh(inner) = {tanh_val}",
        f"GELU = {result}"
    ]
    return _with_steps(result, steps, show_steps)



def softmax(values, show_steps=False):
    # ensure values is list (for multiple uses)
    vals = list(values)

    # calculate exponentials
    exp_vals = [math.exp(v) for v in vals]
    total = sum(exp_vals)

    # compute softmax outputs
    result = [ev / total for ev in exp_vals]

    # steps for show_steps
    steps = [
        f"Input values: {vals}",
        f"Exponentials: {exp_vals}",
        f"Sum of exponentials: {total}",
        f"Softmax output: {result}"
    ]

    return _with_steps(result, steps, show_steps)



# Make __all__ for neat imports
__all__ = [
    # number theory
    'comb','factorial','gcd','isqrt','lcm','perm',

    # float arithmetic
    'ceil','fabs','floor','fma','fmod','modf','remainder','trunc',

    # floating manipulation
    'copysign','frexp','isclose','isfinite','isinf','isnan','ldexp','nextafter','ulp',

    # powers and logs
    'cbrt','exp','exp2','expm1','log','log1p','log2','log10','pow','sqrt',

    # sums and prods
    'dist','fsum','hypot','prod','sumprod',

    # angles
    'degrees','radians',

    # trig
    'acos','asin','atan','atan2','cos','sin','tan',

    # hyperbolic
    'acosh','asinh','atanh','cosh','sinh','tanh',

    # special
    'erf','erfc','gamma','lgamma',

    # constants
    'pi','e','tau','inf','nan',

    # activation function
    'sigmoid','relu','leaky_relu','tanh_activation','softplus','swish','gelu','softmax'
]

