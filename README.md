# JAXBT: JAX Backtesting framework [![codecov](https://codecov.io/gh/richwomanbtc/jaxbt/branch/main/graph/badge.svg?token=P6VA2KK1RX)](https://codecov.io/gh/richwomanbtc/jaxbt)

WIP: Differentiable backtesting with JAX.

## Installation

```bash
pip install https://github.com/richwomanbtc/jaxbt.git
```

## Example

```python
import jax
import pandas as pd
from jaxbt.backtest import backtest_from_order_func, Backtest, OHLC, OrderType

pd = pd.DataFrame(
    columns=["timestamp", "open", "high", "low", "close"],
    data=[
        [1, 1.0, 1.0, 1.0, 1.0],
        [2, 2.0, 2.0, 2.0, 2.0],
        [3, 3.0, 3.0, 3.0, 3.0],
        [4, 4.0, 4.0, 4.0, 4.0],
        [5, 5.0, 5.0, 5.0, 5.0],
    ]
)

ohlc = OHLC.from_pandas(df)

@jax.jit
def f(bt: Backtest, idx: int):
    order_type = jax.lax.cond(
        bt.position[idx] == 0,
        lambda _: OrderType.MARKET_BUY, # if position is 0, perform market buy
        lambda _: OrderType.MARKET_SELL, # if position is not 0, perform market sell
        None
    )
    return order_type, 1., jnp.nan

result = backtest_from_order_func(ohlc, f)

@jax.jit
def f_param(params: jax.Array, bt: Backtest, idx: int):
    order_type = jax.lax.cond(
        bt.position[idx] == 0,
        lambda _: OrderType.MARKET_BUY, # if position is 0, perform market buy
        lambda _: OrderType.MARKET_SELL, # if position is not 0, perform market sell
        None
    )
    return order_type, params[0], jnp.nan

@jax.jit
def loss(param: jax.Array):
    result = backtest_from_order_func(
        df, lambda bt, idx: f_param(param, bt, idx)
    )
    return -result.pl.sum()

grad_fun = jax.grad(loss, argnums=0)

@jax.jit
def train(epoch, params, lr=0.01):
    def body_fun(idx, params):
        grads = grad_fun(params)
        params = params - lr * grads
        return params

    params = jax.lax.fori_loop(0, epoch, body_fun, params)
    return params

init_params = jnp.array([0.1])
result_params = train(100, init_params)
print(loss(init_params), loss(result_params))
```

- If you use conditional branches, you have to use [jax.lax.cond](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html) or [jax.lax.switch](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html) instead of if-else.
- When debugging, you can use [jax.disable_jit](https://jax.readthedocs.io/en/latest/_autosummary/jax.disable_jit.html) to disable jit compilation.