# JAXBT: JAX Backtesting framework

```python
import jax
import pandas as pd
from jaxbt.backtest import backtest_from_order_func, Backtest, OHLC

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
```

- If you use conditional branches, you have to use [jax.lax.cond](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html) or [jax.lax.switch](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html) instead of if-else.
- When debugging, you can use [jax.disable_jit](https://jax.readthedocs.io/en/latest/_autosummary/jax.disable_jit.html) to disable jit compilation.