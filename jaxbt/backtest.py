import numpy as np
from jax import numpy as jnp
import jax
from dataclasses import dataclass
import pandas as pd
from typing import Callable, Optional, Tuple, Union
from functools import partial
from jax.tree_util import register_pytree_node_class
from enum import IntEnum

DataType = Union[pd.Series, jax.Array, np.ndarray]


class OrderType(IntEnum):
    MARKET_BUY = 0
    MARKET_SELL = 1
    LIMIT_BUY = 2
    LIMIT_SELL = 3


# INDEX OF ACTION
class ActionIndex(IntEnum):
    ORDER_TYPE = 0
    SIZE = 1
    PRICE = 2


# INDEX OF OHLC
class OHLCIndex(IntEnum):
    TIMESTAMP = 0
    OPEN = 1
    HIGH = 2
    LOW = 3
    CLOSE = 4


@dataclass
class OHLC:
    timestamp: jax.Array
    open: jax.Array
    high: jax.Array
    low: jax.Array
    close: jax.Array

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "OHLC":
        """Create OHLC from pandas.DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            "timestamp", "open", "high", "low", "close" columns are required.

        Returns
        -------
        OHLC
            OHLC data class.
        """
        return cls(
            timestamp=jnp.array(df["timestamp"]),
            open=jnp.array(df["open"]),
            high=jnp.array(df["high"]),
            low=jnp.array(df["low"]),
            close=jnp.array(df["close"]),
        )

    @classmethod
    def from_ndarray(cls, arr: Union[np.ndarray, jax.Array]) -> "OHLC":
        """Create OHLC from numpy.ndarray or jax.Array.

        Parameters
        ----------
        arr : Union[np.ndarray, jax.Array]
            (n, 5) array. 5 columns are required.

        Returns
        -------
        OHLC
            OHLC data class.
        """
        return cls(
            timestamp=jnp.array(arr[:, OHLCIndex.TIMESTAMP]),
            open=jnp.array(arr[:, OHLCIndex.OPEN]),
            high=jnp.array(arr[:, OHLCIndex.HIGH]),
            low=jnp.array(arr[:, OHLCIndex.LOW]),
            close=jnp.array(arr[:, OHLCIndex.CLOSE]),
        )


@dataclass
class BacktestResult:
    timestamp: jax.Array
    pl: jax.Array
    position: jax.Array


@register_pytree_node_class
class Backtest:
    def __init__(
        self,
        timestamp: jax.Array,
        open: jax.Array,
        high: jax.Array,
        low: jax.Array,
        close: jax.Array,
        pl: jax.Array,
        posision: jax.Array,
        features: jax.Array,
    ):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.pl = pl
        self.position = posision
        self.features = features

    def tree_flatten(self):
        return (
            (
                self.timestamp,
                self.open,
                self.high,
                self.low,
                self.close,
                self.pl,
                self.position,
                self.features,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.jit
def market_buy(
    prev_position: float,
    action: Tuple[int, float, float],
    *ohlc,
):
    return prev_position + action[ActionIndex.SIZE]


@jax.jit
def market_sell(
    prev_position: float,
    action: Tuple[int, float, float],
    *ohlc,
):
    return prev_position - action[ActionIndex.SIZE]


@jax.jit
def limit_buy(
    prev_position: float,
    action: Tuple[int, float, float],
    *ohlc,
):
    return jax.lax.cond(
        action[ActionIndex.PRICE] <= ohlc[OHLCIndex.HIGH],
        lambda _: prev_position + action[ActionIndex.SIZE],
        lambda _: prev_position,
        None,
    )


@jax.jit
def limit_sell(
    prev_position: float,
    action: Tuple[int, float, float],
    *ohlc,
):

    return jax.lax.cond(
        action[ActionIndex.PRICE] >= ohlc[OHLCIndex.LOW],
        lambda _: prev_position - action[ActionIndex.SIZE],
        lambda _: prev_position,
        None,
    )


def initialize_array(arr_like: jax.Array, init_value: jax.Array):
    arr = jnp.full_like(arr_like, jnp.nan, dtype=jnp.float32)
    arr = arr.at[0].set(init_value)
    return arr


@partial(jax.jit, static_argnums=5)
def _backtest(
    timestamp: jax.Array,
    open: jax.Array,
    high: jax.Array,
    low: jax.Array,
    close: jax.Array,
    f: Callable[[Backtest, int], Tuple[int, float, float]],
    features: jax.Array,
):

    # main process
    # TODO: jax.lax.foriloop for multiple actions
    def body(bt: Backtest, i):
        action = f(bt, i - 1)
        bt.open = bt.open.at[i].set(open[i])
        bt.high = bt.high.at[i].set(high[i])
        bt.low = bt.low.at[i].set(low[i])
        bt.close = bt.close.at[i].set(close[i])
        bt.features = bt.features.at[i].set(features[i])
        next_position = jax.lax.switch(
            action[0],
            [
                market_buy,
                market_sell,
                limit_buy,
                limit_sell,
            ],
            bt.position[i - 1],
            action,
            timestamp[i],
            open[i],
            high[i],
            low[i],
            close[i],
        )
        bt.position = bt.position.at[i].set(next_position)

        bt.pl = bt.pl.at[i].set((close[i] / close[i - 1] - 1) * bt.position[i])

        return bt, i + 1

    # initialize
    pl = initialize_array(timestamp, jnp.array(0.0))
    position = initialize_array(timestamp, jnp.array(0.0))
    features_ = initialize_array(features, features[0])
    open_ = initialize_array(open, open[0])
    high_ = initialize_array(high, high[0])
    low_ = initialize_array(low, low[0])
    close_ = initialize_array(close, close[0])

    bt = Backtest(timestamp, open_, high_, low_, close_, pl, position, features_)

    # loop
    result, _ = jax.lax.scan(body, bt, jnp.arange(1, len(timestamp)))

    return (
        result.timestamp,
        result.pl,
        result.position,
    )


def backtest_from_order_func(
    price: Union[OHLC, pd.DataFrame, np.ndarray, jax.Array],
    order_func: Callable[[Backtest, int], Tuple[int, float, float]],
    features: Optional[Union[pd.DataFrame, np.ndarray, jax.Array]] = None,
) -> BacktestResult:
    match price:
        case pd.DataFrame():
            _ohlc = OHLC.from_pandas(price)
        case np.ndarray(), jax.Array():
            _ohlc = OHLC.from_ndarray(price)
        case OHLC():
            _ohlc = price
        case _:
            raise TypeError(
                "price type must be one of OHLC, pd.DataFrame, np.ndarray or jax.Array"
            )

    match features:
        case pd.DataFrame():
            features = jnp.array(features.values)
        case np.ndarray():
            features = jnp.array(features)
        case jax.Array():
            features = features
        case None:
            features = jnp.full_like(_ohlc.timestamp, jnp.nan, dtype=jnp.float32)
        case _:
            raise TypeError(
                "features type must be one of pd.DataFrame, np.ndarray, jax.Array,\
                or None"
            )
    result = _backtest(
        _ohlc.timestamp,
        _ohlc.open,
        _ohlc.high,
        _ohlc.low,
        _ohlc.close,
        order_func,
        features,
    )

    return BacktestResult(*result)


def backtest_from_signal(
    price: Union[OHLC, pd.DataFrame, np.ndarray, jax.Array],
    signal: DataType,
) -> BacktestResult:
    match signal:
        case pd.Series():
            assert signal.dtype == jnp.int32
            signal = jnp.array(signal.values)
        case np.ndarray():
            assert signal.dtype == np.int64
            signal = jnp.array(signal, dtype=jnp.int32)
        case jax.Array():
            assert signal.dtype == jnp.int32
        case _:
            raise TypeError(
                "signal type must be one of pd.Series, np.ndarray or jax.Array"
            )

    @jax.jit
    def order_func(bt: Backtest, idx: int) -> Tuple[int, float, float]:
        return signal[idx], 1.0, jnp.nan  # type: ignore

    return backtest_from_order_func(price, order_func)


def summary(backtest_result: BacktestResult):
    df = pd.DataFrame(
        {"timestamp": backtest_result.timestamp, "pl": backtest_result.pl}
    )
    df.set_index("timestamp", inplace=True)
    return df.describe()
