from jaxbt.backtest import (
    backtest_from_signal,
    backtest_from_order_func,
    OHLC,
    Backtest,
    OrderType,
    summary,
)
import pytest
import jax.numpy as jnp
import pandas as pd
import jax


@pytest.fixture(scope="module")
def ohlc():
    return OHLC(
        timestamp=jnp.array([1, 2, 3, 4, 5]),
        open=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        high=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        low=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        close=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


@pytest.fixture(scope="module")
def signal():
    return jnp.array([0, 1, 0, 1, 0])


@pytest.fixture(scope="module")
def features():
    return jnp.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])


def test_ohlcv(ohlc):
    assert isinstance(ohlc, OHLC)
    assert jnp.array_equal(ohlc.timestamp, jnp.array([1, 2, 3, 4, 5]))
    assert jnp.array_equal(ohlc.open, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert jnp.array_equal(ohlc.high, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert jnp.array_equal(ohlc.low, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert jnp.array_equal(ohlc.close, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_from_pandas(ohlc):
    ohlc_from_pandas = OHLC.from_pandas(
        pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close"],
            data=[
                [1, 1.0, 1.0, 1.0, 1.0],
                [2, 2.0, 2.0, 2.0, 2.0],
                [3, 3.0, 3.0, 3.0, 3.0],
                [4, 4.0, 4.0, 4.0, 4.0],
                [5, 5.0, 5.0, 5.0, 5.0],
            ],
        )
    )
    assert jnp.array_equal(ohlc_from_pandas.timestamp, ohlc.timestamp)
    assert jnp.array_equal(ohlc_from_pandas.open, ohlc.open)
    assert jnp.array_equal(ohlc_from_pandas.high, ohlc.high)
    assert jnp.array_equal(ohlc_from_pandas.low, ohlc.low)
    assert jnp.array_equal(ohlc_from_pandas.close, ohlc.close)


def test_backtest_from_signal(ohlc, signal):
    result = backtest_from_signal(ohlc, signal)
    assert jnp.array_equal(
        result.timestamp,
        jnp.array([1, 2, 3, 4, 5]),
    )
    assert jnp.allclose(
        result.pl,
        jnp.array([0.0, 1.0, 0, 4.0 / 3.0 - 1.0, 0.0]),
        rtol=1e-5,
    )
    assert jnp.allclose(
        result.position,
        jnp.array([0, 1, 0, 1, 0]),
        rtol=1e-5,
    )


def test_backtest_from_market_order_func(ohlc, features):
    @jax.jit
    def f(bt: Backtest, idx: int):
        s = jax.lax.cond(
            bt.position[idx] == 0,
            lambda _: OrderType.MARKET_BUY,  # if position is 0 market buy
            lambda _: OrderType.MARKET_SELL,  # else market sell
            None,
        )
        return s, 1.0, jnp.nan

    with jax.disable_jit():
        result = backtest_from_order_func(ohlc, f, features)
        assert jnp.allclose(
            result.pl,
            jnp.array([0.0, 1.0, 0, 4.0 / 3.0 - 1.0, 0.0]),
            rtol=1e-5,
        )
        assert jnp.allclose(
            result.position,
            jnp.array([0, 1, 0, 1, 0]),
            rtol=1e-5,
        )


def test_backtest_from_limit_order_func(ohlc, features):
    @jax.jit
    def f(bt: Backtest, idx: int):
        s = jax.lax.cond(
            bt.position[idx] == 0,
            lambda _: OrderType.LIMIT_BUY,  # if position is 0 limit buy
            lambda _: OrderType.LIMIT_SELL,  # else limit sell
            None,
        )
        return s, 1.0, 2.0

    result = backtest_from_order_func(ohlc, f, features)
    assert jnp.allclose(
        result.pl,
        jnp.array([0.0, 1.0, 0, 0.0, 0.0]),
        rtol=1e-5,
    )
    assert jnp.allclose(
        result.position,
        jnp.array([0, 1, 0, 0, 0]),
        rtol=1e-5,
    )

    print(summary(result))


def test_backtest_autograd(ohlc, features):
    @jax.jit
    def f(param: jax.Array, bt: Backtest, idx: int):
        s = jax.lax.cond(
            bt.position[idx] == 0,
            lambda _: OrderType.MARKET_BUY,  # if position is 0 limit buy
            lambda _: OrderType.MARKET_SELL,  # else limit sell
            None,
        )

        return s, param[0], jnp.nan

    @jax.jit
    def loss(param: jax.Array):
        result = backtest_from_order_func(
            ohlc, lambda bt, idx: f(param, bt, idx), features
        )
        return -result.pl.sum()

    grad_fun = jax.grad(loss, argnums=0)
    # grad should be -4.0 / 3.0 independent of param in this case
    assert abs(grad_fun(jnp.array([1.0])) - jnp.array([-4.0 / 3.0])) < 1.0e-5

    @jax.jit
    def train(epoch, params, lr=0.01):
        def body_fun(idx, params):
            grads = grad_fun(params)
            params = params - lr * grads
            return params

        params = jax.lax.fori_loop(0, epoch, body_fun, params)
        return params

    init_params = jnp.array([1.0])
    result_params = train(100, init_params)
    # resulted param should be init value + 100 * grad
    assert jnp.allclose(result_params, jnp.array([1.0 + 4.0 / 3.0]), rtol=1e-5)
    # loss should be init value (=1.0 + 4.0 / 3.0) * 100 * grad
    assert abs(-loss(result_params) - (1.0 + 4.0 / 3.0) * 4.0 / 3.0) < 1.0e-5


def test_backtest_autograd_limit(ohlc, features):
    @jax.jit
    def f(param: jax.Array, bt: Backtest, idx: int):
        s = jax.lax.cond(
            bt.position[idx] == 0,
            lambda _: OrderType.LIMIT_BUY,  # if position is 0 limit buy
            lambda _: OrderType.LIMIT_BUY,  # else limit buy
            None,
        )
        return s, 1, bt.close[idx] + param[0]

    @jax.jit
    def loss(param: jax.Array):
        result = backtest_from_order_func(
            ohlc, lambda bt, idx: f(param, bt, idx), features
        )
        return -result.pl.sum()

    grad_fun = jax.grad(loss, argnums=0)
    print(grad_fun(jnp.array([5.0])))
    # assert abs(grad_fun(jnp.array([3.0])) - jnp.array([-4.0 / 3.0])) < 1.0e-5

    # @jax.jit
    # def train(epoch, params, lr=0.01):
    #     def body_fun(idx, params):
    #         grads = grad_fun(params)
    #         params = params - lr * grads
    #         return params

    #     params = jax.lax.fori_loop(0, epoch, body_fun, params)
    #     return params

    # init_params = jnp.array([1.0])
    # result_params = train(100, init_params)
    # # resulted param should be init value + 100 * grad
    # assert jnp.allclose(result_params, jnp.array([1.0 + 4.0 / 3.0]), rtol=1e-5)
    # # loss should be init value (=1.0 + 4.0 / 3.0) * 100 * grad
    # assert abs(-loss(result_params) - (1.0 + 4.0 / 3.0) * 4.0 / 3.0) < 1.0e-5
