import math

import matplotlib.pyplot as plt
from scipy.stats import norm


def N(d):
    return norm.cdf(d)


def N1(d):
    return math.exp(-(d**2)/2) / math.sqrt(2*math.pi)


def call_price(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    C = S * math.exp(-d * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    return C


def put_price(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    C = K * math.exp(-r * T) * N(-d2) - S * math.exp(-d * T) * N(-d1)
    return C


def call_delta(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    return N(d1)


def put_delta(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    return -N(-d1)


def gamma(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    return (math.exp(-d * T) / (S * v * math.sqrt(T))) * (1 / math.sqrt(2 * math.pi)) * (math.exp((-d1 ** 2) / 2))


def call_theta(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)

    theta = call_price(S, K, T-1/365, v, d, r) - call_price(S, K, T, v, d, r)

    # theta = (S*N1(d1)*v) / (2*math.sqrt(T)) - (r*K*math.exp(-r*T)*N(d2)) / 365

    return theta


def put_theta(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    # theta = 1 / T * (-(
    #             S * v * math.exp(-d * T) * math.exp((-d1 ** 2) / 2) / (2 * math.sqrt(T)) / (math.sqrt(2 * math.pi))) + (
    #                              r * K * math.exp(-r * T) * -N(-d2)) + (d * S * math.exp(-d * T) * -N(-d1))) / 365

    theta = put_price(S, K, T-1/365, v, d, r) - put_price(S, K, T, v, d, r)

    return theta


def call_vega(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    vega = S * math.sqrt(T) * N1(d1) * math.exp(-d*T) * .01

    return vega


def put_vega(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    vega = S * math.sqrt(T) * N1(d1) * math.exp(-d * T) * .01

    return vega


def call_rho(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    rho = K * T * math.exp(-r*T) * N(d2) * 0.01
    
    return rho


def put_rho(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    rho = -(K * T * math.exp(-r * T) * N(-d2) * 0.01)

    return rho


def call_psi(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    psi = -T*S*math.exp(-d*T)*N(d1) * 0.01

    return psi


def put_psi(S, K, T, v, d, r):
    d1 = (math.log(S / K, math.e) + T * (r - d + (v ** 2) / 2)) / (v * math.sqrt(T))
    psi = T * S * math.exp(-d * T) * N(-d1) * 0.01

    return psi


def call_elasticity(S, K, T, v, d, r):
    delta = call_delta(S, K, T, v, d, r)
    C = call_price(S, K, T, v, d, r)

    elasticity = (S * delta) / C
    return elasticity


def put_elasticity(S, K, T, v, d, r):
    delta = put_delta(S, K, T, v, d, r)
    P = put_price(S, K, T, v, d, r)

    elasticity = (S * delta) / P
    return elasticity


def graph_q1():
    k1 = 40
    k2 = 60
    S = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T = 6 / 12

    calls = []
    puts = []
    strikes = [k for k in range(k1, k2 + 1)]

    # calls
    for K in range(k1, k2 + 1):
        call = call_price(S, K, T, v, d, r)
        calls.append(call)
        # print(call)

    # puts
    for K in range(k1, k2 + 1):
        put = put_price(S, K, T, v, d, r)
        puts.append(put)
        # print(put)

    plt.plot(strikes, calls, label='Call Price', color='blue')
    plt.plot(strikes, puts, label='Put Price', color='red')

    plt.locator_params(axis="x", nbins=len(strikes) * 2)

    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.legend()

    plt.show()


def graph_q2_1():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph delta as function of stock price for T1, T2, T3
    T1_deltas = []
    T2_deltas = []
    T3_deltas = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = call_delta(S, K, T1, v, d, r)
        T1_deltas.append(T1d)
        T2d = call_delta(S, K, T2, v, d, r)
        T2_deltas.append(T2d)
        T3d = call_delta(S, K, T3, v, d, r)
        T3_deltas.append(T3d)

    plt.plot(prices, T1_deltas, label='Deltas for T1', color='blue')
    plt.plot(prices, T2_deltas, label='Deltas for T2', color='red')
    plt.plot(prices, T3_deltas, label='Deltas for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Call Option Delta')
    plt.legend()

    plt.show()


def graph_q2_2():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph delta as function of stock price for T1, T2, T3
    T1_deltas = []
    T2_deltas = []
    T3_deltas = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = put_delta(S, K, T1, v, d, r)
        T1_deltas.append(T1d)
        T2d = put_delta(S, K, T2, v, d, r)
        T2_deltas.append(T2d)
        T3d = put_delta(S, K, T3, v, d, r)
        T3_deltas.append(T3d)

    plt.plot(prices, T1_deltas, label='Deltas for T1', color='blue')
    plt.plot(prices, T2_deltas, label='Deltas for T2', color='red')
    plt.plot(prices, T3_deltas, label='Deltas for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Put Option Delta')
    plt.legend()

    plt.show()


def graph_q3():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph gamma as function of stock price for T1, T2, T3
    T1_gammas = []
    T2_gammas = []
    T3_gammas = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = gamma(S, K, T1, v, d, r)
        T1_gammas.append(T1d)
        T2d = gamma(S, K, T2, v, d, r)
        T2_gammas.append(T2d)
        T3d = gamma(S, K, T3, v, d, r)
        T3_gammas.append(T3d)

    plt.plot(prices, T1_gammas, label='Gammas for T1', color='blue')
    plt.plot(prices, T2_gammas, label='Gammas for T2', color='red')
    plt.plot(prices, T3_gammas, label='Gammas for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Call & Put Option Gamma')
    plt.legend()

    plt.show()


def graph_q4_1():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph theta as function of stock price for T1, T2, T3
    T1_thetas = []
    T2_thetas = []
    T3_thetas = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = call_theta(S, K, T1, v, d, r)
        T1_thetas.append(T1d)
        T2d = call_theta(S, K, T2, v, d, r)
        T2_thetas.append(T2d)
        T3d = call_theta(S, K, T3, v, d, r)
        T3_thetas.append(T3d)

    plt.plot(prices, T1_thetas, label='Thetas for T1', color='blue')
    plt.plot(prices, T2_thetas, label='Thetas for T2', color='red')
    plt.plot(prices, T3_thetas, label='Thetas for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Call Option Theta')
    plt.legend()

    plt.show()


def graph_q4_2():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph theta as function of stock price for T1, T2, T3
    T1_thetas = []
    T2_thetas = []
    T3_thetas = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = put_theta(S, K, T1, v, d, r)
        T1_thetas.append(T1d)
        T2d = put_theta(S, K, T2, v, d, r)
        T2_thetas.append(T2d)
        T3d = put_theta(S, K, T3, v, d, r)
        T3_thetas.append(T3d)

    plt.plot(prices, T1_thetas, label='Thetas for T1', color='blue')
    plt.plot(prices, T2_thetas, label='Thetas for T2', color='red')
    plt.plot(prices, T3_thetas, label='Thetas for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Put Option Theta')
    plt.legend()

    plt.show()


def graph_q5_1():
    S = 50
    K = 50
    a = .15
    # v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    v1 = 5
    v2 = 50

    # graph stock price as function of volatility for T1, T2, T3
    T1_prices = []
    T2_prices = []
    T3_prices = []
    prices = [S for S in range(v1, v2 + 1)]

    for v in range(v1, v2 + 1):
        v /= 100  # convert to decimal
        T1d = call_price(S, K, T1, v, d, r)
        T1_prices.append(T1d)
        T2d = call_price(S, K, T2, v, d, r)
        T2_prices.append(T2d)
        T3d = call_price(S, K, T3, v, d, r)
        T3_prices.append(T3d)

    plt.plot(prices, T1_prices, label='Prices for T1', color='blue')
    plt.plot(prices, T2_prices, label='Prices for T2', color='red')
    plt.plot(prices, T3_prices, label='Prices for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Volatility')
    plt.ylabel('Call Option Prices')
    plt.legend()

    plt.show()


def graph_q5_2():
    S = 50
    K = 50
    a = .15
    # v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    v1 = 5
    v2 = 50

    # graph stock price as function of volatility for T1, T2, T3
    T1_prices = []
    T2_prices = []
    T3_prices = []
    prices = [S for S in range(v1, v2 + 1)]

    for v in range(v1, v2 + 1):
        v /= 100  # convert to decimal
        T1d = put_price(S, K, T1, v, d, r)
        T1_prices.append(T1d)
        T2d = put_price(S, K, T2, v, d, r)
        T2_prices.append(T2d)
        T3d = put_price(S, K, T3, v, d, r)
        T3_prices.append(T3d)

    plt.plot(prices, T1_prices, label='Prices for T1', color='blue')
    plt.plot(prices, T2_prices, label='Prices for T2', color='red')
    plt.plot(prices, T3_prices, label='Prices for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Volatility')
    plt.ylabel('Put Option Prices')
    plt.legend()

    plt.show()


def graph_q6_1():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph vega as function of stock price for T1, T2, T3
    T1_vegas = []
    T2_vegas = []
    T3_vegas = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = call_vega(S, K, T1, v, d, r)
        T1_vegas.append(T1d)
        T2d = call_vega(S, K, T2, v, d, r)
        T2_vegas.append(T2d)
        T3d = call_vega(S, K, T3, v, d, r)
        T3_vegas.append(T3d)

    plt.plot(prices, T1_vegas, label='Vegas for T1', color='blue')
    plt.plot(prices, T2_vegas, label='Vegas for T2', color='red')
    plt.plot(prices, T3_vegas, label='Vegas for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Call Option Vegas')
    plt.legend()

    plt.show()


def graph_q6_2():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph vega as function of stock price for T1, T2, T3
    T1_vegas = []
    T2_vegas = []
    T3_vegas = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = put_vega(S, K, T1, v, d, r)
        T1_vegas.append(T1d)
        T2d = put_vega(S, K, T2, v, d, r)
        T2_vegas.append(T2d)
        T3d = put_vega(S, K, T3, v, d, r)
        T3_vegas.append(T3d)

    plt.plot(prices, T1_vegas, label='Vegas for T1', color='blue')
    plt.plot(prices, T2_vegas, label='Vegas for T2', color='red')
    plt.plot(prices, T3_vegas, label='Vegas for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Put Option Vegas')
    plt.legend()

    plt.show()


def graph_q7_1():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph rho as function of stock price for T1, T2, T3
    T1_rhos = []
    T2_rhos = []
    T3_rhos = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = call_rho(S, K, T1, v, d, r)
        T1_rhos.append(T1d)
        T2d = call_rho(S, K, T2, v, d, r)
        T2_rhos.append(T2d)
        T3d = call_rho(S, K, T3, v, d, r)
        T3_rhos.append(T3d)

    plt.plot(prices, T1_rhos, label='Rhos for T1', color='blue')
    plt.plot(prices, T2_rhos, label='Rhos for T2', color='red')
    plt.plot(prices, T3_rhos, label='Rhos for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Call Option Rhos')
    plt.legend()

    plt.show()


def graph_q7_2():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph rho as function of stock price for T1, T2, T3
    T1_rhos = []
    T2_rhos = []
    T3_rhos = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = put_rho(S, K, T1, v, d, r)
        T1_rhos.append(T1d)
        T2d = put_rho(S, K, T2, v, d, r)
        T2_rhos.append(T2d)
        T3d = put_rho(S, K, T3, v, d, r)
        T3_rhos.append(T3d)

    plt.plot(prices, T1_rhos, label='Rhos for T1', color='blue')
    plt.plot(prices, T2_rhos, label='Rhos for T2', color='red')
    plt.plot(prices, T3_rhos, label='Rhos for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Put Option Rhos')
    plt.legend()

    plt.show()


def graph_q8_1():
    S = 50
    K = 50
    a = .15
    v = .30
    # d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    d1 = 0
    d2 = 4

    # graph stock price as function of dividend yield for T1, T2, T3
    T1_prices = []
    T2_prices = []
    T3_prices = []
    prices = [S for S in range(d1, d2 + 1)]

    for d in range(d1, d2 + 1):
        d /= 100  # convert to decimal
        T1d = call_price(S, K, T1, v, d, r)
        T1_prices.append(T1d)
        T2d = call_price(S, K, T2, v, d, r)
        T2_prices.append(T2d)
        T3d = call_price(S, K, T3, v, d, r)
        T3_prices.append(T3d)

    plt.plot(prices, T1_prices, label='Prices for T1', color='blue')
    plt.plot(prices, T2_prices, label='Prices for T2', color='red')
    plt.plot(prices, T3_prices, label='Prices for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Dividend Yield %')
    plt.ylabel('Call Option Prices')
    plt.legend()

    plt.show()


def graph_q8_2():
    S = 50
    K = 50
    a = .15
    v = .30
    # d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    d1 = 0
    d2 = 4

    # graph stock price as function of dividend yield for T1, T2, T3
    T1_prices = []
    T2_prices = []
    T3_prices = []
    prices = [S for S in range(d1, d2 + 1)]

    for d in range(d1, d2 + 1):
        d /= 100  # convert to decimal
        T1d = put_price(S, K, T1, v, d, r)
        T1_prices.append(T1d)
        T2d = put_price(S, K, T2, v, d, r)
        T2_prices.append(T2d)
        T3d = put_price(S, K, T3, v, d, r)
        T3_prices.append(T3d)

    plt.plot(prices, T1_prices, label='Prices for T1', color='blue')
    plt.plot(prices, T2_prices, label='Prices for T2', color='red')
    plt.plot(prices, T3_prices, label='Prices for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Dividend Yield %')
    plt.ylabel('Put Option Prices')
    plt.legend()

    plt.show()


def graph_q9_1():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph psi as function of stock price for T1, T2, T3
    T1_psis = []
    T2_psis = []
    T3_psis = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = call_psi(S, K, T1, v, d, r)
        T1_psis.append(T1d)
        T2d = call_psi(S, K, T2, v, d, r)
        T2_psis.append(T2d)
        T3d = call_psi(S, K, T3, v, d, r)
        T3_psis.append(T3d)

    plt.plot(prices, T1_psis, label='Psis for T1', color='blue')
    plt.plot(prices, T2_psis, label='Psis for T2', color='red')
    plt.plot(prices, T3_psis, label='Psis for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Call Option Psis')
    plt.legend()

    plt.show()


def graph_q9_2():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph psi as function of stock price for T1, T2, T3
    T1_psis = []
    T2_psis = []
    T3_psis = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = put_psi(S, K, T1, v, d, r)
        T1_psis.append(T1d)
        T2d = put_psi(S, K, T2, v, d, r)
        T2_psis.append(T2d)
        T3d = put_psi(S, K, T3, v, d, r)
        T3_psis.append(T3d)

    plt.plot(prices, T1_psis, label='Psis for T1', color='blue')
    plt.plot(prices, T2_psis, label='Psis for T2', color='red')
    plt.plot(prices, T3_psis, label='Psis for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Put Option Psis')
    plt.legend()

    plt.show()


def graph_q10_1():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph elasticity as function of stock price for T1, T2, T3
    T1_elasticities = []
    T2_elasticities = []
    T3_elasticities = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = call_elasticity(S, K, T1, v, d, r)
        T1_elasticities.append(T1d)
        T2d = call_elasticity(S, K, T2, v, d, r)
        T2_elasticities.append(T2d)
        T3d = call_elasticity(S, K, T3, v, d, r)
        T3_elasticities.append(T3d)

    plt.plot(prices, T1_elasticities, label='Elasticities for T1', color='blue')
    plt.plot(prices, T2_elasticities, label='Elasticities for T2', color='red')
    plt.plot(prices, T3_elasticities, label='Elasticities for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Call Option Elasticities')
    plt.legend()

    plt.show()


def graph_q10_2():
    K = 50
    a = .15
    v = .30
    d = .02
    r = .05
    T1 = 1 / 12
    T2 = 1
    T3 = 2

    s1 = 20
    s2 = 100

    # graph elasticity as function of stock price for T1, T2, T3
    T1_elasticities = []
    T2_elasticities = []
    T3_elasticities = []
    prices = [S for S in range(s1, s2 + 1)]

    for S in range(s1, s2 + 1):
        T1d = put_elasticity(S, K, T1, v, d, r)
        T1_elasticities.append(T1d)
        T2d = put_elasticity(S, K, T2, v, d, r)
        T2_elasticities.append(T2d)
        T3d = put_elasticity(S, K, T3, v, d, r)
        T3_elasticities.append(T3d)

    plt.plot(prices, T1_elasticities, label='Elasticities for T1', color='blue')
    plt.plot(prices, T2_elasticities, label='Elasticities for T2', color='red')
    plt.plot(prices, T3_elasticities, label='Elasticities for T3', color='green')

    plt.locator_params(axis="x", nbins=len(prices))

    plt.xlabel('Stock Price')
    plt.ylabel('Put Option Elasticities')
    plt.legend()

    plt.show()


def main():
    graph_q1()
    graph_q2_1()
    graph_q2_2()
    graph_q3()
    graph_q4_1()
    graph_q4_2()
    graph_q5_1()
    graph_q5_2()
    graph_q6_1()
    graph_q6_2()
    graph_q7_1()
    graph_q7_2()
    graph_q8_1()
    graph_q8_2()
    graph_q9_1()
    graph_q9_2()
    graph_q10_1()
    graph_q10_2()


if __name__ == '__main__':
    main()
