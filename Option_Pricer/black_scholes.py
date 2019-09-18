# Black-Scholes model for European call/put options.

from math import log, sqrt, exp
from scipy.stats import norm


class BlackScholes:
    def __init__(self, S, K, r, T, σ, t=0.0, q=0.0):
        self.__S, self.__K = S, K
        self.__r, self.__q = r, q
        self.__Δ = T - t

        self.__d1 = (log(S / K) + (r - q) * self.__Δ) / (σ * sqrt(self.__Δ)) + σ * sqrt(self.__Δ) / 2
        self.__d2 = (log(S / K) + (r - q) * self.__Δ) / (σ * sqrt(self.__Δ)) - σ * sqrt(self.__Δ) / 2

    def get_d1(self):
        return self.__d1

    def call(self):
        return self.__S * exp(-self.__q * self.__Δ) * norm.cdf(self.__d1) - \
               self.__K * exp(-self.__r * self.__Δ) * norm.cdf(self.__d2)

    def put(self):
        return self.__K * exp(-self.__r * self.__Δ) * norm.cdf(-self.__d2) - \
               self.__S * exp(-self.__q * self.__Δ) * norm.cdf(-self.__d1)
