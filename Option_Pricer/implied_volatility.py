import numpy as np
from math import sqrt, log, exp, pi
from black_scholes import BlackScholes


class NewtonRaphson:
    def __init__(self, S, K, r, T, q=0.0, type_='call', t=0.0, tol=1e-8, nmax=1000):
        self.__S, self.__K, self.__r = S, K, r
        self.__T, self.__t, self.__q = T, t, q
        self.__tol, self.__type, self.__nmax = tol, type_, nmax

        self.__σinit = self.__σhat()

    def __verify(self, V):
        if self.__type == 'call' and not (max(self.__S * exp(-self.__q * (self.__T - self.__t)) -
                                           self.__K * exp(-self.__r * (self.__T - self.__t)), 0) < V <
                                       self.__S * exp(-self.__q * (self.__T - self.__t))):
            raise ValueError("Arbitrage opportunity!")

        elif self.__type == 'put' and not (max(self.__K * exp(-self.__r * (self.__T - self.__t)) -
                                             self.__S * exp(-self.__q * (self.__T - self.__t)), 0) < V <
                                         self.__K * exp(-self.__r * (self.__T - self.__t))):
            raise ValueError("Arbitrage opportunity!")

    def __σhat(self):
        return sqrt(2 * abs((log(self.__S / self.__K) + (self.__r - self.__q) * (self.__T - self.__t))
                            / (self.__T - self.__t)))

    def __vega(self, d1):
        return self.__S * exp(-self.__q * (self.__T - self.__t)) * sqrt(self.__T - self.__t) * \
               exp(-d1 ** 2 / 2) / sqrt(2 * pi)

    def calc_σ(self, V):
        try:
            self.__verify(V)
        except ValueError:
            return np.NaN

        σ = self.__σinit
        n = 0
        σdiff = 1

        while σdiff >= self.__tol and n < self.__nmax:
            bs = BlackScholes(self.__S, self.__K, self.__r, self.__T, σ, self.__t, self.__q)
            c = bs.call() if self.__type == 'C' else bs.put()

            if c == V:
                return σ

            vega = self.__vega(bs.get_d1())
            if vega == 0:
                return np.NaN

            increment = (c - V) / vega
            σ -= increment
            n += 1
            σdiff = abs(increment)

        return σ
    
#impliedVolatility = NewtonRaphson(S=100, K=100, r=0.01, T=0.5, q=0.0, type_='put')
#print(impliedVolatility.calc_σ(5.3373))
