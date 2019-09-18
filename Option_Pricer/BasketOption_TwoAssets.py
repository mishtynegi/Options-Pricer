# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from scipy.stats import norm


class BasketOption:
    def __init__(self, S, K, r, T, σ, ρ, n=50, option_type="call", m=int(1e5), t=0.0):
        self.__S, self.__K, self.__r = S, K, r
        self.__σ, self.__ρ, self.__Δ = σ, ρ, T - t
        self.__n, self.__m, self.__dt = n, m, T / n
        self.__option_type = option_type.lower()

        self.__S_path = []

    def __gen_paths(self):
        if not len(self.__S_path):
            drift = np.exp((self.__r - np.array([self.__σ[i] ** 2 for i in range(len(self.__σ))]) / 2) * self.__dt)
            np.random.seed(52)
            randn1 = np.random.randn(self.__m, self.__n)
            np.random.seed(42)
            randn2 = np.random.randn(self.__m, self.__n)    
            randn3 = np.add(self.__ρ[0][1] * randn1 , np.sqrt(1-self.__ρ[0][1]**2) * randn2) 
            Z = [randn1,randn3]
            for i in range(len(self.__S)):
                # TODO: add correlation for Z       
                growth_factor = drift[i] * np.exp(np.array(self.__σ[i]) *
                                                  np.sqrt(self.__dt) * Z[i])
                self.__S_path.append(self.__S[i] * np.cumprod(growth_factor, 1))

    def __gen_geo_payoff(self):
        self.__gen_paths()
        # first mean by number of steps in simulation, second mean by number of underlying assets
        # output: array of size = number of simulated paths
        #geo_mean = np.exp(np.sum(np.log(self.__S_path), axis=2) / self.__n)
        geo_mean = np.exp(np.sum(np.log(self.__S_path), axis=0) / len(self.__S_path))
        if self.__option_type == 'call':
            self.__geo_payoff = np.exp(-self.__r * self.__Δ) * np.maximum(geo_mean[:,-1] - self.__K, 0)
        elif self.__option_type == 'put':
            self.__geo_payoff = np.exp(-self.__r * self.__Δ) * np.maximum(self.__K - geo_mean[:,-1], 0)

    def __gen_arith_payoff(self):
        self.__gen_paths()
        # in the same way as geometric payoff
        arith_mean = np.mean(self.__S_path, axis=0)
        if self.__option_type == 'call':
            self.__arith_payoff = np.exp(-self.__r * self.__Δ) * np.maximum(arith_mean[:,-1] - self.__K, 0)
        elif self.__option_type == 'put':
            self.__arith_payoff = np.exp(-self.__r * self.__Δ) * np.maximum(self.__K - arith_mean[:,-1], 0)

    def __θ(self):
        covXY = np.mean(self.__arith_payoff * self.__geo_payoff) - \
                np.mean(self.__arith_payoff) * np.mean(self.__geo_payoff)
        return covXY / np.var(self.__geo_payoff)

    def control_variate(self):
        self.__gen_geo_payoff()
        self.__gen_arith_payoff()

        Z = self.__arith_payoff + self.__θ() * (self.closed_form() - self.__geo_payoff)
        Zmean = np.mean(Z)
        Zstd = np.std(Z)
        # confidence interval
        confcv = [Zmean - 1.96 * Zstd / np.sqrt(self.__m), Zmean + 1.96 * Zstd / np.sqrt(self.__m)]

        return Zmean, confcv

    def closed_form(self):
        B0 = np.exp(np.sum(np.log(self.__S), axis=0) / len(self.__S))

        σB_sqΔ = self.__Δ * sum([self.__ρ[i][j] * self.__σ[i] * self.__σ[j]
                                 for i in range(len(self.__σ)) for j in range(len(self.__σ))]) / (len(self.__σ) ** 2)
        μBΔ = self.__Δ * (self.__r - sum([self.__σ[i] ** 2 for i in range(len(self.__σ))]) /
                          (2 * len(self.__σ))) + σB_sqΔ / 2

        d1hat = (np.log(B0 / self.__K) + μBΔ + σB_sqΔ / 2) / np.sqrt(σB_sqΔ)
        d2hat = d1hat - np.sqrt(σB_sqΔ)

        if self.__option_type == 'call':
            return np.exp(-self.__r * self.__Δ) * (B0 * np.exp(μBΔ) * norm.cdf(d1hat) - self.__K * norm.cdf(d2hat))
        elif self.__option_type == 'put':
            return np.exp(-self.__r * self.__Δ) * (self.__K * norm.cdf(-d2hat) - B0 * np.exp(μBΔ) * norm.cdf(-d1hat))

    def geo_std_MC(self):
        self.__gen_geo_payoff()
        return np.mean(self.__geo_payoff)

    def arith_std_MC(self):
        self.__gen_arith_payoff()
        return np.mean(self.__arith_payoff)


# # test, closed-form formula
# basket_call = BasketOption(S=[100, 100], K=100, r=0.05, T=3, σ=[0.3, 0.3], ρ=[[1, 1], [1, 1]], n=15, option_type='Call')
# print('Call Options:(By Setting Correlation = 1,Same Assets, all values converge to 1 result)')
# print('Arithmetic standard MC\t{:f}'.format(basket_call.arith_std_MC()))
# print('Geometric standard MC\t{:f}'.format(basket_call.geo_std_MC()))
# print('Arithmetic MC with Control Variate\t' + str(basket_call.control_variate()))
# print('Geometric closed-form formula\t{:f}'.format(basket_call.closed_form()))

# print('\nPut Options:(By Setting Correlation = 1,Same Assets, all values converge to 1 result)')
# basket_put = BasketOption(S=[50, 50], K=100, r=0.15, T=6, σ=[0.3, 0.3], ρ=[[1, 1], [1, 1]], n=50, option_type='Put')
# print('Arithmetic standard MC\t{:f}'.format(basket_put.arith_std_MC()))
# print('Geometric standard MC\t{:f}'.format(basket_put.geo_std_MC()))
# print('Arithmetic MC with Control Variate\t' + str(basket_put.control_variate()))
# print('Geometric closed-form formula\t{:f}'.format(basket_put.closed_form()))


# print('\nPut Options:(Ari and Geo are diff when σ1 σ2 are very diff)')
# basket_put = BasketOption(S=[5, 5], K=5, r=0.15, T=3, σ=[0.7, 0.3], ρ=[[1, 0.5], [0.5, 1]], n=5, option_type='Call')
# print('Arithmetic standard MC\t{:f}'.format(basket_put.arith_std_MC()))
# print('Geometric standard MC\t{:f}'.format(basket_put.geo_std_MC()))
# print('Arithmetic MC with Control Variate\t' + str(basket_put.control_variate()))
# print('Geometric closed-form formula\t{:f}'.format(basket_put.closed_form()))

# print('\nPut Options:(Ari and Geo  are diff when S1 S2 are very diff)')
# basket_put = BasketOption(S=[25, 75], K=50, r=0.05, T=10, σ=[0.7, 0.3], ρ=[[1, 0.1], [0.1, 1]], n=10, option_type='Put')
# print('Arithmetic standard MC\t{:f}'.format(basket_put.arith_std_MC()))
# print('Geometric standard MC\t{:f}'.format(basket_put.geo_std_MC()))
# print('Arithmetic MC with Control Variate\t' + str(basket_put.control_variate()))
# print('Geometric closed-form formula\t{:f}'.format(basket_put.closed_form()))
