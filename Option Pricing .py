import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
Monte Carlo European Call Option Pricer vs Black Scholes Pricing

"""

S0 = 100 #Stock at price 0
K = 105 # Strike Price
r = 0.05 #Risk free
sigma = 0.20 #Volatility
T = 1 #Time

rng = np.random.default_rng(42) #Sets a seed
sample_sizes = [1000, 10000, 100000, 1000000]
output = {}



# Black Scholes


d1 = (np.log(S0/K) + (r + (1/2*(sigma ** 2)))* T) / (sigma * (np.sqrt(T)))
d2 = d1 - (sigma * (np.sqrt(T)))

# Standard Normal Cumulative Distribution Function
Nd1 = norm.cdf(d1) # Analytical Delta
Nd2 = norm.cdf(d2)

blackscholes_price = (S0 * Nd1) - ((K * np.exp(-r * T)) * Nd2 ) #Black Scholes Formulae


for n in range(len(sample_sizes)):
    N = sample_sizes[n]
    Z = rng.standard_normal(N) #Generates random values from (0,1)


    exponent = (r - (1/2 * (sigma ** 2))) * T + (sigma * np.sqrt(T) * Z)

# Stock price at time T
    St = S0 * np.exp(exponent)

#Payoff
    payoffs = np.maximum(St - K, 0)
    average_payoff = np.mean(payoffs)

# Discount back |  PV
    C0 = np.exp(-r * T) * average_payoff

# Standard of Error
    SE = np.exp(-r * T) * ((np.std(payoffs,ddof=1))/np.sqrt(N)) # ddof =1 is standard deviation


# Confidence Interval

    lower = C0 - 1.96 * SE
    upper = C0 + 1.96 * SE

    error = abs(C0 - blackscholes_price)

    output[sample_sizes[n]] = {
        "Present Value| Price" : C0,
        "Standard of Error" : SE,
        "Upper limit": upper,
        "Lower limit": lower,
        "Error": error
        }

convergance_of_error = [output[key]["Error"] for key in sample_sizes]



plt.plot(sample_sizes, convergance_of_error, marker="o")
plt.xscale("log")
plt.xlabel("Number of Simulations (N)")
plt.ylabel("Absolute Pricing Error")
plt.title("Monte Carlo Pricing Error vs. Number of Simulations")

plt.grid(True)
plt.tight_layout()
plt.show()



