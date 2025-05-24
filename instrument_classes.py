import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#Barrier call option class
class BarrierCall:
    def __init__(self, S0, K, T, r, sigma, B, M=10000, N=252, random_seed=None):
        self.S0 = S0  # Initial stock price
        self.K = K    # Strike price
        self.T = T    # Time to maturity
        self.r = r    # Risk-free rate
        self.sigma = sigma  # Volatility
        self.B = B    # Barrier level (up-and-in)
        self.M = M    # Number of simulations
        self.N = N    # Number of time steps
        self.random_seed = random_seed
        self.paths = None  # Store simulation paths for reuse

    def simulate_paths(self):
        """Simulate GBM paths for the underlying asset."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        dt = self.T / self.N
        S = np.zeros((self.N + 1, self.M))
        S[0] = self.S0

        for t in range(1, self.N + 1):
            Z = np.random.normal(0, 1, self.M)
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)

        self.paths = S
        return S

    def price(self):
        """Price the up-and-in barrier call option."""
        if self.paths is None:
            self.simulate_paths()

        S = self.paths
        breached = np.any(S > self.B, axis=0)
        payoffs = np.where(breached, np.maximum(S[-1] - self.K, 0), 0)
        discount_factor = np.exp(-self.r * self.T)
        option_price = discount_factor * np.mean(payoffs)
        return option_price

    def visualize(self, max_paths=100):
        """Visualize the GBM paths with barrier and payoff outcomes."""
        if self.paths is None:
            self.simulate_paths()

        S = self.paths[:, :max_paths]
        t = np.linspace(0, self.T, self.N + 1)
        breached = np.any(S > self.B, axis=0)
        in_the_money = (S[-1] > self.K) & breached

        plt.figure(figsize=(12, 6))
        for j in range(S.shape[1]):
            if in_the_money[j]:
                plt.plot(t, S[:, j], color='green', alpha=0.7, linewidth=1.0)
            elif breached[j]:
                plt.plot(t, S[:, j], color='red', alpha=0.5, linewidth=0.8)
            else:
                plt.plot(t, S[:, j], color='grey', alpha=0.3, linewidth=0.8)

        plt.axhline(y=self.B, color='blue', linestyle='--', linewidth=1.5, label='Barrier Level (B)')

        legend_elements = [
            Line2D([0], [0], color='green', lw=1, label='Contributing Paths'),
            Line2D([0], [0], color='red', lw=1, label='Breached, No Payoff'),
            Line2D([0], [0], color='grey', lw=1, label='Never Breached'),
            Line2D([0], [0], color='blue', lw=1, linestyle='--', label='Barrier Level (B)')
        ]
        plt.legend(handles=legend_elements, loc="upper left")
        plt.title(f"Barrier Call Option Paths (S₀={self.S0}, K={self.K}, B={self.B})")
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        plt.show()

# Zero Curve Class
class ZeroCurve:
    def __init__(self):
        # set up empty list
        self.maturities = []
        self.zero_rates = []
        self.AtMats = []
        self.discount_factors = []
    
    def add_zero_rate(self, maturity, zero_rate):
        self.maturities.append(maturity)
        self.zero_rates.append(zero_rate)
        self.AtMats.append(math.exp(zero_rate*maturity))
        self.discount_factors.append(1/self.AtMats[-1])

    def add_discount_factor(self, maturity, discount_factor):
        self.maturities.append(maturity)
        self.discount_factors.append(discount_factor)
        self.AtMats.append(1/discount_factor)
        self.zero_rates.append(math.log(1/discount_factor)/maturity)
    
    def get_AtMat(self, maturity):
        if maturity in self.maturities:
            return self.AtMats[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.AtMats, maturity)

    def get_discount_factor(self, maturity):
        if maturity in self.maturities:
            return self.discount_factors[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.discount_factors, maturity)

    def get_zero_rate(self, maturity):
        if maturity in self.maturities:
            return self.zero_rates[self.maturities.index(maturity)]
        else:
            return math.log(self.get_AtMat(maturity))/maturity
        
    def get_zero_curve(self):
        return self.maturities, self.discount_factors
    
    def npv(self, cash_flows):
        npv = 0
        for maturity in cash_flows.get_maturities():
            npv += cash_flows.get_cash_flow(maturity)*self.get_discount_factor(maturity)
        return npv
        
def exp_interp(xs, ys, x):
    """
    Interpolates a single point for a given value of x 
    using continuously compounded rates.

    Parameters:
    xs (list or np.array): Vector of x values sorted by x.
    ys (list or np.array): Vector of y values.
    x (float): The x value to interpolate.

    Returns:
    float: Interpolated y value.
    """
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Find the interval [x0, x1] where x0 <= x <= x1
    idx = np.searchsorted(xs, x) - 1
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    
    # Calculate the continuously compounded rate
    rate = (np.log(y1) - np.log(y0)) / (x1 - x0)
    
    # Interpolate the y value for the given x
    y = y0 * np.exp(rate * (x - x0))
    
    return y

# Yield Curve class, inherits functionality from Zero Curve class 
class YieldCurve(ZeroCurve):
    def __init__(self):
        super().__init__()
        self.portfolio = []

    # set the constituent portfolio
    # the portfolio must contain bills and bonds in order of maturity
    # where all each successive bond only introduces one new cashflow beyond 
    #       the longest maturity to that point (being the maturity cashflow)
    def set_constituent_portfolio(self, portfolio):
        self.portfolio = portfolio

    def bootstrap(self):
        options = self.portfolio.get_options()
        self.add_zero_rate(0,0)
        for option in options:
            # calculate the PV of the bond cashflows excluding the maturity cashflow
            pv = 0
            option_dates = option.get_maturities()
            option_amounts = option.get_amounts()
            for i in range(1, len(option_amounts)-1):
                pv += option_amounts[i]*self.get_discount_factor(option_dates[i])
            print("PV of all the cashflows except maturity is: ", pv)
            print("The bond price is: ", option.get_price())
            print("The last cashflow is: ", option_amounts[-1])
            self.add_discount_factor(option.get_maturity(),(option.get_price()-pv)/option.get_amounts()[-1])
