import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from scipy.stats import norm

#Generic European Call Option Class - will be used as a parent class that other options inherit from
# This class uses Black Scholes pricing 
class EuropeanCall:
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = S0      # Initial stock price
        self.K = K        # Strike price
        self.T = T        # Time to maturity (in years)
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset

    def price(self):
        """Price the European call option using the Black-Scholes formula."""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        call_price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price
    
#Generic American Put Option Class 
# This class uses Binomial pricing 
class AmericanPut:
    def __init__(self, S0, K, T, r, sigma, N=100):
        self.S0 = S0      # Initial stock price
        self.K = K        # Strike price
        self.T = T        # Time to maturity
        self.r = r        # Risk-free rate
        self.sigma = sigma  # Volatility
        self.N = N        # Number of time steps

    def price(self):
        """Price the American put option using the binomial tree method."""
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)

        stock_tree = np.zeros((self.N + 1, self.N + 1))
        option_tree = np.zeros((self.N + 1, self.N + 1))

        # Populate stock tree
        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S0 * (u ** j) * (d ** (i - j))

        # Option value at maturity
        for j in range(self.N + 1):
            option_tree[self.N, j] = max(self.K - stock_tree[self.N, j], 0)

        # Backward induction
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                hold = np.exp(-self.r * dt) * (
                    p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j]
                )
                exercise = self.K - stock_tree[i, j]
                option_tree[i, j] = max(hold, exercise)

        self.stock_tree = stock_tree
        self.option_tree = option_tree
        return option_tree[0, 0]
    def visualize(self):
        """Visualize the binomial stock and option value trees using NetworkX."""
        G = nx.DiGraph()
        pos = {}

        for i in range(self.N + 1):
            for j in range(i + 1):
                node_id = f"{i},{j}"
                stock_price = self.stock_tree[i, j]
                option_value = self.option_tree[i, j]
                G.add_node(node_id, stock_price=stock_price, option_value=option_value)
                pos[node_id] = (i, j - i / 2)

        for i in range(self.N):
            for j in range(i + 1):
                G.add_edge(f"{i},{j}", f"{i+1},{j}")
                G.add_edge(f"{i},{j}", f"{i+1},{j+1}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"American Put Option - Binomial Tree (N={self.N})")

        nx.draw(G, pos, ax=ax1, with_labels=False, node_size=600, node_color='skyblue', edge_color='gray')
        labels1 = {n: f"${G.nodes[n]['stock_price']:.2f}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels1, ax=ax1, font_size=8)
        ax1.set_title("Stock Price Tree")

        pos_option = {n: (x, -y) for n, (x, y) in pos.items()}
        nx.draw(G, pos_option, ax=ax2, with_labels=False, node_size=600, node_color='salmon', edge_color='gray')
        labels2 = {n: f"${G.nodes[n]['option_value']:.2f}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos_option, labels=labels2, ax=ax2, font_size=8)
        ax2.set_title("Option Value Tree")

        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

#Barrier call option class - inherits from European call
class BarrierCall(EuropeanCall):
    def __init__(self, S0, K, T, r, sigma, B, M=10000, N=252, random_seed=None):
        super().__init__(S0, K, T, r, sigma)
        self.B = B              # Barrier level
        self.M = M              # Number of Monte Carlo simulations
        self.N = N              # Number of time steps
        self.random_seed = random_seed
        self.paths = None       # Cached simulated paths

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
        """Override: Price the up-and-in barrier call option using Monte Carlo."""
        if self.paths is None:
            self.simulate_paths()

        S = self.paths
        breached = np.any(S > self.B, axis=0)
        payoffs = np.where(breached, np.maximum(S[-1] - self.K, 0), 0)
        discount_factor = np.exp(-self.r * self.T)
        return discount_factor * np.mean(payoffs)

    def visualize(self, max_paths=100):
        """Visualize GBM paths with barrier and payoff outcomes."""
        if self.paths is None:
            self.simulate_paths()

        S = self.paths[:, :max_paths]
        t = np.linspace(0, self.T, self.N + 1)
        breached = np.any(S > self.B, axis=0)
        in_the_money = (S[-1] > self.K) & breached

        plt.figure(figsize=(12, 6))
        for j in range(S.shape[1]):
            color = (
                'green' if in_the_money[j] else
                'red' if breached[j] else
                'grey'
            )
            alpha = 0.7 if in_the_money[j] else 0.5 if breached[j] else 0.3
            plt.plot(t, S[:, j], color=color, alpha=alpha, linewidth=1.0)

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

class EuropeanBasketCallOption(EuropeanCall):
    def __init__(self, spot_prices, sigmas, correlation_matrix, weights,
                 K, T, r, M=10000, random_seed=None):
        # Initialise parent EuropeanCall with dummy S0 and sigma (not used)
        super().__init__(S0=None, K=K, T=T, r=r, sigma=None)
        
        self.spot_prices = np.array(spot_prices)
        self.sigmas = np.array(sigmas)
        self.correlation_matrix = np.array(correlation_matrix)
        self.weights = np.array(weights)
        self.M = M
        self.random_seed = random_seed
        self.cholesky = np.linalg.cholesky(self.correlation_matrix)
        self.basket_paths = None

    def simulate_terminal_basket(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        Z = np.random.normal(size=(self.M, len(self.spot_prices)))
        correlated_Z = Z @ self.cholesky.T

        drift = (self.r - 0.5 * self.sigmas**2) * self.T
        diffusion = self.sigmas * np.sqrt(self.T)
        drift = drift[None, :]
        diffusion = diffusion[None, :]

        terminal_prices = self.spot_prices * np.exp(drift + correlated_Z * diffusion)
        basket_vals = terminal_prices @ self.weights
        self.basket_paths = basket_vals
        return basket_vals

    def price(self):
        if self.basket_paths is None:
            self.simulate_terminal_basket()
        payoffs = np.maximum(self.basket_paths - self.K, 0)
        discounted = np.exp(-self.r * self.T) * payoffs
        return np.mean(discounted)



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
