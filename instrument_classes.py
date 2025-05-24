import numpy as np
import math
#Barrier call option class


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
