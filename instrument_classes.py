import numpy as np
import math

# Creating cash flow class - inherited for each instrument
class CashFlows:
    def __init__(self):
        self.maturities = []
        self.amounts = []
    
    # add a cash flow to the cash flow list
    def add_cash_flow(self, maturity, amount):
        self.maturities.append(maturity)
        self.amounts.append(amount)

    def get_cash_flow(self, maturity):
        if maturity in self.maturities:
            return self.amounts[self.maturities.index(maturity)]
        else:
            return None
        
    def get_maturities(self):
        return list(self.maturities)
    
    def get_amounts(self):
        return list(self.amounts)
    
    def get_cash_flows(self):
        return list(zip(self.maturities, self.amounts))

    ### NEED TO MAKE THIS OPTION SPECIFIC

# create a class for options that inherits from CashFlows
class Options(CashFlows):
    
    def __init__(self, face_value=100, maturity=3, coupon=0, frequency=4, ytm=0, price=100):
        super().__init__()
        self.face_value = face_value    
        self.maturity = maturity
        self.coupon = coupon
        self.frequency = frequency
        self.ytm = ytm
        self.price = price
    
    def set_face_value(self, face_value):
        self.face_value = face_value

    def set_maturity(self, maturity):
        self.maturity = maturity

    def set_coupon(self, coupon):
        self.coupon = coupon

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_ytm(self, ytm):
        self.ytm = ytm
        # set the price of the option using option pricing
        self.price = (self.face_value*self.coupon/self.frequency)*(1-(1+ytm/self.frequency)**(-self.maturity*self.frequency))/(ytm/self.frequency) \
          + self.face_value/((1 + ytm/self.frequency)**(self.maturity*self.frequency))

    def get_price(self):
        return self.price
    
    def get_face_value(self):
        return self.face_value
    
    def get_maturity(self):
        return self.maturity
    
    def get_coupon_rate(self):
        return self.coupon
    
    def get_ytm(self):
        return self.ytm
    
    def set_cash_flows(self):
        self.add_cash_flow(0, -self.price)
        for i in range(1, self.maturity*self.frequency):
            self.add_cash_flow(i/self.frequency, self.face_value*self.coupon/self.frequency)
        self.add_cash_flow(self.maturity, self.face_value + self.face_value*self.coupon/self.frequency)


# Portfolio class containing all instruments which inherits from parent CashFlows class
class Portfolio(CashFlows):
    
    def __init__(self):
        super().__init__()
        self.options = []
    
    def add_option(self, options):
        self.options.append(options)
    
    def get_options(self):
        return self.options
    
    def set_cash_flows(self):
        for option in self.options:
            for cash_flow in option.get_cash_flows():
                self.add_cash_flow(cash_flow[0], + cash_flow[1])

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
