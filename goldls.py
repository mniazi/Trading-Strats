rom quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, AverageDollarVolume
from quantopian.pipeline.data.quandl import bundesbank_bbk01_wt5511 as gold


import pandas as pd
import numpy as np
from scipy.stats import linregress

class QuandlLinearRegression(CustomFactor):
    
    outputs = ['alpha', 'beta', 'r_value', 'p_value', 'stderr'] 
    
    def compute(self, today, assets, out, benchmark, y):
        returns = np.diff(y, axis=0) / y[:-1]
        benchmark_values = benchmark[:,0]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]

        alpha = out.alpha
        beta = out.beta
        r_value = out.r_value
        p_value = out.p_value
        stderr = out.stderr
        for i in range(len(out)):
            other_asset = returns[:, i]
            regr_results = linregress(y=other_asset, x=benchmark_returns)
            # `linregress` returns its results in the following order:
            # slope, intercept, r-value, p-value, stderr
            alpha[i] = regr_results[1]
            beta[i] = regr_results[0]
            r_value[i] = regr_results[2]
            p_value[i] = regr_results[3]
            stderr[i] = regr_results[4]


# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    
    pipe = Pipeline()
    pipe = attach_pipeline(pipe, name='regression_metrics')
    
    dollar_volume = AverageDollarVolume(window_length=20)
    high_dollar_volume = (dollar_volume > 10**7)
    
    data_regression_results = QuandlLinearRegression(inputs=[gold.value, USEquityPricing.close],
                                                              window_length=60, 
                                                              mask=high_dollar_volume)
    
    beta = data_regression_results.beta
    
    pipe.add(beta, "gold_beta")
    
    pipe.set_screen(high_dollar_volume)
    
    context.shorts = None
    context.longs = None
    
    schedule_function(plot, date_rules.every_day())
    schedule_function(rebalance, date_rules.month_start())
    
def before_trading_start(context, data):
    results = pipeline_output('regression_metrics').dropna()      
    ranks = results["gold_beta"].abs().order()
    
    context.shorts = ranks.tail(50)
    context.longs = ranks.head(50)
    

# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    pass

def plot(context, data):
    record(lever=context.account.leverage,
           exposure=context.account.net_leverage,
           num_pos=len(context.portfolio.positions),
           oo=len(get_open_orders()))

        
    
def rebalance(context, data):
    for security in context.shorts.index:
        if get_open_orders(security):
            continue
        order_target_percent(security, -0.5 / len(context.shorts))
            
    for security in context.longs.index:
        if get_open_orders(security):
            continue
        order_target_percent(security, 0.5 / len(context.longs))
            
    for security in context.portfolio.positions:
        if get_open_orders(security):
            continue
        if security not in (context.longs.index | context.shorts.index):
            order_target_percent(security, 0)
