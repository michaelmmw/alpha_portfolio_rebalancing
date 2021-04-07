


#Initialization:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

equity_ret_disc = 0.1 #Based on SPX average
equity_std_disc = 0.3

equity_std = (np.log((equity_std_disc**2)/((1+equity_ret_disc)**2)+1))**0.5
equity_ret = np.log(1 + equity_ret_disc) - (equity_std**2)/2

risky_bond_ret_disc = 0.05           #Long term US corporate bonds
risky_bond_std_disc = 0.08

risky_bond_std = (np.log((risky_bond_std_disc**2)/((1+risky_bond_ret_disc)**2)+1))**0.5
risky_bond_ret = np.log(1 + risky_bond_ret_disc) - (risky_bond_std**2)/2

n = 10 #number of days simulated

monthly_intervals = []
for i in range(0, n, 21):
    monthly_intervals.append(i)
monthly_intervals.__delitem__(0)


yearly_intervals = []
for i in range(0, n, 252):
    yearly_intervals.append(i)
yearly_intervals.__delitem__(0)

weekly_intervals = []
for i in range(0, n, 5):
    weekly_intervals.append(i)
weekly_intervals.__delitem__(0)


semi_annual_intervals = []
for i in range(0, n, 126):
    semi_annual_intervals.append(i)
semi_annual_intervals.__delitem__(0)


def generator(nb_paths, equity_ret, equity_std, risky_bond_ret, risky_bond_std, corr, proportion_equity, proportion_bond):
#Proportion of equity and bonds must sum to 1

    res = pd.DataFrame(np.zeros((nb_paths, 5)), columns= ['No Rebalancing Geometric Average Return', 'Yearly Rebalancing Geometric Average Return', 'Semi Annually Rebalancing Geometric Average Return', 'Monthly Rebalancing Geometric Average Return', 'Weekly Rebalancing Geometric Average Return'])
    for l in range(0, nb_paths):

        a = math.sqrt(1 / n)
        x = np.random.normal(0, 1, n)
        # US equity estimate
        ret_equity = np.cumsum(equity_ret - 0.5 * (equity_std ** 2) * ( 1 / n) + x * equity_std * a)
        y = np.random.normal(0, 1, n)
        z = corr * x + math.sqrt(1 - corr ** 2) * y
        risky_bond_return = np.cumsum(risky_bond_ret - 0.5 * (risky_bond_std ** 2) * (1 / n) + z * risky_bond_std * a)

        #We now have the returns of one stock and one bond:
        #Let's assume all securities start with a price of $1, and we invest 1 in each :


        #No rebalancing:
        portfolio = np.zeros((n, 2))
        portfolio[0, 0] = proportion_equity
        portfolio[0, 1] = proportion_bond
        for e in range(1, n):
            portfolio[e, :] = portfolio[e-1, :]*[np.exp(ret_equity[e-1]),np.exp(risky_bond_return[e-1])]
        avg_return_no_reb = (np.sum(1 + portfolio[n-1, :]))**(1/n)-1


        #Yearly rebalancing:
        portfolio_yearly = np.zeros((n, 2))
        portfolio_yearly[0, 0] = proportion_equity
        portfolio_yearly[0, 1] = proportion_bond
        for r in range(1, n):
            portfolio_yearly[r, :] = portfolio_yearly[r - 1, :] * [np.exp(ret_equity[r - 1]), np.exp(risky_bond_return[r-1])]
            if r in yearly_intervals:
                portfolio_yearly[r, 0] = proportion_equity*np.sum(portfolio_yearly[r - 1, :])
                portfolio_yearly[r, 1] = proportion_bond*np.sum(portfolio_yearly[r - 1, :])
        avg_return_yearly = (np.sum(1 + portfolio_yearly[n-1, :]))**(1/n)-1


        #Semi-Annual rebalancing:
        portfolio_semi_annual = np.zeros((n, 2))
        portfolio_semi_annual[0, 0] = proportion_equity
        portfolio_semi_annual[0, 1] = proportion_bond
        for g in range(1, n):
            portfolio_semi_annual[g, :] = portfolio_semi_annual [g - 1, :] * [np.exp(ret_equity[g - 1]),np.exp(risky_bond_return[g - 1])]
            if g in semi_annual_intervals:
                portfolio_semi_annual[g, 0] = proportion_equity*np.sum(portfolio_semi_annual[g - 1, :])
                portfolio_semi_annual[g, 1] = proportion_bond*np.sum(portfolio_semi_annual[g - 1, :])
        avg_return_semi_annual = ((np.sum(1 + portfolio_semi_annual[n-1, :]))**(1/n))-1


        #Monthly rebalancing:
        portfolio_monthly = np.zeros((n, 2))
        portfolio_monthly [0, 0] = proportion_equity
        portfolio_monthly[0, 1] = proportion_bond
        for f in range(1, n):
            portfolio_monthly[f, :] = portfolio_monthly [f - 1, :] * [np.exp(ret_equity[f - 1]),np.exp(risky_bond_return[f - 1])]
            if f in monthly_intervals:
                portfolio_monthly[f, 0] = proportion_equity*np.sum(portfolio_monthly[f - 1, :])
                portfolio_monthly[f, 1] = proportion_bond*np.sum(portfolio_monthly[f - 1, :])
        avg_return_monthly = (np.sum(1 + portfolio_monthly[n-1, :]))**(1/n)-1


        #Weekly rebalancing:

        portfolio_weekly = np.zeros((n, 2))
        portfolio_weekly[0, 0] = proportion_equity
        portfolio_weekly[0, 1] = proportion_bond
        for d in range(1, n):
            portfolio_weekly[d, :] = portfolio_weekly[d - 1, :] * [np.exp(ret_equity[d - 1]),np.exp(risky_bond_return[d - 1])]
            if d in weekly_intervals:
                portfolio_weekly[d, 0] = proportion_equity*np.sum(portfolio_weekly[d, :])
                portfolio_weekly[d, 1] = proportion_bond*np.sum(portfolio_weekly[d, :])
        avg_return_weekly = (np.sum(1 + portfolio_weekly[n-1, :]))**(1/n)-1


        res.iloc[l, 0] = avg_return_no_reb
        res.iloc[l, 1] = avg_return_yearly
        res.iloc[l, 2] = avg_return_semi_annual
        res.iloc[l, 3] = avg_return_monthly
        res.iloc[l, 4] = avg_return_weekly


    return res




RES = generator(1, equity_ret, equity_std, risky_bond_ret, risky_bond_std,  0 , 0.8, 0.2)

p = [np.mean(RES.iloc[:, 0]), np.mean(RES.iloc[:, 1]), np.mean(RES.iloc[:, 2]), np.mean(RES.iloc[:, 3]), np.mean(RES.iloc[:, 4])]

a = math.sqrt(1 / n)
x = np.random.normal(0, 1, n)
# US equity estimate
ret_equity = np.cumsum(equity_ret - 0.5 * (equity_std ** 2) * (1 / n) + x * equity_std * a)
y = np.random.normal(0, 1, n)
z = 0 * x + math.sqrt(1 - 0 ** 2) * y
risky_bond_return = np.cumsum(risky_bond_ret - 0.5 * (risky_bond_std ** 2) * (1 / n) + z * risky_bond_std * a)

# No rebalancing:
portfolio = np.zeros((n, 2))
portfolio[0, 0] = 0.5
portfolio[0, 1] = 0.5
for e in range(1, n):
    portfolio[e, :] = portfolio[e - 1, :] * [np.exp(ret_equity[e - 1]), np.exp(risky_bond_return[e - 1])]
avg_return_no_reb = (np.sum(1 + portfolio[n - 1, :])) ** (1 / n) - 1

mean = [equity_ret, risky_bond_ret]
std = [equity_std, risky_bond_std]
covs = np.array([[std[0] ** 2, std[1] * std[0] * 0], [std[1] * std[0] * 0, std[1] ** 2]])
a = np.random.multivariate_normal(mean, covs, n)
ret_equity = a[:, 0]
risky_bond_return = a[:, 1]


# We now have the returns of one stock and one bond:
# Let's assume all securities start with a price of $1, and we invest 1 in each :


# No rebalancing:
portfolio = np.zeros((n, 2))
portfolio[0, 0] = 0.5
portfolio[0, 1] = 0.5
for e in range(1, n):
    portfolio[e, :] = portfolio[e - 1, :] * [1 + ret_equity[e - 1], 1 + risky_bond_return[e - 1]]
avg_return_no_reb = (np.prod(1+ portfolio[n - 1, :])) ** (1 / n) - 1
x = np.mean(portfolio[n - 1, :])

# Yearly rebalancing:
portfolio_yearly = np.zeros((n, 2))
portfolio_yearly[0, 0] = 0.5
portfolio_yearly[0, 1] = 0.5
for r in range(1, n):
    portfolio_yearly[r, :] = portfolio_yearly[r - 1, :] * [1 + ret_equity[r - 1], 1 + risky_bond_return[r - 1]]
    if r in yearly_intervals:
        portfolio_yearly[r, 0] = 0.5 * np.sum(portfolio_yearly[r, :])
        portfolio_yearly[r, 1] = 0.5 * np.sum(portfolio_yearly[r, :])
avg_return_yearly = (np.prod(1 + portfolio_yearly[n - 1, :])) ** (1 / n) - 1
y = np.mean(portfolio_yearly[n - 1, :])

# Semi-Annual rebalancing:
portfolio_semi_annual = np.zeros((n, 2))
portfolio_semi_annual[0, 0] = 0.5
portfolio_semi_annual[0, 1] = 0.5
for g in range(1, n):
    portfolio_semi_annual[g, :] = portfolio_semi_annual[g - 1, :] * [1 + ret_equity[g - 1],
                                                                     1 + risky_bond_return[g - 1]]
    if g in semi_annual_intervals:
        portfolio_semi_annual[g, 0] = 0.5 * np.sum(portfolio_semi_annual[g, :])
        portfolio_semi_annual[g, 1] = 0.5 * np.sum(portfolio_semi_annual[g, :])
avg_return_semi_annual = ((np.prod(1 + portfolio_semi_annual[n - 1, :])) ** (1 / n)) - 1
arith_avg_semi_annual = np.mean(portfolio_semi_annual[n-1, :])

# Monthly rebalancing:
portfolio_monthly = np.zeros((n, 2))
portfolio_monthly[0, 0] = 0.5
portfolio_monthly[0, 1] = 0.5
for f in range(1, n):
    portfolio_monthly[f, :] = portfolio_monthly[f - 1, :] * [1 + ret_equity[f - 1], 1 + risky_bond_return[f - 1]]
    if f in monthly_intervals:
        portfolio_monthly[f, 0] = 0.5 * np.sum(portfolio_monthly[f, :])
        portfolio_monthly[f, 1] = 0.5 * np.sum(portfolio_monthly[f, :])
avg_return_monthly = (np.sum(portfolio_monthly[n - 1, :])) ** (1 / n) - 1

# Weekly rebalancing:

portfolio_weekly = np.zeros((n, 2))
portfolio_weekly[0, 0] = 0.5
portfolio_weekly[0, 1] = 0.5
for d in range(1, n):
    portfolio_weekly[d, :] = portfolio_weekly[d - 1, :] * [1 + ret_equity[d - 1], 1 + risky_bond_return[d - 1]]
    if d in weekly_intervals:
        portfolio_weekly[d, 0] = 0.5 * np.sum(portfolio_weekly[d, :])
        portfolio_weekly[d, 1] = 0.5 * np.sum(portfolio_weekly[d, :])
avg_return_weekly = (np.sum(portfolio_weekly[n - 1, :])) ** (1 / n) - 1



