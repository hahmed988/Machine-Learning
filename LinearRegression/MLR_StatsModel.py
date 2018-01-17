# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:55:41 2017

@author: HAHMED [Source: DataRobots.com]
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

df_adv = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
X = df_adv[['TV', 'radio']]
y = df_adv['sales']
df_adv.head()

"""
The multiple regression model describes the response as a weighted sum of the predictors:
\(Sales = \beta_0 + \beta_1 \times TV + \beta_2 \times Radio\)
"""

## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

est.summary()


"""
You can also use the formulaic interface of statsmodels to compute regression with multiple predictors. 
You just need append the predictors to the formula via a '+' symbol.
"""

# import formula api as alias smf
import statsmodels.formula.api as smf

# formula: response ~ predictor + predictor
est = smf.ols(formula='sales ~ TV + radio', data=df_adv).fit()