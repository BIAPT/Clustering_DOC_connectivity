"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP3 to compute the statics for the analsis
"""
import numpy as np
import scipy.stats as stats
from scipy.stats import levene


def ANOVA_assumptions_test(R,N,C,H):
    # RNCH are the provided groups

    valid = False

    # test for normality
    ps =[]
    for i in [R,N,C,H]:
        shapiro_test = stats.shapiro(i)
        ps.append(shapiro_test.pvalue)

    # test for equal variances
    _, p = levene(R,N,C,H)
    ps.append(p)

    if (np.array(ps) > 0.05).all():
        valid = True

    if valid:
        # stats f_oneway functions takes the groups as input and returns F and P-value
        fvalue, pvalue = stats.f_oneway(R, N, C, H)
        test = 'ANOVA'
    else:
        fvalue, pvalue = stats.kruskal(R, N, C, H)
        test = 'kruskal'

    return fvalue, pvalue, test
