"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP3 to compute the statics for the analsis
"""
import numpy as np
import scipy.stats as stats
from scipy.stats import levene
from scipy.stats import ttest_ind

def run_comparison(group1, group2):
    """
    This function runs assumption test for equal variance and performs the t
    or welch test
    :param group1: group1_values
    :param group2: group2_values
    :return: string with bonferroni corrected p-values or reason for no test
    """
    # test for equal variances
    if np.var(group1)==0 or np.var(group2)==0:
        return("Null-var", np.nan )

    else:
        # test for equal variance
        _, p = levene(group1, group2)
        if np.array(p) > 0.05:
            equ_var = True
        if np.array(p) <= 0.05:
            equ_var = False

        # t-test or welch test:
        t, p = ttest_ind(group1, group2, equal_var = equ_var)
        # Bonferroni_adjustment
        # Create a list of the adjusted p-values

        return("t={0:.3f}".format(t), p)


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

def ANOVA_assumptions_test_state(C,A,H):
    # RNCH are the provided groups

    valid = False

    # test for normality
    ps =[]
    for i in [C,A,H]:
        shapiro_test = stats.shapiro(i)
        ps.append(shapiro_test.pvalue)

    # test for equal variances
    _, p = levene(C,A,H)
    ps.append(p)

    if (np.array(ps) > 0.05).all():
        valid = True

    if valid:
        # stats f_oneway functions takes the groups as input and returns F and P-value
        fvalue, pvalue = stats.f_oneway(C,A,H)
        test = 'ANOVA'
    else:
        fvalue, pvalue = stats.kruskal(C,A,H)
        test = 'kruskal'

    return fvalue, pvalue, test

def ANOVA_assumptions_test_Diag(R,N,C,H,NC):
    minval = min(len(R),len(N),len(C),len(H),len(NC))

    valid = False

    if minval >= 3:

        # test for normality
        ps =[]
        for i in [R,N,C,H,NC]:
            shapiro_test = stats.shapiro(i)
            ps.append(shapiro_test.pvalue)

        # test for equal variances
        _, p = levene(R,N,C,H,NC)
        ps.append(p)

        if (np.array(ps) > 0.05).all():
            valid = True

    if valid:
        # stats f_oneway functions takes the groups as input and returns F and P-value
        fvalue, pvalue = stats.f_oneway(R, N, C, H,NC)
        test = 'ANOVA'
    else:
        fvalue, pvalue = stats.kruskal(R, N, C, H,NC)
        test = 'kruskal'

    return fvalue, pvalue, test
