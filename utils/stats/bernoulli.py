"""
author: az
"""
from collections import namedtuple

import numpy as np
from contracts import contract
from scipy import stats


class Estimate(namedtuple('Est', 'est lb ub conf method')):

    def __new__(cls, est, lb=None, ub=None, conf=None, method=None):
        return super(Estimate, cls).__new__(cls, est, lb, ub, conf, method)

    def __str__(self):
        return 'Val: estimate={:.3f}  \n' \
               'lower bound={:.3f}  upper bound={:.3f}\n' \
               'confidence level: {:2f}' \
               'method={}'.format(self.est, self.lb, self.ub, self.conf, self.method)

    @property
    def interval_length(self, ):
        return self.ub - self.lb


def bernoulli_conf_int(ns, n, conf_level, conf_method):
    """
    :param ns:
    :param n:
    :param conf_level:
    :param conf_method:
    :return:
    """
    # convert to tuple if not already
    ns, n, conf_level = list(map(lambda x: (x,) if type(x) is not tuple else x,
                                 [ns, n, conf_level]))
    # actual func. implementation
    available_methods = {"wilson": _wilson_ci, "clopper_person": _clopper_pearson_ci}
    if conf_method in available_methods:
        results = []
        for ns_, n_, conf_level_ in zip(ns, n, conf_level):
            lb, ub = available_methods[conf_method](ns_, n_, conf_level_)
            results.append(
                    Estimate(est=ns_ / n_, lb=lb, ub=ub, conf=conf_level_, method=conf_method))
        return results
    else:
        raise ValueError("The requested method for confidence intervals is not available")


@contract
def _wilson_ci(ns, n, confidence=0.95):
    """
    Wilson score interval with continuity correction
    :param ns: number of successes
    :type ns: int,>=0
    :param n: sample size
    :type n: int,>=0
    :param confidence: confidence interval
    :type confidence: float,>0,<1
    :return: symmetric value
    """
    if n == 0:
        return 0, 0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = ns / n
    # compute partial results for more readable final formula
    mean = 2 * n * p + z ** 2
    int_lb = - z * np.sqrt(z ** 2 - 1 / n + 4 * n * p * (1 - p) + (4 * p - 2) + 1)
    int_ub = z * np.sqrt(z ** 2 - 1 / n + 4 * n * p * (1 - p) - (4 * p - 2) + 1)
    # final lower and upper bound
    lb = (mean + int_lb) / (2 * (n + z ** 2))
    ub = (mean + int_ub) / (2 * (n + z ** 2))
    return max(0, lb), min(1, ub)


@contract
def _clopper_pearson_ci(ns, n, confidence=0.95):
    """
    Clopper–Pearson interval based on the beta inverse function
    :param ns: number of successes
    :type ns: int,>=0
    :param n: sample size
    :type n: int,>=0
    :param confidence: confidence interval
    :type confidence: float,>0,<1
    :return: lower bound, upper bound
    """
    # todo test
    if n == 0:
        return 0, 0
    a = 1 - confidence
    lb = 1 - stats.beta.cdf(1 - a / 2, n - ns, ns + 1)
    ub = 1 - stats.beta.cdf(a / 2, n - ns + 1, ns)
    return lb, ub
