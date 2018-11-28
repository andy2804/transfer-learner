"""
author: az
"""
from collections import namedtuple

import numpy as np
from contracts import contract
from scipy import stats


class Bernoulli(namedtuple('Ber', 'ns n')):
    """
    Bernoulli experiment
    :param ns: number of successes
    :param n:number of total trials
    """

    @contract(ns='int,>=0', n='int,>=0')
    def __new__(cls, ns, n):
        return super(Bernoulli, cls).__new__(cls, ns, n)

    def __str__(self):
        low_bound, high_bound = self.get_confidence_interval(.95, "wilson")
        return 'Bernoulli experiment:\n' \
               'Successes/trials={:d}/{:d}\t=~{:.2f}  \n' \
               'confidence bounds at 95% (wilson):\n' \
               'low={:3f}, high={:3f}'.format(self.ns, self.n, self.estimate, low_bound, high_bound)

    @property
    def estimate(self, ):
        return 0 if self.n == 0 else self.ns / self.n

    def get_confidence_interval(self, conf_level, conf_method):
        """
        :param conf_level:
        :param conf_method:
        :return:
        """
        available_methods = {
            "wilson":          self._wilson_ci,
            "clopper_pearson": self._clopper_pearson_ci
        }

        if conf_method in available_methods:
            return available_methods[conf_method](self.ns, self.n, conf_level)
        else:
            raise ValueError("The requested method for confidence intervals is not available")

    @staticmethod
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
        z = stats.norm.isf((1 - confidence) / 2)
        print("z:", z)
        p = ns / n
        # compute partial results for more readable final formula
        z2 = z ** 2
        mean = 2 * n * p + z2
        int_lb = - z * np.sqrt(z2 - 1 / n + 4 * n * p * (1 - p) + (4 * p - 2) + 1)
        int_ub = z * np.sqrt(z2 - 1 / n + 4 * n * p * (1 - p) - (4 * p - 2) + 1)
        # final lower and upper bound
        lb = (mean + int_lb) / (2 * (n + z2))
        ub = (mean + int_ub) / (2 * (n + z2))
        return max(0, lb), min(1, ub)

    @staticmethod
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
        lb = 1 - stats.beta.ppf(1 - a / 2, n - ns, ns + 1)
        ub = 1 - stats.beta.ppf(a / 2, n - ns + 1, ns)
        return max(0, lb), min(1, ub)
