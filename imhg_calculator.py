import numpy as np
import scipy

import utils
from decimal import Decimal, getcontext


class imHGCalculator(object):
    def __init__(self, default_precision=40):
        self.hgt_cache_map = {}
        self.default_precision = default_precision
        getcontext().prec = default_precision

    def calculate_imhg(self, N, B, lamda):
        """
        :param N: Number of genes in the chromosome
        :param B: Number of active genes in the chromosome
        :param lamda: chromosome as a tuple of 0s and 1s
        """
        min_val = None
        indexes = (-1, -1)

        for i in range(0, N):
            b_i_sum = sum(lamda[i:i + 1])
            for j in range(i + 1, N):
                if lamda[j] == 0:
                    continue
                b_i_sum += 1
                key = str(b_i_sum) + "_" + str(N) + "_" + str(B) + "_" + str(j - i + 1)
                if key in self.hgt_cache_map:
                    val = self.hgt_cache_map[key]
                else:
                    val = utils.fast_calculate_hgt(b_i_sum, N, B, j - i + 1)
                    if val == 0 or np.isnan(val):
                        val = utils.accurate_calculate_hgt(b_i_sum, N, B, j - i + 1)
                        if val == 0 or np.isnan(val):
                            raise RuntimeError("imHG calculation failed for N: {}, B: {}, b: {}, i: {}, j: {}"
                                               .format(N, B, b_i_sum, i, j))

                    self.hgt_cache_map[key] = val
                if min_val is None:
                    min_val = val
                    indexes = (i, j)
                elif min_val > val:
                    min_val = val
                    indexes = (i, j)

        return min_val, indexes

    def calculate_p_value(self, N, B, p):
        """
        The following method calculates the p-value of the given N,B,p params.
        For more information about the algorithm, please refer to the paper.
        """
        arr = [[0] * (B + 1) for _ in range(N + 1)]
        arr[0][0] = 1
        for n in range(1, N + 1):
            for b in range(max(0, B - N + n), min(B, n) + 1):
                key = str(b) + "_" + str(N) + "_" + str(B) + "_" + str(n)
                if key in self.hgt_cache_map:
                    hgt_score = self.hgt_cache_map[key]
                else:
                    hgt_score = utils.fast_calculate_hgt(b, N, B, n)
                    if np.isnan(hgt_score) or hgt_score == 0:
                        hgt_score = utils.accurate_calculate_hgt(b, N, B, n)
                    self.hgt_cache_map[key] = hgt_score

                if np.isnan(hgt_score) or hgt_score == 0:
                    continue
                if hgt_score <= p:
                    arr[n][b] = 0
                else:
                    if b == 0:
                        arr[n][b] = arr[n - 1][b]
                    else:
                        arr[n][b] = arr[n - 1][b] + arr[n - 1][b - 1]

        result = 1 - ((arr[N][B]) / scipy.special.binom(N, B))

        if result <= 0:
            result = self._increase_p_val_calculation_precision(N, B, arr)
            if arr[N][B] == 0 or result == 0:
                raise ValueError()
        return result

    def _increase_p_val_calculation_precision(self, N, B, arr):
        """
        The following method increases the precision of the p-value calculation.
        Due to the fact that the p-value is very small, the calculated p-value might be zero
        because of the precision of the calculation. To overcome this issue, we increase the
        precision of the calculation.
        """
        for i in range(1, 4):
            getcontext().prec = self.default_precision + i * 20
            result = Decimal(1 - (Decimal((arr[N][B]) / utils.binomial_coefficient(N, B))))
            if result > 0:
                getcontext().prec = self.default_precision
                return result

        getcontext().prec = self.default_precision
        return 0
