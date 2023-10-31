import math
from decimal import Decimal

import numpy as np
import scipy
from scipy.stats import hypergeom


def create_chromosomes_dict(chromosomes):
    genome = dict()
    for chroIndex, chro in enumerate(chromosomes):
        for geneIndex, gene in enumerate(chro):
            genome[gene] = (chroIndex, geneIndex)
    return genome


def fast_calculate_hgt(b, N, B, n):
    """
    Calculates the hypergeometric tail probability using the following formula:
    sum_{i=b}^{min(n,B)} \frac{\binom{n}{i} \binom{N-n}{B-i}}{\binom{N}{B}}
    Note: This method is basically implementing scipy.stats.hypergeom.sf(b-1, N, B, n) method, but it is much faster,
    however, it is less stable with large numbers.
    If you want to use the accurate method, use the method accurate_calculate_hgt method instead.
    """
    return np.sum(
        [((scipy.special.binom(n, i) * scipy.special.binom(N - n, B - i)) / scipy.special.binom(N, B)) for i in
         range(b, min(n, B) + 1)])


def accurate_calculate_hgt(b, N, B, n):
    """
    Calculates the hypergeometric tail probability using the following formula:
    sum_{i=b}^{min(n,B)} \frac{\binom{n}{i} \binom{N-n}{B-i}}{\binom{N}{B}}
    """
    return hypergeom(N, B, n).sf(b - 1)


def generate_printable_chromosome_number(chromosome_number):
    print_chromosome_key = str(chromosome_number + 1)
    if chromosome_number == 22:
        print_chromosome_key = "X"
    elif chromosome_number == 23:
        print_chromosome_key = "Y"

    return print_chromosome_key


def binomial_coefficient(a, b):
    numerator = math.factorial(a)
    denominator = math.factorial(b) * math.factorial(a - b)
    return Decimal(numerator) / Decimal(denominator)
