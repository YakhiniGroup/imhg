import datetime
import json

import numpy as np

import utils
from chromosomes import chromosomes
from imhg_calculator import imHGCalculator

significance_threshold = 1e-10


def create_dict():
    genome = dict()
    for chromosome_index, chromosome in enumerate(chromosomes):
        for gene_index, gene in enumerate(chromosome):
            genome[gene] = (chromosome_index, gene_index)
    return genome


def run_gsea_human_analysis():
    print('Running GSEA human analysis')
    with open("/Users/shahar.mor/Downloads/msigdb.v2023.1.Hs.json", "r") as f:
        data = json.load(f)
    keys = list(data.keys())
    genes = {}

    for k in keys:
        genes[k] = data[k]['geneSymbols']

    genes = list(genes.items())
    genome_main = create_dict()
    p_val_results = []
    imhg_calculator = imHGCalculator()

    print("Starting {}".format(str(datetime.datetime.now())))
    for gene, gene_symbols in genes:
        results = []
        for idx, c in enumerate(chromosomes):
            results.append(np.zeros(len(c), dtype=int))

        gene_error = False
        for symbol in gene_symbols:
            try:
                a, b = genome_main[symbol]
            except KeyError:
                gene_error = True
            results[a][b] = 1

        if gene_error:
            continue

        total_B = 0
        for chr in results:
            total_B += sum(chr)

        for i, chr in enumerate(results):
            try:
                N = len(chr)
                B = sum(chr)
                if B < 3:
                    continue
                v = tuple(chr)
                try:
                    p, indexes = imhg_calculator.calculate_imhg(N, B, v)
                except RuntimeError:
                    print("Failed to calculate imhg for chromosome: {} \n".format(gene))
                    continue
                if p == 0:
                    print("imhg score for {} is 0".format(gene))
                    continue

                try:
                    mhg_p_value = imhg_calculator.calculate_p_value(N, B, p)
                except ValueError:
                    print("Unable to calculate imhg p-value, either p-value is zero or amount of options is "
                          "too small")
                    continue

                bounded_p_value = N * mhg_p_value
                if bounded_p_value <= significance_threshold:
                    printable_chromosome_number = utils.generate_printable_chromosome_number(i)
                    total_b = int(sum(chr[indexes[0]:indexes[1] + 1]))
                    p_val_results.append((gene, printable_chromosome_number, str(bounded_p_value),
                                          int(indexes[1] - indexes[0] + 1), total_b, int(B), int(N), int(total_B),
                                          str((indexes[0], indexes[1]))))
            except Exception as e:
                print("Failed to calculate {} for chromosome {} with error: {}".format(gene, i, e))

    print("Done")


def main():
    run_gsea_human_analysis()


if __name__ == '__main__':
    main()
