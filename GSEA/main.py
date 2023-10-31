import datetime
import json
from decimal import Decimal
import os
import numpy as np
import pandas as pd

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


def find_set_extra_special_chromosomes(results, set_name):
    return np.asarray([chromosome for chromosome in results if chromosome[0] == set_name])


def get_parsed_data(data):
    data = [[*d[0:2], Decimal("{:.4g}".format(Decimal(d[2]))), *d[3:]] for d in data]
    data.sort(key=lambda x: x[2])
    filtered = []
    exclude = ["AMPLIFIED", "DELETION", "AMPLICON", "DELETED"]
    for val in data:
        name = val[0]
        if name.startswith("chr"):
            continue
        if any(word in name for word in exclude):
            continue
        filtered.append(val)

    n = np.asarray(filtered)
    new_col = list((n[:, 4] / n[:, 7]) * 100)
    b_to_tot_b_ratio = np.asarray([float("{:.3f}".format(x)) for x in new_col])
    new_col = list(n[:, 0])
    extra_special_chromosomes = np.asarray([len(find_set_extra_special_chromosomes(n, name)) for name in new_col])
    n = np.hstack((n, b_to_tot_b_ratio.reshape(-1, 1)))
    n = np.hstack((n, extra_special_chromosomes.reshape(-1, 1)))
    return n


def dump_results_to_excel(results, excel_name="results.xlsx", start=0, end=None):
    df = raw_results_to_df(results, start, end)
    writer = pd.ExcelWriter(excel_name)
    df.to_excel(writer, 'Genomic analysis', index=False)
    writer.close()


def raw_results_to_df(results, start=0, end=None):
    df = pd.DataFrame(results[start:end],
                      columns=["Set", "Chromosome", "p-value", "Length", "b", "B", "N", "total_B", "indices",
                               "b_to_Tot_B_ratio", "number_of_enriched_genomic_intervals"])
    df['lift'] = (df['b'] / df['Length']) / (df['B'] / df['N'])
    return df


def run_gsea_human_analysis(gsea_all_gene_set_json_bundle_path):
    print('Running GSEA human analysis')
    with open(gsea_all_gene_set_json_bundle_path, "r") as f:
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
    return get_parsed_data(p_val_results)


def main():
    gsea_json_path = "assets/msigdb.v2023.1.Hs.json"
    results = run_gsea_human_analysis(gsea_json_path)
    parsed_results_df = raw_results_to_df(results, start=0, end=None)
    dump_results_to_excel(results, excel_name="results.xlsx", start=0, end=None)


if __name__ == '__main__':
    main()
