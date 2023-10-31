import os
import traceback

import numpy as np

import utils
from chromosomes import chromosomes
from imhg_calculator import imHGCalculator
from results import ChromosomeResult, ExperimentResults


class DifferentiallyExpressedGenes(object):
    def __init__(self, experiment_name, significance_threshold=1e-4, output_directory="enriched_genes"):
        self.experiment_name = experiment_name
        self.imhg_calculator = imHGCalculator()
        self.significance_threshold = significance_threshold

        # genes is expected to be a list of lists, where each list is a gene and its p-value
        self.genes = None
        self.experiment_chromosomes = None
        self.output_directory = output_directory

    def load_genes_from_spaced_text_file(self, file_path, sort_genes=True):
        """
        Assuming the following file structure:
        - gene_name p_value
        """
        with open(file_path, 'r') as f:
            data = f.read()

        lines = data.split("\n")
        genes = []
        for line in lines:
            if line == "":
                continue
            gene, pvalue = line.split(" ")
            genes.append([gene, float(pvalue)])

        if sort_genes:
            genes = sorted(genes, key=lambda x: x[1])
            assert genes[0][1] < genes[-1][1]

        print(np.asarray(genes)[:10][:, 1])

        gene_names = list(np.asarray(genes)[:, 0])
        found = set()
        reduced_chromosomes = []
        for chromosome in chromosomes:
            reduced_chromosome = []
            for gene in chromosome:
                if gene in gene_names:
                    found.add(gene)
                    reduced_chromosome.append(gene)
            reduced_chromosomes.append(reduced_chromosome)

        # remove elements from genes which not in found
        genes = [gene for gene in genes if gene[0] in found]
        self.genes = genes

    def load_precomputed_genes(self, genes, sort_genes=True):
        """
        Assuming the following structure:
        genes is a list of lists, where each list is a gene and its p-value
        """
        if sort_genes:
            genes = sorted(genes, key=lambda x: x[1])
            assert genes[0][1] < genes[-1][1]

        print(np.asarray(genes)[:10][:, 1])
        self.genes = genes

    def load_precomputed_genes_from_dict(self, genes_dict, sort_genes=True):
        genes = []
        for gene in genes_dict.keys():
            genes.append([gene, float(genes_dict[gene])])
        if sort_genes:
            genes = sorted(genes, key=lambda x: x[1])
            assert genes[0][1] < genes[-1][1]

        print(np.asarray(genes)[:10][:, 1])
        self.genes = genes

    def generate_reduced_chromosomes(self, gene_names=None):
        if not gene_names:
            gene_names = self.get_genes_names()

        print("Working with total: {} genes".format(len(gene_names)), end=" ")
        reduced_chromosomes = []
        for chromosome in chromosomes:
            reduced_chromosome = []
            for gene in chromosome:
                if gene in gene_names:
                    reduced_chromosome.append(gene)
            reduced_chromosomes.append(reduced_chromosome)
        reduced_chromosome_size = sum([len(i) for i in reduced_chromosomes])
        print(",mapped into {} chromosomes".format(reduced_chromosome_size))
        self.experiment_chromosomes = reduced_chromosomes

    def get_genes_names(self):
        return list(np.asarray(self.genes)[:, 0])

    def get_genes_symbols(self, count=None):
        return list(np.asarray(self.genes)[:count, 0])

    def find_enriched_intervals(self, gene_group_size, verbose=True, ignore_errors=True, save_to_file=True,
                                include_group_size_in_chromosome=True):
        if verbose:
            print("Using {} genes".format(gene_group_size))

        enriched_intervals = ExperimentResults(self.experiment_name, self.output_directory)
        binary_chromosomes_map = self._create_binary_chromosomes_map(gene_group_size)

        set_Bs = self._get_set_Bs(binary_chromosomes_map)

        for chromosome_number, chromosome in enumerate(binary_chromosomes_map):
            printable_chromosome_number = utils.generate_printable_chromosome_number(chromosome_number)
            try:
                N = len(chromosome)
                B = sum(chromosome)
                if B < 3:
                    continue
                v = tuple(chromosome)

                try:
                    p, indexes = self.imhg_calculator.calculate_imhg(N, B, v)
                except RuntimeError:
                    if ignore_errors:
                        if verbose:
                            print("Failed to calculate imhg for chromosome: {}".format(printable_chromosome_number))
                        continue
                    raise RuntimeError(
                        "Failed to calculate imhg for chromosome: {}".format(printable_chromosome_number))
                if p == 0:
                    if ignore_errors:
                        if verbose:
                            print("imhg score for chromosome: {} is 0".format(printable_chromosome_number))
                        continue
                    raise RuntimeError("imhg score for chromosome: {} is 0".format(printable_chromosome_number))

                try:
                    mhg_p_value = self.imhg_calculator.calculate_p_value(N, B, p)
                except ValueError:
                    if ignore_errors:
                        if verbose:
                            print("Unable to calculate imhg p-value, either p-value is zero or amount of options is "
                                  "too small")
                        continue
                    raise RuntimeError(
                        "p-value calculation failed for chromosome: {}".format(printable_chromosome_number))

                bounded_p_value = N * mhg_p_value
                if bounded_p_value <= self.significance_threshold:
                    if include_group_size_in_chromosome:
                        printable_chromosome_number = str(gene_group_size) + "-" + printable_chromosome_number
                    enriched_intervals.add_result(
                        self.generate_chromosome_result(B, N, bounded_p_value, chromosome, indexes, set_Bs,
                                                        printable_chromosome_number))
                    print("Found enriched interval: {}, indices: {}".format(printable_chromosome_number,
                                                                            indexes))
            except Exception as e:
                print("Unknown error occurred")
                traceback.print_exc()
                raise e

        if save_to_file and len(enriched_intervals) > 0:
            self._assert_output_directory()
            enriched_intervals.save_to_file()

        return enriched_intervals

    def _assert_output_directory(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    @staticmethod
    def _get_set_Bs(results):
        total_b = 0
        for chromosome in results:
            total_b += sum(chromosome)
        return total_b

    def generate_chromosome_result(self, B, N, bounded_p_value, chromosome, indexes, set_Bs,
                                   printable_chromosome_number):
        interval_bs = int(sum(chromosome[indexes[0]:indexes[1] + 1]))
        interval_length = int(indexes[1] - indexes[0] + 1)
        return ChromosomeResult(self.experiment_name, printable_chromosome_number, bounded_p_value, interval_length,
                                interval_bs, int(B), int(N), int(set_Bs), str(indexes))

    def _create_binary_chromosomes_map(self, gene_group_size=None):
        if self.experiment_chromosomes is None:
            self.generate_reduced_chromosomes(self.get_genes_names())

        genome_main = utils.create_chromosomes_dict(self.experiment_chromosomes)
        binary_map = []
        symbols = self.get_genes_symbols(gene_group_size)

        for idx, c in enumerate(self.experiment_chromosomes):
            binary_map.append(np.zeros(len(c), dtype=int))

        # gene_error = False
        for symbol in symbols:
            try:
                a, b = genome_main[symbol]
            except KeyError:
                pass
            binary_map[a][b] = 1

        return binary_map
