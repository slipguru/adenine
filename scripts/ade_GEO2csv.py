#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import argparse
import pandas as pd

from adenine.utils import GEO2csv
from adenine import __version__


def main():
    """Adenine GEO2csv main script."""
    parser = argparse.ArgumentParser(description='Adenine script for '
                                                 'GEO2csv conversion.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument('accession_number', help='GEO DataSets Accession number')
    parser.add_argument('--label_field', dest='pheno_name',
                        default='title', help='The field in which '
                        'phenotypes information are stored.')
    parser.add_argument('--phenotypes', '--pheno', dest='pheno',
                        action='store', default=None,
                        help='Select samples by their phenotypes ('
                        'comma separated) e.g.: Severe,Mild,Control,...')
    parser.add_argument('--gene_symbol', action='store_true', dest='gs',
                        help='Use this option to convert the platform IDs '
                        'to gene symbols')
    parser.add_argument('--signature', dest='signature',
                        default=None, help='Generate a data matrix comprising '
                        'only the genes in the signature.')
    args = parser.parse_args()

    # Get the data
    try:
        if args.gs or (args.signature is not None):
            data, gse = GEO2csv.get_GEO(args.accession_number, args.pheno_name, True)
        else:
            data = GEO2csv.get_GEO(args.accession_number, args.pheno_name)[0]
        print('* GEO dataset {} loaded'.format(args.accession_number))

        # Filter samples per phenotype
        if args.pheno is not None:
            data = GEO2csv.GEO_select_samples(
                    data.data, data.target, selected_labels=args.pheno.split(','),
                    index=data.index, feature_names=data.feature_names)
            print('* Phenotypes {}'.format(args.pheno))

        if args.gs or (args.signature is not None):
            data = GEO2csv.id2gs(data, gse)
            print('* Probe ID converted to gene symbols')

        if args.signature is not None:
            data = GEO2csv.restrict_to_signature(data, args.signature.split(','))
            print('* Dataset restricted to {}'.format(data.feature_names))

        # Save dataset
        pd.DataFrame(data=data.data, columns=data.feature_names,
                     index=data.index).to_csv('{}_data.csv'.format(args.accession_number))
        print('* {}_data.csv created: {} samples x {} features'.format(args.accession_number,
                                                                     *data.data.shape))
        pd.DataFrame(data=data.target, columns=['Phenotype'],
                     index=data.index).to_csv('{}_labels.csv'.format(args.accession_number))
        print('* {}_labels.csv created: {} samples'.format(args.accession_number,
                                                         len(data.target)))

    except Exception as e:
        print('Raised {}'.format(e))
        raise ValueError('Cannot parse {}. Check '
        'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={}'
        ' for more info on the GEO series'.format(args.accession_number,
        args.accession_number))


if __name__ == '__main__':
    main()
