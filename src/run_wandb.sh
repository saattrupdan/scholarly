#!/bin/sh
python main.py --epochs 100 --fname arxiv_data_test --mcat_min 0.1 --mcat_factor 0.9 --name factor_test
python main.py --epochs 100 --fname arxiv_data_test --mcat_min 0.1 --mcat_factor 0.8 --name factor_test
python main.py --epochs 100 --fname arxiv_data_test --mcat_min 0.1 --mcat_factor 0.7 --name factor_test
