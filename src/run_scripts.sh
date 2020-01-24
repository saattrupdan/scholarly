#!/bin/sh
python simple_models.py --fname arxiv_data_mini_pp --model_type naive_bayes
python simple_models.py --fname arxiv_data_mini_pp --model_type logreg
python simple_models.py --fname arxiv_data_mini_pp --model_type forest
