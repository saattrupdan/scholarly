#!/bin/sh
python simple_models.py --workers 1 --model_type naive_bayes --use_fasttext false
python simple_models.py --workers 1 --model_type svm --use_fasttext false
python simple_models.py --workers 1 --model_type logreg --use_fasttext false

python simple_models.py --workers 1 --model_type naive_bayes --use_fasttext true
python simple_models.py --workers 1 --model_type svm --use_fasttext true
python simple_models.py --workers 1 --model_type logreg --use_fasttext true
