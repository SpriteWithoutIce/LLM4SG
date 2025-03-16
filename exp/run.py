import argparse
from argparse import Namespace
import yaml
import decision_tree
import knn
import logistic
import random_forest
import svm

def load_args(file_path):
    with open(file_path, 'r') as f:
        args = yaml.safe_load(f)
    args = Namespace(**args)
    return args


parser = argparse.ArgumentParser(
    description='Load parameters from a YAML file.')
parser.add_argument('--config', type=str, default='config/bay_point.yaml')
parameters = parser.parse_args()

# load args
global args
args = load_args(parameters.config)

train_path = args.trainPath
test_path = args.testPath
output_path = "./results/" + args.outputPath

decision_tree.decision_tree(train_path, test_path, output_path)
knn.knn(train_path, test_path, output_path)
logistic.logistic(train_path, test_path, output_path)
random_forest.RandomForest(train_path, test_path, output_path)
svm.SVM(train_path, test_path, output_path)
