################################################################################
#                                                                               #
# This module serves as a configuration file, defining default parameters for   #
# the deep learning training process. Defines Enums for training modes,         #
# activation functions, and model types. It sets default values for parameters. #
# The provided configurations                                                   #
#                                                                               #
################################################################################

from enum import Enum
import torch.nn as nn
from Model import *

TrainingMode = Enum("TrainingMode", ["Sample", "Mini_Batch"])
Activation = Enum("Activation", ["Sigmoid", "ReLU", "Tanh"])
Models = Enum("Models", ["Linear", "ANN"])

LEARNING_RATE = 1e-3
epochs = 100
MODE = TrainingMode.Mini_Batch

model = LinearRegression()
dataset_size = 50

BATCH_SIZE = 10
activation_fn_sigmoid = nn.Sigmoid()
activation_fn_relu = nn.ReLU()


if MODE == TrainingMode.Sample:
    BATCH_SIZE = 1
