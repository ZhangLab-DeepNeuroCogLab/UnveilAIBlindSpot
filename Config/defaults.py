#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
import math
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# Mode of the configuration (Normal, OOD, AA)
_C.MODE = "Normal"

_C.DATA_SPLIT_MODE = "NA"

_C.ALPHA = 100.0
# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = []

# Root path of Dataset.
_C.TRAIN.DATASET_ROOT = []

# Dataset split for training.
_C.TRAIN.SPLIT = ""

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = True

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = []

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Test model every test period epochs.
_C.TEST.TEST_PERIOD = 10


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.BASE_MODEL = CfgNode()

# Model name
_C.BASE_MODEL.MODEL_NAME = "vgg11" 

# Path to the checkpoint to load the initial weight.
_C.BASE_MODEL.CHECKPOINT_FILE_PATH = ""

# The number of classes to predict for the model.
_C.BASE_MODEL.NUM_CLASSES = 400

# Use pretrained model or not (for base model training only)
_C.BASE_MODEL.PRETRAIN = False

# -----------------------------------------------------------------------------
# Teacher options
# -----------------------------------------------------------------------------
_C.TEACHER = CfgNode()

# Model name
_C.TEACHER.MODEL_NAME = "vgg11" 

_C.TEACHER.NUM_CLASSES = 400

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = 1.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Output basedir.
_C.OUTPUT_DIR = "."

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training/testing process.
_C.DATA_LOADER.NUM_WORKERS = 8



def assert_and_infer_cfg(cfg):
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
