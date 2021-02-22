
import torch.nn as nn
import torch.optim as torch_optim

from .common import build_from_cfg
from .registry import (ENVIRONMENTS, AGENTS, ALGORITHMS, MEMORIES, POLICIES,
                       BACKBONES, HEADS, MODELS, RUNNERS)


def build_env(cfg, default_args=None):
    env = build_from_cfg(cfg, ENVIRONMENTS, default_args)
    return env


def build_agent(cfg, default_args=None):
    agent = build_from_cfg(cfg, AGENTS, default_args)
    return agent


def build_algorithm(cfg, default_args=None):
    algorithm = build_from_cfg(cfg, ALGORITHMS, default_args)
    return algorithm


def build_memory(cfg, default_args=None):
    memory = build_from_cfg(cfg, MEMORIES, default_args)
    return memory


def build_policy(cfg, default_args=None):
    policy = build_from_cfg(cfg, POLICIES, default_args)
    return policy


def build_backbone(cfg, default_args=None):
    backbone = build_from_cfg(cfg, BACKBONES, default_args)
    return backbone


def build_head(cfg, default_args=None):
    head = build_from_cfg(cfg, HEADS, default_args)
    return head


def build_model(cfg, default_args=None):
    model = build_from_cfg(cfg, MODELS, default_args)
    return model


def build_optimizer(cfg, default_args=None):
    model = build_from_cfg(cfg, torch_optim, default_args, 'module')
    return model


def build_criterion(cfg, default_args=None):
    model = build_from_cfg(cfg, nn, default_args, 'module')
    return model


def build_runner(cfg, default_args=None):
    runner = build_from_cfg(cfg, RUNNERS, default_args)
    return runner
