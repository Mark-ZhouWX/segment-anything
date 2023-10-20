import argparse
import collections
import copy
import os
from copy import deepcopy
from typing import Union, Any, Sequence

import yaml
from omegaconf import OmegaConf
from omegaconf._utils import get_yaml_loader

from segment_anything.utils import logger

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

__all__ = ["parse_args"]


def parse_args(parser_config):
    args_cmd= parser_config.parse_args()

    args = OmegaConf.create()
    cfg_file = args_cmd.config
    # merge command line config and file config
    if cfg_file is not None:
        assert os.path.exists(cfg_file), f'config file not exist at {cfg_file}'
        cfg, _, _ = load_config(args_cmd.config)
        cfg = OmegaConf.create(cfg)
        args.merge_with(cfg)
    if args_cmd.override_cfg is not None:
        args.merge_with_dotlist(args_cmd.override_cfg)

    return args


def load_config(file_path):
    BASE = "__BASE__"
    assert os.path.splitext(file_path)[-1] in [".yaml", ".yml"], f"[{file_path}] not yaml format."
    cfg_default, cfg_helper, cfg_choices = _parse_yaml(file_path)

    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE in cfg_default:
        all_base_cfg_default = {}
        all_base_cfg_helper = {}
        all_base_cfg_choices = {}
        base_yamls = list(cfg_default[BASE])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)
            if not base_yaml.startswith("/"):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            base_cfg_default, base_cfg_helper, base_cfg_choices = load_config(base_yaml)
            all_base_cfg_default = _merge_config(base_cfg_default, all_base_cfg_default)
            all_base_cfg_helper = _merge_config(base_cfg_helper, all_base_cfg_helper)
            all_base_cfg_choices = _merge_config(base_cfg_choices, all_base_cfg_choices)

        del cfg_default[BASE]
        return (
            _merge_config(cfg_default, all_base_cfg_default),
            _merge_config(cfg_helper, all_base_cfg_helper),
            _merge_config(cfg_choices, all_base_cfg_choices),
        )

    return cfg_default, cfg_helper, cfg_choices


def _parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, "r") as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=get_yaml_loader())
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg = cfgs[0]
                cfg_helper = {}
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def _merge_config(config, base):
    """Merge config"""
    new = deepcopy(base)
    for k, v in config.items():
        if k in new and isinstance(new[k], dict) and isinstance(config[k], collectionsAbc.Mapping):
            new[k] = _merge_config(config[k], new[k])
        else:
            new[k] = config[k]
    return new
