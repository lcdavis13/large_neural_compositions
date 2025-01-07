import argparse
import itertools
from typing import Any, List, Dict
from dotsy import dicy


class HyperparameterBuilder:
    def __init__(self):
        self.params = {}
        self.parser = argparse.ArgumentParser(description="Experiment Hyperparameters")

    def add_param(self, name: str, *default_values: Any, help: str = "") -> "HyperparameterBuilder":
        """
        Add a parameter that supports multiple values (comma-separated or defaults).
        - `name`: Parameter name.
        - `default_values`: One or more default values of the same type.
        - `help`: Help string for argparse.
        """
        if not default_values:
            raise ValueError(f"Parameter '{name}' must have at least one default value.")

        # Infer type from the first value
        param_type = type(default_values[0])
        if not all(isinstance(value, param_type) for value in default_values):
            raise ValueError(f"All default values for '{name}' must be of the same type.")

        self.params[name] = {
            "type": param_type,
            "defaults": default_values,
            "help": help,
            "multiple": True,
        }

        # Add argparse argument
        self.parser.add_argument(
            f"--{name}",
            default=",".join(map(str, default_values)),
            type=str,
            help=f"{help} (default: {','.join(map(str, default_values))})",
        )
        return self

    def add_flag(self, name: str, default: bool, help: str = "") -> "HyperparameterBuilder":
        """
        Add a boolean flag that does not support multiple values.
        - `name`: Parameter name.
        - `default`: Default boolean value.
        - `help`: Help string for argparse.
        """
        self.params[name] = {
            "type": bool,
            "defaults": [default],
            "help": help,
            "multiple": False,
        }

        # Add argparse argument
        action = "store_true" if not default else "store_false"
        self.parser.add_argument(
            f"--{name}",
            action=action,
            help=f"{help} (default: {default})",
        )
        return self

    def parse_and_generate_combinations(self) -> List[Dict[str, Any]]:
        """
        Parse command-line arguments, handle types, and generate combinations.
        :return: List of dictionaries representing all hyperparameter combinations.
        """
        args = vars(self.parser.parse_args())

        # Process parameters and types
        param_lists = {}
        for name, config in self.params.items():
            raw_value = args[name]

            if config["multiple"]:
                # Parse comma-separated values
                values = raw_value.split(",")
                param_lists[name] = [config["type"](v) for v in values]
            else:
                # Single value for flags
                param_lists[name] = [config["type"](raw_value)]

        # Generate combinations
        combinations = [
            dicy(dict(zip(param_lists.keys(), values)))
            for values in itertools.product(*param_lists.values())
        ]
        return combinations