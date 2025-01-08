import argparse
import itertools
from typing import Any, List, Dict, Optional
from dotsy import dicy


class HyperparameterBuilder:
    def __init__(self):
        self.params = {}
        self.categories = {}
        self.parser = argparse.ArgumentParser(description="Experiment Hyperparameters")

    def add_param(self, name: str, *default_values: Any, category: str = None, help: str = "") -> "HyperparameterBuilder":
        """
        Add a parameter that supports multiple values (comma-separated or defaults).
        - `name`: Parameter name.
        - `default_values`: One or more default values of the same type.
        - `category`: Subcategory for grouping this parameter.
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
            "category": category,
        }

        # Add to category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)

        # Add argparse argument
        self.parser.add_argument(
            f"--{name}",
            default=",".join(map(str, default_values)),
            type=str,
            help=f"{help} (default: {','.join(map(str, default_values))})",
        )
        return self

    def add_flag(self, name: str, default: bool, category: str=None, help: str = "") -> "HyperparameterBuilder":
        """
        Add a boolean flag that does not support multiple values.
        - `name`: Parameter name.
        - `default`: Default boolean value.
        - `category`: Subcategory for grouping this parameter.
        - `help`: Help string for argparse.
        """
        self.params[name] = {
            "type": bool,
            "defaults": [default],
            "help": help,
            "multiple": False,
            "category": category,
        }

        # Add to category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)

        # Add argparse argument
        action = "store_true" if not default else "store_false"
        self.parser.add_argument(
            f"--{name}",
            action=action,
            help=f"{help} (default: {default})",
        )
        return self

    def parse_and_generate_combinations(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse command-line arguments, handle types, and generate combinations.
        Only include parameters from the matching category.

        :param category: Optional subcategory to filter by.
        :return: List of dictionaries representing all hyperparameter combinations.
        """
        args = vars(self.parser.parse_args())

        # Filter parameters by category
        param_names = self.categories.get(category, [])

        # Process parameters and types
        param_lists = {}
        for name in param_names:
            config = self.params[name]
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

# test
if __name__ == "__main__":
    builder = HyperparameterBuilder()
    builder.add_param("learning_rate", 0.01, 0.001, category="model", help="Learning rate for training")
    builder.add_param("batch_size", 32, 64, category="data", help="Batch size for training")
    builder.add_flag("use_augmentation", True, category="data", help="Whether to use data augmentation")

    data_combinations = builder.parse_and_generate_combinations(category="data")
    print("Data combinations:", data_combinations)

    model_combinations = builder.parse_and_generate_combinations(category="model")
    print("Model combinations:", model_combinations)
