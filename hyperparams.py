import argparse
import itertools
from typing import Any, List, Dict, Optional
from dotsy import dicy
import random

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    elif value.lower() in ("false", "0", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false, 1/0, yes/no).")
    

def parse_random_value(value: str, expected_type: type) -> Any:
    """
    Parse values of the formats either "[X]", "[A...B]", or "[A^^^B]" and return a random value. 
    If the expected_type is a boolean, it will attempt to interpret the interior of the brackets as a single float X representing probability for the True condition. If the value cannot be interpreted this way, it will default to 0.5. 
    If the expected_type is a numeric type, it will interpret the interior of the brackets as a range of values from A to B which will be sampled uniformly. If using "..." the range is sampled linearly, if using "^^^" the range is sampled logarithmically.
    Otherwise it raises an error.
    """
    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError(f"Value '{value}' must be enclosed in brackets.")

    value = value[1:-1]
    if "..." in value:
        start, end = map(expected_type, value.split("..."))
        return expected_type(random.uniform(start, end))
    elif "^^^" in value:
        start, end = map(expected_type, value.split("^^^"))
        return expected_type(start * (end / start) ** random.uniform(0.0, 1.0))
    else:
        if expected_type == bool:
            try:
                return expected_type(float(value))
            except ValueError:
                return False
        else:
            return expected_type(value)


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

        if param_type == bool:
            param_type = str2bool  # Use custom bool parsing

        self.params[name] = {
            "type": param_type,
            "defaults": default_values,
            "help": help,
            "multiple": True,
            "category": category,
        }

        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)

        # Add argparse argument
        self.parser.add_argument(
            f"--{name}",
            default=",".join(map(str, default_values)),
            type=str if param_type == str2bool else str,
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
        fixed_param_lists = {}
        random_param_flags = {}

        for name in param_names:
            config = self.params[name]
            raw_value = args[name]

            if config["multiple"]:
                if isinstance(raw_value, str) and raw_value.startswith("[") and raw_value.endswith("]"):
                    # Mark this param as needing random sampling
                    random_param_flags[name] = (raw_value, config["type"])
                    fixed_param_lists[name] = [None]  # Placeholder
                else:
                    # Parse comma-separated fixed values
                    values = raw_value.split(",")
                    fixed_param_lists[name] = [config["type"](v) for v in values]
            else:
                # Single value for flags
                fixed_param_lists[name] = [config["type"](raw_value)]

        # Generate combinations
        combinations = []
        fixed_param_names = [name for name in param_names if name not in random_param_flags]

        for idx, fixed_values in enumerate(itertools.product(*[fixed_param_lists[name] for name in fixed_param_names])):
            combination = {}

            # Assign fixed parameters
            for name, value in zip(fixed_param_names, fixed_values):
                combination[name] = value

            # Sample random parameters independently for each combination
            for name, (raw_value, value_type) in random_param_flags.items():
                combination[name] = parse_random_value(raw_value, value_type)

            # Add configid
            configid_key = f"{category}_configid" if category else "configid"
            combination[configid_key] = idx

            combination = dicy(combination)  # Convert to dotsy.dicy object

            combinations.append(combination)

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
