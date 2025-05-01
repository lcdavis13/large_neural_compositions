import argparse
import csv
import itertools
import sys
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
        if expected_type == int:
            end += 1
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


class HyperparameterComposer:
    def __init__(self, hyperparam_csv: str = None, cli_args: List[str] = None):
        # default to not use global CLI, must be passed in explicitly, while avoiding a mutable default in the function signature
        self.cli_args = cli_args if cli_args is not None else []

        self.params = {}
        self.categories = {}
        self.parser = argparse.ArgumentParser(description="Experiment Hyperparameters")
        self.hyperparam_csv = hyperparam_csv
        self.load_mode = hyperparam_csv is not None

        # params for tracking state in CSV load mode
        self.category_call_order = []
        self.per_category_multiplier = {}
        self.csv_data = []
        self.csv_index = 0
        self.innermost_call_count = 0
        self.innermost_category = None
        self.innermost_found = False
        self.outer_product = 1000000

        if self.load_mode:
            with open(hyperparam_csv, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                self.csv_data = list(reader)




    def add_param(self, name: str, *default_values: Any, category: str = None, help: str = "") -> "HyperparameterComposer":
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
            "help": help,
            "is_flag": False,
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

    def add_flag(self, name: str, default: bool, category: str=None, help: str = "") -> "HyperparameterComposer":
        """
        Add a boolean flag that does not support multiple values.
        Uses default only if no CLI args are provided (e.g. run from IDE).
        """
        self.params[name] = {
            "type": bool,
            "help": help,
            "is_flag": True,
            "category": category,
        }

        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)

        # Detect whether running from CLI or IDE
        cli_invoked = len(sys.argv) > 1

        if cli_invoked:
            # Typical flag behavior: False by default, True if flag is present
            self.parser.add_argument(
                f"--{name}",
                action="store_true",
                help=f"{help} (default: False, set to true by passing the flag)",
            )
        else:
            # IDE-like behavior: use provided default
            self.parser.add_argument(
                f"--{name}",
                action="store_false" if default else "store_true",
                help=f"{help} (default: False, set to true by passing the flag)",
            )

        return self

    def _parse_and_generate_combinations_for_rawvalues(self, args: dict) -> List[Dict[str, Any]]:
        # Process parameters and types
        fixed_param_lists = {}
        random_param_flags = {}

        param_names = list(args.keys())
        category = self.params[param_names[0]]["category"] if param_names else None

        for name in param_names:
            config = self.params[name]
            raw_value = args[name]

            if not config["is_flag"]:
                if isinstance(raw_value, str) and raw_value.startswith("[") and raw_value.endswith("]"):
                    # Mark this param as needing random sampling
                    random_param_flags[name] = (raw_value, config["type"])
                    fixed_param_lists[name] = [None]  # Placeholder
                else:
                    # Parse comma-separated fixed values
                    if isinstance(raw_value, str):
                        values = raw_value.split(",")
                    else:
                        values = [raw_value]
                    if config["type"] == bool:
                        values = [str2bool(v) for v in values]
                    fixed_param_lists[name] = [config["type"](v) for v in values]
            else:
                # Single value for flags
                if config["type"] == bool:
                    raw_value = str2bool(raw_value)
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


    def parse_and_generate_combinations(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse command-line arguments, handle types, and generate combinations.
        Only include parameters from the matching category.

        :param category: Optional subcategory to filter by.
        :return: List of dictionaries representing all hyperparameter combinations.
        """

        # Filter parameters by category
        param_names = self.categories.get(category, [])

        # run either csv or argparse mode
        if self.load_mode:
            return self._parse_csv_mode(param_names, category)
        else:
            return self._parse_argparse_mode(param_names, category)
        
    def _parse_argparse_mode(self, param_names: List[str], category: Optional[str] = None) -> List[Dict[str, Any]]:
        # Argparse mode (uses permutation and random selection)
        
        args = vars(self.parser.parse_args(self.cli_args))
        
        # Filter parameters by category
        param_names_subset = [name for name in param_names if name in args]
        args = {name: args[name] for name in param_names_subset}

        # # retrieve defaults for missing params in category
        # for name in param_names:
        #     if name not in args:
        #         args[name] = self.params[name]["defaults"]

        return self._parse_and_generate_combinations_for_rawvalues(args)

    def _parse_csv_mode(self, param_names: List[str], category: Optional[str] = None) -> List[Dict[str, Any]]:

        # Step 1: Record call order
        outermost = False
        if category not in self.category_call_order:
            if len(self.category_call_order) == 0:
                outermost = True

            self.category_call_order.append(category)

        # If outermost category, we want to use all CSV rows immediately (this is the only time it will be called)
        if outermost:
            indices = list(range(len(self.csv_data)))
        # Otherwise, we only want to grab the current row from CSV.
        else:
            # figure out what the innermost category is
            if not self.innermost_found: 
                # the first time we find a repeated category, set it as the innermost category
                if category in self.per_category_multiplier: # relies on the fact that this dict isn't filled until later in the function
                    # print(f"Category '{category}' is already in per_category_multiplier, setting as innermost category.")
                    self.innermost_category = category
                    self.innermost_call_count = 0 # we know it was called once previously, but we'll increment it below
                    self.innermost_found = True
                    self.outer_product = 1
                    for cat in self.category_call_order[:-1]:  # exclude innermost
                        self.outer_product *= self.per_category_multiplier[cat]
                    # print (f"Outer product is {self.outer_product}")
                    # print(f"from categories {[self.per_category_multiplier[i] for i in self.category_call_order[:-1]]}")

            # Step 5: If this is the innermost category, increment row every a*b calls
            is_innermost = category == self.innermost_category
            if is_innermost:
                self.innermost_call_count += 1

                if self.innermost_found and self.innermost_call_count >= self.outer_product:
                    self.csv_index += 1
                    self.innermost_call_count = 0

            # We've decided whether or not we needed to increment the index, now grab row
            indices = [self.csv_index]

        defaults = vars(self.parser.parse_args(self.cli_args))

        # This section is basically equivalent to _parse_argparse_mode but once per row we've retrieved.
        expand_rate = 1
        combos = []
        for i in indices:
            # print(i)
            args = self.csv_data[i]
            
            # Filter parameters by category
            param_names_subset = [name for name in param_names if name in args]
            args = {name: args[name] for name in param_names_subset}

            # retrieve defaults for missing params in category
            for name in param_names:
                if name not in args:
                    args[name] = defaults[name]
                    # print(f"Param '{name}' not found in CSV row {i}, using default value: {args[name]}")

            # print(args)

            # Step 3: Parse combinations for this category
            new_combos = self._parse_and_generate_combinations_for_rawvalues(args)
            expand_rate = len(new_combos)
            combos.extend([dicy(new_combo) for new_combo in new_combos])

        # Below is logic to track if we need to increment the CSV index for our inner category loops. Only increment when the outer loops have all completed a single pass through their respective expansion rates.


        # Step 4: Track multiplier on first call
        if category not in self.per_category_multiplier:
            self.per_category_multiplier[category] = expand_rate

        # Step 6: Return results as-is (no repetition or alignment necessary)
        return combos




# test
if __name__ == "__main__":
    builder = HyperparameterComposer()
    builder.add_param("learning_rate", 0.01, 0.001, category="model", help="Learning rate for training")
    builder.add_param("batch_size", 32, 64, category="data", help="Batch size for training")
    builder.add_flag("use_augmentation", True, category="data", help="Whether to use data augmentation")

    data_combinations = builder.parse_and_generate_combinations(category="data")
    print("Data combinations:", data_combinations)

    model_combinations = builder.parse_and_generate_combinations(category="model")
    print("Model combinations:", model_combinations)
