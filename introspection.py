import inspect


def call(fn, full_args: dict):
    """
    Call a function with only the arguments it accepts, filtering out any
    extra arguments that are not part of the function's signature.
    """
    sig = inspect.signature(fn)
    accepted_args = {
        k: v for k, v in full_args.items()
        if k in sig.parameters
    }
    return fn(**accepted_args)


def construct(cls, base_kwargs: dict, override_args: dict = None):
    """
    Construct `cls` using arguments from `base_kwargs` and `override_args`,
    passing only the arguments accepted by `cls.__init__`.
    """
    if override_args is None:
        override_args = {}

    combined_args = {**base_kwargs, **override_args}
    return call(cls, combined_args)

