import os


MODEL_PATH_ENV = "CUBIC_MODEL_PATH"


def get_default_model_path() -> str | None:
    model_path = os.environ.get(MODEL_PATH_ENV)
    if model_path is None:
        return None
    model_path = model_path.strip()
    return model_path or None


def resolve_model_path(model_path: str | None, arg_name: str) -> str:
    resolved = model_path or get_default_model_path()
    if resolved is None:
        raise ValueError(
            f"Missing model path. Pass {arg_name} explicitly or launch through launch.py "
            f"so {MODEL_PATH_ENV} is populated."
        )
    return resolved


def resolve_required_arg(value, arg_name: str, description: str):
    if value is None:
        raise ValueError(
            f"Missing {description}. Pass {arg_name} explicitly or launch through launch.py."
        )
    if isinstance(value, str):
        resolved = value.strip()
        if not resolved:
            raise ValueError(f"Empty {description}. Pass a non-empty value via {arg_name}.")
        return resolved
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(
                f"Missing {description}. Pass one or more values via {arg_name} or launch through launch.py."
            )
    return value
