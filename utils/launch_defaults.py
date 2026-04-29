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
