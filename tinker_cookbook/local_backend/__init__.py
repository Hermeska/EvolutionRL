__all__ = ["LocalServiceClient"]


def __getattr__(name):
    if name == "LocalServiceClient":
        from tinker_cookbook.local_backend.client import LocalServiceClient

        return LocalServiceClient
    raise AttributeError(name)
