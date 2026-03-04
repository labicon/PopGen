import warnings

_FILTERS_CONFIGURED = False


def configure_warning_filters(*, force: bool = False) -> None:
    """Install project-wide warning filters once per process."""
    global _FILTERS_CONFIGURED
    if _FILTERS_CONFIGURED and not force:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*Pydantic serializer warnings.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*(UnboundedContinuousTensorSpec|DiscreteTensorSpec|CompositeSpec).*(deprecated|removed in v0\.8).*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*reward returned by `step\(\)`.*actual type: <class 'list'>.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*TaskConfig python dataclass not found, task is being loaded without type checks.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*(UnboundedContinuousTensorSpec|DiscreteTensorSpec|CompositeSpec).*(deprecated|removed in v0\.8).*",
        category=FutureWarning,
    )

    _FILTERS_CONFIGURED = True
