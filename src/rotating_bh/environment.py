"""Report the Python and scientific-package versions in use."""

from importlib.metadata import PackageNotFoundError, version
import platform


REQUIRED = ("numpy", "scipy", "sympy", "matplotlib")


def environment_report() -> dict[str, object]:
    """Return the interpreter version and installed required packages."""
    packages: dict[str, str | None] = {}
    for name in REQUIRED:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    return {"python": platform.python_version(), "packages": packages}
