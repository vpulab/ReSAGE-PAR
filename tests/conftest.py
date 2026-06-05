import sys
import types
from pathlib import Path

import pytest


# Ensure repo root is on sys.path so `import src...` works when tests/ is parallel to src/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_package(fullname: str) -> types.ModuleType:
    """Ensure a (possibly nested) package exists in sys.modules."""
    if fullname in sys.modules:
        return sys.modules[fullname]
    pkg = types.ModuleType(fullname)
    pkg.__path__ = []  # mark as package
    sys.modules[fullname] = pkg
    if "." in fullname:
        parent_name, child = fullname.rsplit(".", 1)
        parent = _ensure_package(parent_name)
        setattr(parent, child, pkg)
    return pkg


@pytest.fixture
def install_module(monkeypatch):
    """Factory fixture to create/install a module (and parent packages) into sys.modules."""

    def _install(fullname: str) -> types.ModuleType:
        if "." in fullname:
            parent_name, child = fullname.rsplit(".", 1)
            _ensure_package(parent_name)
        mod = types.ModuleType(fullname)
        sys.modules[fullname] = mod
        if "." in fullname:
            parent_name, child = fullname.rsplit(".", 1)
            parent = sys.modules[parent_name]
            setattr(parent, child, mod)
        return mod

    return _install
