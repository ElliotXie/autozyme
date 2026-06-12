from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _restore_autozyme_between_tests():
    """Keep monkey-patched upstream modules from leaking between tests."""
    import autozyme

    autozyme.deactivate_all()
    try:
        yield
    finally:
        autozyme.deactivate_all()
