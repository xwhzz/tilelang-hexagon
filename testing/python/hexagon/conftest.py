"""Hooks for the Hexagon tests."""

import pytest


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    # A kernel's device session holds the device, and the device lock that
    # parallel workers share, until it closes; a later call reopens it. Close
    # the sessions a test leaves open, so an idle worker holds nothing.
    from tilelang.hexagon.runtime.adapter import HexagonKernelAdapter

    HexagonKernelAdapter.close_all()
