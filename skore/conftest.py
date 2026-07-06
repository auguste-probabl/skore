"""Top-level conftest applying to both `tests/` and doctests under `src/`."""

import matplotlib.pyplot as plt


def pytest_runtest_teardown(item):
    """Close any matplotlib figures left open by the test.

    Guards against silent figure accumulation across the suite (which used to
    trip matplotlib's `figure.max_open_warning`). Applied via a hook rather
    than an autouse fixture so it also covers doctests.
    """
    plt.close("all")
