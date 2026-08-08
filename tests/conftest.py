def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: runs full optimizations, so it takes seconds rather than "
        "milliseconds. Deselect with -m 'not slow' while iterating; CI runs "
        "them, as they are what catches a change that quietly searches worse.",
    )
