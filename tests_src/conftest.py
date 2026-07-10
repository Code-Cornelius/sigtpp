"""
conftest.py, run fast tests first, slow tests last.

Slow test files (training loops, integration, smoke matrix) are deferred to the end of the collection so that fast unit-test failures are visible immediately without waiting for training runs to complete.
"""

_SLOW_FILE_PATTERNS = {
    "test_train_model_sigtpp.py",
    "test_training_manager_integration.py",
    "test_sample_for_fixed_batch.py",
}


def pytest_collection_modifyitems(items):
    fast, slow = [], []
    for item in items:
        if item.fspath.basename in _SLOW_FILE_PATTERNS:
            slow.append(item)
        else:
            fast.append(item)
    items[:] = fast + slow
