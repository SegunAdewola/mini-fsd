import minifsd


def test_package_imports_with_version() -> None:
    assert minifsd.__version__ == "0.1.0"
