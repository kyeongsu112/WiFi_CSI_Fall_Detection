from datasets.manifest import binary_label


def test_binary_label_mapping():
    assert binary_label("fall") == "fall"
    assert binary_label("walk") == "non_fall"
