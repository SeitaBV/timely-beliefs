def equal_lists(list_a: list, list_b: list):
    return all(a == b for a, b in zip(list_a, list_b))
