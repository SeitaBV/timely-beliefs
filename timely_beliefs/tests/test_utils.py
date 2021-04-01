from timely_beliefs.utils import remove_class_init_kwargs


def test_remove_used_kwargs():
    class MyCls:
        def __init__(self, a, b):
            pass

    kwargs = dict(a=1, b=2, c=3, d=4, self=5)
    remaining_kwargs = remove_class_init_kwargs(MyCls, kwargs)
    assert remaining_kwargs == dict(c=3, d=4, self=5)
