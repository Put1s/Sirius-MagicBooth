class ObjectGenerator:
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)


class Dict2Obj:
    def __init__(self, d: dict):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
