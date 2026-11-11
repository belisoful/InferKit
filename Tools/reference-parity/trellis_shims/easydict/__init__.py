class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        super().__init__()
        for k, v in dict(d or {}, **kwargs).items():
            self[k] = v

    def __setitem__(self, k, v):
        if isinstance(v, dict) and not isinstance(v, EasyDict):
            v = EasyDict(v)
        elif isinstance(v, (list, tuple)):
            v = type(v)(EasyDict(x) if isinstance(x, dict) else x for x in v)
        super().__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v
