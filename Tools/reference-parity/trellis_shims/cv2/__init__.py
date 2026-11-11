"""Import-time stand-in: cv2 is not on TRELLIS's mesh inference path. Attribute access returns a
placeholder so annotations resolve; calling anything raises."""


class _Unavailable:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(self._name + "." + attr)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(self._name + " is not available in the TRELLIS CPU oracle")


def __getattr__(attr):
    if attr.startswith("__"):
        raise AttributeError(attr)
    return _Unavailable("cv2." + attr)
