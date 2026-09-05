# Minimal stub of BiRefNet's config.Config for the swin_v1 backbone oracle. The backbone reads only
# SDPA_enabled; keeping it False takes the explicit softmax path the MLX port reproduces.
class Config:
    def __init__(self):
        self.SDPA_enabled = False
