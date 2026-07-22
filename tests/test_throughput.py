import time

import numpy as np

from pixgrep.throughput import measure


def test_measure_returns_positive_rate():
    def fake_embed(images):
        time.sleep(0.001 * len(images))
        return np.zeros((len(images), 4), dtype=np.float32)

    imgs = list(range(20))
    rate = measure(fake_embed, imgs, warmup=2)
    assert rate > 0.0


def test_empty_input_returns_zero():
    assert measure(lambda x: x, [], warmup=0) == 0.0
