"""Checkpointer."""
import os
import glob
import numpy as np
import torch
from dl import rng


class Checkpointer():
    """Save and load model and training state.

    RNG state is saved and loaded automatically.
    """

    def __init__(self, ckptdir, ckpt_period=None, format='{:09d}'):
        """Init."""
        self.ckptdir = ckptdir
        self.ckpt_period = ckpt_period
        self.format = format
        os.makedirs(ckptdir, exist_ok=True)

    def _all_steps(self) -> list[int]:
        """Get list of checkpoints."""
        ckpts = glob.glob(os.path.join(self.ckptdir, "*.pt"))
        # return sorted([int(c.split('/')[-1][:-3]) for c in ckpts])
        return sorted(int(os.path.basename(c)[:-3]) for c in ckpts)
    
    def _ckpt_path(self, t: int) -> str:
        """Convert checkpoint timestep to path."""
        return os.path.join(self.ckptdir, self.format.format(t) + '.pt')

    def save(self, payload: dict, t: int):
        """
        payload: any dict (may itself contain nested dicts / tensors)
        t      : current global step
        """
        # forbid overwriting newer checkpoints
        if self._all_steps() and t < max(self._all_steps()):
            raise ValueError(f"step {t} older than latest checkpoint")

        # stash RNG
        if "_rng" in payload:
            raise KeyError("'_rng' key reserved by Checkpointer")
        payload = payload | {"_rng": rng.get_state()}

        torch.save(payload, self._ckpt_path(t))
        self._prune()

    def load(self, t: int | None = None) -> dict | None:
        """
        Returns the saved dict (without touching it) and restores RNG.
        If no checkpoints exist and t is None, returns None.
        """
        steps = self._all_steps()
        if not steps:
            return None
        if t is None:
            t = steps[-1]
        if t not in steps:
            raise FileNotFoundError(f"no checkpoint at step {t}")

        obj = torch.load(self._ckpt_path(t), map_location="cpu",weights_only=False)
        rng.set_state(obj.pop("_rng"))
        return obj

    def _prune(self):
        if self.ckpt_period is None:
            return
        steps = np.array(self._all_steps())
        keep  = {steps[-1]}                     # always keep the latest
        periods = steps // self.ckpt_period
        for p in np.unique(periods):
            # keep the first ckpt of every period
            keep.add(steps[periods == p][0])
        for step in steps:
            if step not in keep:
                os.remove(self._ckpt_path(step))


if __name__ == '__main__':
    import unittest
    from shutil import rmtree

    class TestCheckpointer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            ckptr = Checkpointer('./.test_ckpt_dir', ckpt_period=10)
            for t in range(100):
                ckptr.save({'test': t},  t)

            assert ckptr.load()['test'] == 99

            assert ckptr._all_steps() == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
            for t in [0, 10, 50]:
                assert ckptr.load(t)['test'] == t

            try:
                ckptr.load(5)
                assert False
            except Exception:
                pass

            try:
                ckptr.save({'test': 1},  1)
                assert False
            except Exception:
                pass

            rmtree('.test_ckpt_dir')

            ckptr = Checkpointer('./.test_ckpt_dir')
            for t in range(100):
                ckptr.save({'test': t},  t)
            for t in range(100):
                assert os.path.exists(ckptr._ckpt_path(t))
            rmtree('.test_ckpt_dir')

        def test_rng(self):
            ckptr = Checkpointer('./.test_ckpt_dir', ckpt_period=10)
            rng.seed(0)
            ckptr.save({}, 0)
            r1 = np.random.rand(10)
            ckptr.load(0)
            r2 = np.random.rand(10)
            assert np.allclose(r1, r2)
            rmtree('.test_ckpt_dir')

    unittest.main()