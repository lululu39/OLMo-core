from types import SimpleNamespace

import pytest

from olmo_core.train.callbacks.wandb import WandBCallback


@pytest.mark.parametrize("exit_code", [0, 1])
def test_finalize_supports_current_wandb_signature(exit_code):
    calls = []

    # W&B 0.29 removed the deprecated quiet argument from finish().
    def finish(*, exit_code=None):
        calls.append(exit_code)

    callback = WandBCallback()
    callback._wandb = SimpleNamespace(finish=finish)
    callback.finalize(exit_code=exit_code)
    callback.finalize(exit_code=exit_code)

    assert calls == [exit_code]
    assert callback.finalized
