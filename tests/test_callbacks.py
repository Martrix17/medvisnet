"""Unit test for EarlyStopping class."""

from src.training.callbacks import EarlyStopping


def test_improvement_resets_counter():
    early_stopping = EarlyStopping(patience=3, delta=0.1)
    losses = [1.0, 0.8, 0.7]

    for loss in losses:
        stopped = early_stopping(loss)
        assert not stopped
        assert early_stopping.counter == 0


def test_no_improvement_triggers_stop(capsys):
    early_stopping = EarlyStopping(patience=2, delta=0.01)
    losses = [1.0, 1.005, 1.01]

    stopped = False
    for loss in losses:
        stopped = early_stopping(loss)

    assert stopped
    assert early_stopping.early_stop
    out = capsys.readouterr().out
    assert "Early stopping triggered" in out


def test_improvement_after_plateau_resets_counter():
    early_stopping = EarlyStopping(patience=3, delta=0.01)
    early_stopping(1.0)
    early_stopping(1.0)
    assert early_stopping.counter == 1
    early_stopping(0.8)
    assert early_stopping.counter == 0
    assert not early_stopping.early_stop
