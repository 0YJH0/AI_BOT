import numpy as np
import pytest

from isaac_lab_data_engine.vla.batch_execution import BatchChunkExecutor


@pytest.mark.parametrize('horizon', [1, 2, 4])
def test_chunk_order_hold_and_live_delta_anchor(horizon):
    schedule = BatchChunkExecutor(4, horizon, 5)
    chunk = np.zeros((4, 4, 8), np.float32)
    chunk[:, :, 0] = np.arange(1, 5) * .01
    chunk[:, :, 3] = 1
    chunk[:, :, 7] = [1, 1, -1, -1]
    requests, starts = [], []
    last = None
    for step in range(40):
        pose = np.zeros((4, 7), np.float32)
        pose[:, 0] = np.arange(4) + step * .001
        pose[:, 3] = 1
        if schedule.needs_prediction:
            requests.append(step)
            schedule.load(chunk)
        boundary = schedule.starts_action
        result = schedule.advance(pose)
        if boundary:
            starts.append(step)
            index = (step // 5) % horizon
            np.testing.assert_allclose(result[:, 0], pose[:, 0] + chunk[:, index, 0])
            np.testing.assert_array_equal(result[:, 7], chunk[:, index, 7])
        else:
            np.testing.assert_array_equal(result, last)
        last = result
    assert requests == list(range(0, 40, 5*horizon))
    assert starts == list(range(0, 40, 5))


def test_invalid_chunk_and_reset_isolation():
    schedule = BatchChunkExecutor(4, 4)
    with pytest.raises(ValueError, match='short'):
        schedule.load(np.zeros((4, 2, 8)))
    with pytest.raises(ValueError, match='Prediction required'):
        schedule.advance(np.zeros((4, 7)))
    schedule.load(np.zeros((4, 4, 8)))
    with pytest.raises(ValueError, match='unexecuted'):
        schedule.load(np.zeros((4, 4, 8)))
    assert BatchChunkExecutor(4, 4).needs_prediction
