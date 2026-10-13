import numpy as np
import pytest
from isaac_lab_data_engine.vla.future_joint_labels import future_joint_labels


def test_future_arm_target_and_current_gripper_command_are_aligned():
    state=np.arange(150,dtype=np.float32).reshape(5,30)
    action=np.zeros((5,8),np.float32);action[:,7]=[1,1,-1,-1,-1]
    current,target=future_joint_labels(state,action,np.arange(1,22,5))
    np.testing.assert_array_equal(current,state[:-1])
    np.testing.assert_array_equal(target[:,:7],state[1:,:7])
    np.testing.assert_array_equal(target[:,7],action[:-1,7])
    assert len(target)==4


def test_gap_in_recording_is_rejected():
    with pytest.raises(ValueError,match='contiguous'):
        future_joint_labels(np.zeros((4,30)),np.zeros((4,8)),[1,6,16,21])


@pytest.mark.parametrize('steps', [[1,6], [[1],[6],[11],[16]], [1,6,11,np.nan]])
def test_wrong_time_shape_and_nonfinite_time_are_rejected(steps):
    with pytest.raises(ValueError,match='contiguous'):
        future_joint_labels(np.zeros((4,30)),np.zeros((4,8)),steps)
