import numpy as np
import pytest
from isaac_lab_data_engine.vla.joint_interface import JointActionGuard, contract, validate_joint_trajectory
from isaac_lab_data_engine.vla.rpc import VLARequest
from isaac_lab_data_engine.vla.joint_interface import JointCommandBridge
from isaac_lab_data_engine.vla.joint_interface import validate_execution_settings


def test_two_action_execution_is_deployment_only_and_preserves_checkpoint_contract():
    validate_execution_settings(1, 5)
    validate_execution_settings(2, 5)
    validate_execution_settings(4, 5)
    assert contract()['execution_horizon'] == 1
    for horizon, repeat, teacher in [(2,5,True), (4,5,True), (3,5,False), (2,10,False), (0,5,False)]:
        with pytest.raises(ValueError):
            validate_execution_settings(horizon, repeat, teacher_or_replay=teacher)


def test_joint_safety_and_latch_without_target_coordinates():
    guard=JointActionGuard(np.tile([-2.,2.],(7,1)))
    a=np.r_[np.ones(7),-1].astype(np.float32)
    result=guard.filter(a,np.zeros(7))
    np.testing.assert_allclose(result.action[:7],.2)
    assert result.position_was_clipped and result.action[-1]==-1
    result=guard.filter(np.r_[np.zeros(7),1],np.zeros(7))
    assert result.action[-1]==-1
    assert guard.filter(np.r_[np.full(7,np.nan),1],np.zeros(7)).source!='vla'


def test_joint_semantic_tag_cannot_silently_use_cartesian_rpc():
    request=VLARequest(np.zeros((4,4,3),np.uint8),np.zeros(25),'lift',input_interface='joint8')
    assert request.as_arrays()['input_interface'].item()=='joint8'
    with pytest.raises(ValueError):
        VLARequest(request.rgb,np.zeros(30),'lift',input_interface='joint8').as_arrays()
    assert contract()['execution']=='direct_PD_joint_target_no_student_IK'


def synthetic():
    n=15
    steps=np.arange(1,5*n,5)
    phase=np.repeat(np.arange(5),3)
    state=np.zeros((n,30),np.float32);state[:,29]=phase
    action=np.zeros((n,8),np.float32);action[:,7]=1
    physics=np.zeros((5*n+1,8),np.float32);physics[:,7]=1
    a=dict(state=state,action=action,controller_phase=phase,control_step=steps,
        physics_step=np.arange(1,len(physics)+1),physics_joint_action=physics,
        physics_q=np.zeros((len(physics),9),np.float32),
        rgb=np.tile(np.array([0,255,0],np.uint8),(n,4,4,1)),depth=np.ones((n,4,4),np.float32))
    m=dict(format=contract()['version'],success=True,maximum_lift_cm=11,sustained_success_steps=10,
           model_requests=0,scripted_fallback_actions=0)
    return a,m


def test_only_actual_causal_executed_labels_accepted():
    a,m=synthetic()
    assert validate_joint_trajectory(a,m)==15
    a['physics_joint_action'][2,0]=.1
    with pytest.raises(ValueError,match='actually held'): validate_joint_trajectory(a,m)


def test_current_q_copy_is_not_confused_with_original_cartesian_data():
    a,m=synthetic();m['format']='vla_execution_aligned_full_v1'
    with pytest.raises(ValueError,match='newly collected'): validate_joint_trajectory(a,m)


def test_bridge_computes_teacher_once_but_student_does_not_use_ik():
    import torch
    from types import SimpleNamespace
    class Robot:
        def __init__(self):
            self.data=SimpleNamespace(joint_pos=torch.zeros(1,9),joint_pos_target=torch.zeros(1,9),
                soft_joint_pos_limits=torch.tensor([[[-2.,2.]]*9]))
        def set_joint_position_target(self, target, joint_ids):
            self.data.joint_pos_target[:,joint_ids]=target
    robot=Robot()
    class Arm:
        _joint_ids=list(range(7)); _joint_names=[f'panda_joint{i}' for i in range(1,8)]
        calls=0
        def process_actions(self, action): self.last=action
        def apply_actions(self):
            self.calls+=1
            robot.set_joint_position_target(robot.data.joint_pos[:,:7]+.08,self._joint_ids)
    arm=Arm()
    env=SimpleNamespace(scene={'robot':robot},device='cpu',action_manager=SimpleNamespace(get_term=lambda _:arm))
    bridge=JointCommandBridge(env)
    teacher=bridge.teacher_target(np.array([.5,0,.2,1,0,0,0,-1],np.float32))
    assert arm.calls==1
    np.testing.assert_allclose(teacher.action[:7],.08)
    for _ in range(10):
        bridge.apply()
    assert arm.calls==1
    np.testing.assert_allclose(robot.data.joint_pos_target[:,:7].numpy(),.08)
    bridge.reset()
    student=bridge.set_target(np.r_[np.full(7,.1),1])
    bridge.apply()
    assert arm.calls==1 and student.action[-1]==1
