import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaac_lab_data_engine.vla.rpc import VLARequest, VLABatchRequest, encode_arrays, decode_arrays
from isaac_lab_data_engine.vla.hybrid_controller import HybridActionGuard
from isaac_lab_data_engine.vla.visual_initialization import ConsecutiveVisualWindow


def test_batch_roundtrip_and_unique_routing():
    request = VLABatchRequest(np.zeros((4, 8, 8, 3),np.uint8),np.zeros((4,30)),['lift']*4,np.array([7,2,8,1]))
    result=decode_arrays(encode_arrays(request.as_arrays()))
    np.testing.assert_array_equal(result['env_ids'], [7,2,8,1])
    with pytest.raises(ValueError,match='unique'):
        VLABatchRequest(request.rgb,request.state,request.tasks,np.array([0,0,1,2])).as_arrays()
    with pytest.raises(ValueError,match='row counts'):
        VLABatchRequest(request.rgb,request.state[:2],request.tasks,request.env_ids).as_arrays()


def test_runtime_one_forward_and_legacy_equivalence():
    path=Path(__file__).resolve().parents[1]/'scripts/serve_smolvla.py'
    spec=importlib.util.spec_from_file_location('batch_server_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    runtime=module.SmolVLARuntime.__new__(module.SmolVLARuntime)
    runtime.torch=torch; runtime.state_dim=30; runtime.action_dim=8
    runtime.joint_interface=False; runtime.relative_interface=False
    calls=[]
    def forward(batch):
        calls.append(batch)
        return batch['observation.state'][:,None,:8].repeat(1,4,1)
    runtime.policy=SimpleNamespace(config=SimpleNamespace(n_action_steps=1),predict_action_chunk=forward)
    runtime.preprocess=lambda x:x; runtime.postprocess=lambda x:x
    state=np.arange(120,dtype=np.float32).reshape(4,30)
    rgb=np.zeros((4,8,8,3),np.uint8)
    actions=runtime.predict_batch(VLABatchRequest(rgb,state,['lift']*4,np.arange(4)))
    assert len(calls)==1 and actions.shape==(4,1,8)
    np.testing.assert_array_equal(actions[:,0],state[:,:8])
    single=runtime.predict(VLARequest(rgb[2],state[2],'lift'))
    np.testing.assert_array_equal(single,actions[2])


def test_slot_guard_and_window_isolation():
    guards=[HybridActionGuard() for _ in range(2)]
    ee=np.array([.5,0,.03,1,0,0,0],np.float32)
    action=np.r_[ee,-1].astype(np.float32)
    guards[0].filter(action,np.r_[ee,1],current_eef_pose=ee,visual_object_position=ee[:3],perception_valid=True)
    assert guards[0].grasp_latched and not guards[1].grasp_latched
    windows=[ConsecutiveVisualWindow(2) for _ in range(2)]
    for frame in (1,2): windows[0].observe(frame,ee[:3],quality_valid=True,pose_valid=True)
    assert windows[0].consecutive==2 and windows[1].consecutive==0
    guards[0].reset()
    assert not guards[0].grasp_latched
