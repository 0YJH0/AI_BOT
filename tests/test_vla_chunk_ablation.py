from copy import deepcopy
import pytest
from isaac_lab_data_engine.vla.chunk_ablation import compare_train_configs, validate_interface


def configs():
    a={'policy':{'chunk_size':4,'n_action_steps':1,
                 'input_features':{'observation.state':{'shape':[30]}},
                 'output_features':{'action':{'shape':[8]}},'optimizer_lr':1e-5},
       'seed':1000,'batch_size':1,'steps':6000,'output_dir':'a','job_name':'a'}
    b=deepcopy(a)
    b['policy']['chunk_size']=1
    b['output_dir']=b['job_name']='b'
    return a,b


def test_only_chunk_and_run_identity_may_differ():
    a,b=configs()
    compare_train_configs(a,b)
    b['seed']=1001
    with pytest.raises(ValueError,match='Training differs'):
        compare_train_configs(a,b)


def test_second_policy_factor_rejected():
    a,b=configs()
    b['policy']['optimizer_lr']=3e-6
    with pytest.raises(ValueError,match='Training differs'):
        compare_train_configs(a,b)


def test_runtime_chunk_override_and_fixed_execution_steps():
    a,b=configs()
    a['policy']['input_features']['observation.state']['shape']=(30,)
    validate_interface(a['policy'],4)
    with pytest.raises(ValueError,match='Runtime policy'):
        validate_interface(a['policy'],1)
    a['policy']['n_action_steps']=4
    with pytest.raises(ValueError,match='Runtime policy'):
        validate_interface(a['policy'],4)
