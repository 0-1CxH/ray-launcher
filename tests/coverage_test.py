import os
import time
from ray_launcher import ClusterLauncher, BaseLocalModule, RemoteModule, ModuleToActorCallingPolicy, ActorToModuleCollectingPolicy, RemoteModuleType
import ray


import sys

TEST_CASE = int(sys.argv[1])


os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

class MockBackend(BaseLocalModule):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def get_sum(self):
        return self.a + self.b

    def get_devices(self):
        return f"{self.backend_name}: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}"

    def get_env(self, var_name):
        return os.environ.get(var_name)

class MockBackendWithPrivate(BaseLocalModule):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def get_sum(self):
        return self.a + self.b

    def _private_method(self):
        return "private"

with ClusterLauncher(
    cluster_nodes_count=int(os.environ["NNODES"]),
    head_node_addr=os.environ["MASTER_ADDR"],
    export_env_var_names=["RAY_DISABLE_DOCKER_CPU_WARNING"]
) as launcher:
    print("Cluster ready")
    assert launcher.is_head_node, "Only head node executes this code"

    # Create multiple placement groups for different test cases
    bundle_cpu = [{"CPU": 4}]
    pg_cpu = ray.util.placement_group(bundle_cpu, strategy="PACK")
    
    bundle_continuous_gpu = [{"GPU": 2, "CPU": 8}]
    pg_continuous = ray.util.placement_group(bundle_continuous_gpu, strategy="PACK")
    
    bundle_exclusive_gpu = [{"GPU": 2, "CPU": 8}]
    pg_exclusive = ray.util.placement_group(bundle_exclusive_gpu, strategy="PACK")
    
    bundle_colocate_gpu = [{"GPU": 1, "CPU": 8}]
    pg_colocate = ray.util.placement_group(bundle_colocate_gpu, strategy="PACK")

    print("Waiting for placement groups to be ready...")
    ray.get([pg_cpu.ready(), pg_continuous.ready(), pg_exclusive.ready(), pg_colocate.ready()])
    print("Placement groups ready")


    # Test Case 1: CPU Module
    module3 = RemoteModule(
        MockBackend, [(pg_cpu, 0)], module_name="CPUModule",
        is_discrete_gpu_module=False, backend_actor_kwargs={"a": 3, "b": 33}
    )
    assert module3.get_remote_module_type() == RemoteModuleType.CPUModule
    assert len(module3.backend_actors) == 1
    print(module3.get_devices())  # Expected "MockBackend: "

    # Test Case 2: Continuous GPU Module (2 GPUs)
    module4 = RemoteModule(
        MockBackend, [(pg_continuous, 0)], module_name="ContinuousGPUModule2",
        is_discrete_gpu_module=False, backend_actor_kwargs={"a": 4, "b": 44}
    )
    assert module4.get_remote_module_type() == RemoteModuleType.ContinuousGPUModule
    assert len(module4.backend_actors) == 1
    print(module4.get_devices())  # Expected "MockBackend: 0,1"

    # Test Case 3: Exclusive Discrete GPU Module (2 GPUs, 2 actors)
    module5 = RemoteModule(
        MockBackend, [(pg_exclusive, 0)], module_name="ExclusiveDiscreteModule2",
        is_discrete_gpu_module=True, resource_reservation_ratio=1.0,
        backend_actor_kwargs={"a": 5, "b": 55}
    )
    assert module5.get_remote_module_type() == RemoteModuleType.ExclusiveDiscreteGPUModule
    assert len(module5.backend_actors) == 2
    print(module5.get_devices())  # Expect list ["MockBackend: 0", "MockBackend: 1"]

    # Test Case 4: Colocate Discrete GPU Module (1 GPU, 0.5 ratio)
    module6 = RemoteModule(
        MockBackend, [(pg_colocate, 0)], module_name="ColocateDiscreteModule",
        is_discrete_gpu_module=True, resource_reservation_ratio=0.5,
        backend_actor_kwargs={"a": 6, "b": 66}
    )
    assert module6.get_remote_module_type() == RemoteModuleType.ColocateDiscreteGPUModule
    assert len(module6.backend_actors) == 1  # Current code creates 1 actor due to bundle GPU count
    print(module6.get_devices())  # Expect ["MockBackend: 0"]

    # Test Case 5: Call and Collect Policies
    module7 = RemoteModule(
        MockBackend, [(pg_colocate, 0)], module_name="PolicyModule",
        is_discrete_gpu_module=True, resource_reservation_ratio=0.5,
        call_policy=ModuleToActorCallingPolicy.CallFirstBackendActor.value,
        collect_policy=ActorToModuleCollectingPolicy.CollectFirstReturnAsItem.value,
        backend_actor_kwargs={"a": 7, "b": 77}
    )
    sum_result = module7.get_sum()
    assert sum_result == 84  # 7+77
    assert isinstance(sum_result, int)

    # Test Case 6: Async Calls
    async_sum = module7.get_sum_async()
    sum_result_async = ray.get(async_sum)
    assert sum_result_async == 84

    # Test Case 7: Private Function Skipping
    module8 = RemoteModule(
        MockBackendWithPrivate, [(pg_cpu, 0)], module_name="SkipPrivateModule",
        is_discrete_gpu_module=False, skip_private_func=True,
        backend_actor_kwargs={"a": 8, "b": 88}
    )
    assert not hasattr(module8, '_private_method'), "Private method should be skipped"

    module9 = RemoteModule(
        MockBackendWithPrivate, [(pg_cpu, 0)], module_name="IncludePrivateModule",
        is_discrete_gpu_module=False, skip_private_func=False,
        backend_actor_kwargs={"a": 9, "b": 99}
    )
    assert hasattr(module9, '_private_method'), "Private method should be registered"
    assert ray.get(module9._private_method.remote()) == "private"

    # Test Case 8: Environment Variables
    module10 = RemoteModule(
        MockBackend, [(pg_cpu, 0)], module_name="EnvVarModule",
        is_discrete_gpu_module=False, export_env_var_names=["RAY_DISABLE_DOCKER_CPU_WARNING"],
        backend_actor_kwargs={"a": 10, "b": 100}
    )
    env_value = ray.get(module10.get_env.remote("RAY_DISABLE_DOCKER_CPU_WARNING"))
    assert env_value == "1", "Environment variable not set correctly"

    print("All tests passed!")
    time.sleep(5)
    print("Stopping cluster...")