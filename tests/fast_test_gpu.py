import os
import time
from ray_launcher import ClusterLauncher, BaseLocalModule, RemoteModule
import ray


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
        return self.backend_name + ": " + os.environ.get("CUDA_VISIBLE_DEVICES")

with ClusterLauncher(
    cluster_nodes_count=int(os.environ["NNODES"]),
    head_node_addr=os.environ["MASTER_ADDR"],
    export_env_var_names=["RAY_DISABLE_DOCKER_CPU_WARNING"]
) as launcher:
    print("cluster ready")
    assert launcher.is_head_node, f"only head node reaches here"

    bundle = [{"GPU": 2, "CPU": 8}]
    # bundle = [{"GPU": 2, "CPU": 8}, {"GPU": 2, "CPU": 8}]
    pg = ray.util.placement_group(bundle, strategy="PACK")
    print(f"created pg")
    module1 = RemoteModule(
        MockBackend, [(pg, 0)], 
        is_discrete_gpu_module=True,
        resource_reservation_ratio=0.25,
        call_policy="FIRST", collect_policy="FIRST",
        backend_actor_kwargs={"a": 1, "b": 11}
    )
    module2 = RemoteModule(
        MockBackend, [(pg, 0)], 
        is_discrete_gpu_module=True,
        resource_reservation_ratio=0.5,
        backend_actor_kwargs={"a": 2, "b": 22}
    )
    # module3 = RemoteModule(
    #     MockBackend, [(pg, 1)], 
    #     is_discrete_gpu_module=False,
    #     resource_reservation_ratio=0.5,
    #     backend_actor_kwargs={"a": 3, "b": 33}
    # )
    print("created modules")

    print(module1.get_devices())
    print(module2.get_devices())

    print(module1.get_sum())
    fut = module2.get_sum_async()
    print(ray.get(fut))

    time.sleep(5)

    print("prepare to stop the cluster")