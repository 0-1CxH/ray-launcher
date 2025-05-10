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
    
    def _private_func(self):
        return 1
    
    def async_in_func_name(self):
        return 1

with ClusterLauncher(
    cluster_nodes_count=int(os.environ["NNODES"]),
    head_node_addr=os.environ["MASTER_ADDR"],
    export_env_var_names=["RAY_DISABLE_DOCKER_CPU_WARNING"]
) as launcher:
    print("cluster ready")
    assert launcher.is_head_node, f"only head node reaches here"

    bundle = [{"CPU": 2}, {"CPU": 2}]
    pg = ray.util.placement_group(bundle, strategy="PACK")
    print(f"created pg")
    module1 = RemoteModule(
        MockBackend, [(pg, 0)], 
        backend_actor_kwargs={"a": 1, "b": 11}, 
        collect_policy="FIRST", register_async_call=False
    )
    module2 = RemoteModule(
        MockBackend, [(pg, 1)],
        backend_actor_kwargs={"a": 2, "b": 22}, 
        skip_private_func=False
    )
    print("created modules")
    

    print(module1.get_sum())
    fut = module2.get_sum_async()
    print(fut)
    print(ray.get(fut))
    print(module2._private_func())
    

    time.sleep(5)

    print("prepare to stop the cluster")