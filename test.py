import os
import time
from ray_launcher import ClusterLauncher, BaseBackend, RemoteModule
import ray

os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

class MockBackend(BaseBackend):
    def get_devices(self):
        return self.backend_name + ": " + os.environ.get("CUDA_VISIBLE_DEVICES")
        

with ClusterLauncher(
    cluster_nodes_count=int(os.environ["NNODES"]),
    head_node_addr=os.environ["MASTER_ADDR"],
    export_env_var_names=["RAY_DISABLE_DOCKER_CPU_WARNING"]
) as launcher:
    print("cluster ready")
    assert launcher.is_head_node, f"only head node reaches here"
    bundle = [{"GPU": 2, "CPU": 32}, {"GPU": 2, "CPU": 32}]
    pg = ray.util.placement_group(bundle, strategy="PACK")
    module1 = RemoteModule(MockBackend, [(pg, 0)], True)
    module2 = RemoteModule(MockBackend, [(pg, 1)], False)

    print(module1.get_devices())
    print(module2.get_devices())

    time.sleep(5)
    print("prepare to stop the cluster")