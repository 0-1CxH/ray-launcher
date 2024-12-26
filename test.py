import os
import time
from core import ClusterLauncher

os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"



with ClusterLauncher(
    cluster_nodes_count=int(os.environ["NNODES"]),
    head_node_addr=os.environ["MASTER_ADDR"],
    export_env_var_names=["RAY_DISABLE_DOCKER_CPU_WARNING"]
) as launcher:
    print("cluster ready")
    assert launcher.is_head_node, f"only head node reaches here"
    time.sleep(15)
    print("prepare to stop the cluster")