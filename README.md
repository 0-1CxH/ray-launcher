# Ray Launcher: An out-of-the-box ray cluster launcher

## Introduction

This ray cluster launcher wraps the following steps internally:

- run `ray start` commands on head and worker noodes
- run `ray.init` on all nodes
- head node spin wait for all nodes to start
- cluster start after all nodes joined
- head node returns context to main code while worker nodes spin waits for cluster to be torn down
- worker node run `ray.shutdown` and `ray stop` command after cluster starting to be torn down
- head exits after all worker nodes exited successfully


## Quick Start

step1: install
```bash
pip install ray-launcher
```

step2: use
```python
from ray_launcher import ClusterLauncher

with ClusterLauncher(
    cluster_nodes_count=int(os.environ["NNODES"]),
    head_node_addr=os.environ["MASTER_ADDR"],
) as launcher:
    # write the code for head node to execute

```