# Ray Launcher

## Introduction

`ray-launcher` is an out-of-the-box python library that wraps the frequently used `ray`  practices enabling migrating from local classes and starting ray cluster with minimum amount of code.

## Updates 

- v1.2.0: add support for: fractional GPU, colocate,  async call; add module type define, register options, call/collect options
- v1.1.2: add support for actor init kwargs, fix 0 GPU case, refactored `BaseBackend` as `BaseLocalModule`
- v1.1.1: add option of not setting cuda devices when creating backend actors
- v1.1.0: `RemoteModule` provides the wrap for fast converting local class to ray remote class
- v1.0.1: fixed problem of exiting with 1 node 
- v1.0.0: `ClusterLauncher` that wraps dirty scripts and spin waits on multi nodes

## Features


### `ClusterLauncher`

This ray cluster launcher wraps the following steps internally:

- run `ray start` commands on head and worker noodes
- run `ray.init` on all nodes
- head node spin wait for all nodes to start
- cluster start after all nodes joined
- head node returns context to main code while worker nodes spin waits for cluster to be torn down
- worker node run `ray.shutdown` and `ray stop` command after cluster starting to be torn down
- head exits after all worker nodes exited successfully

### `RemoteModule`


The `RemoteModule` class provides a high-level abstraction for creating and managing distributed modules (actors) using Ray (wrapped `ray.remote`). It handles resource allocation, actor placement, environment setup, and function registration, allowing seamless execution across CPU/GPU resources with different parallelism strategies.

#### Creation Steps of RemoteModule

**(Step 1) Create Remote Actors**: Instantiate backend actor(s) of the specified `backend_actor_class` (must inherit from `BaseLocalModule`) on target resources.

**(Step 2) Resource Allocation**:  
- **For CPU Modules**: Create 1 actor with multiple CPUs (uses all allocated CPU resources in target placement group bundle).  
- **For Continuous GPU Modules**: Create 1 actor with multiple GPUs (uses all allocated GPU resources in target placement group bundle).  
- **For Discrete GPU Modules**:  
  - *Exclusive*: Create N actors (N = allocated GPU count), each exclusively using 1 full GPU.  
  - *Colocated*: Create N actors (N = allocated GPU count), each reserving fractional GPU resources (<1.0 GPU via resource_reservation_ratio) to enable GPU sharing between actors.  

**(Step 3) Environment Configuration**:
- Propagate specified environment variables (`export_env_var_names`) to all actors
- Automatically configure distributed parameters:
  - Query first actor for IP/port to establish `module_master_addr` and `module_master_port`
  - Dispatch these parameters across all actors via `set_distributed_environs()`

**(Step 4) Function Registration**:
- Auto-discover public methods from `backend_actor_class`
- Create dual interfaces for remote execution:
  - **Sync version**: `module.method()` (blocks until all actors complete)
  - **Async version**: `module.method_async()` (returns futures immediately)
- Apply calling/collecting policies to method returns

#### Enums and Options

**(1) `RemoteModuleType`**  
Controls physical resource allocation strategy:
- **CPUModule**: CPU-only actor (no GPU resources allocated)
- **ContinuousGPUModule**: Single actor using multiple GPUs (monolithic allocation)
- **ExclusiveDiscreteGPUModule**: Multiple actors, each with exclusive access to 1 GPU (1:1 actor-GPU mapping)
- **ColocateDiscreteGPUModule**: Multiple actors sharing fractional GPUs (N:1 GPU mapping via resource reservations)

**(2) `ModuleToActorCallingPolicy`**  
Controls routing of method calls:
- **CallAllBackendActors**: Broadcast call to all actors (parallel execution)  
  *Use case: Distributed batch processing*
- **CallFirstBackendActor**: Only execute on first actor  
  *Use case: Singleton pattern/coordinator pattern*

**(3) `ActorToModuleCollectingPolicy`**  
Controls result aggregation:
- **CollectAllReturnsAsList**: Returns `List[results]` from all actors  
  *Note: Order matches actor creation order*
- **CollectFirstReturnAsItem**: Returns single result from first actor  

**(4) Function Registration Flags**  
Controls method exposure patterns:
- `register_aync_call` (bool):  
  When True (default), creates async versions of all remote methods with `_async` suffix.  
  *Async benefits*:  
  - Enables parallel execution across actors  
  - Returns futures immediately for lazy aggregation  
  - Use `ray.get()` to resolve results when needed
- `skip_private_func` (bool):  
  When True (default), skips methods starting with "_"  
  *Example*: `_internal_helper()` won't be exposed remotely

**(5) Other Init Parameters**  
Essential configuration knobs:
- `is_discrete_gpu_module`:  
  `True`=GPU-per-actor mode, `False`=shared-GPU mode
- `resource_reservation_ratio`:  
  GPU reservation per actor (1.0=full GPU, 0.5=half GPU)
- `export_env_var_names`:  
  Environment variables to propagate to actors (e.g., `["OMP_NUM_THREADS"]`)
- `do_not_set_cuda_visible_devices`:  
  Disable automatic GPU visibility management (default=False)

## Quick Start

step1: install
```bash
pip install ray-launcher
# if encounter problem starting dashboard, need to install ray by:
#   `pip install -U "ray[defaut]"`
```


step2: change local class
```python
class YourLocalModuleClass(BaseLocalModule):
    def some_method(self):
        # ...
```

step3: start cluster and use remote module
```python
from ray_launcher import ClusterLauncher

with ClusterLauncher(
    cluster_nodes_count=int(os.environ["NNODES"]),
    head_node_addr=os.environ["MASTER_ADDR"],
) as launcher:

    bundle = [{"GPU": 2, "CPU": 32}, {"GPU": 2, "CPU": 32}]
    pg = ray.util.placement_group(bundle, strategy="PACK")
    module1 = RemoteModule(YourLocalModuleClass, [(pg, 0)], is_discrete_gpu_module=True)
    module2 = RemoteModule(YourLocalModuleClass, [(pg, 1)], is_discrete_gpu_module=False)

    print(module1.some_method()) # this will get a list of results of calling each backend actor
    print(module2.some_method()) # this will get one single result, since there is only one backend actor

    # write other code for head node to execute

```

For example, see: `tests/fast_test_cpu.py` and `tests/fast_test_gpu.py`

