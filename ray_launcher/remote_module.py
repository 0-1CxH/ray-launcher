import os
import ray
import inspect
from enum import Enum
from collections import namedtuple
from typing import Optional, List
from functools import partial
from loguru import logger
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .base_local_module import BaseLocalModule

PlacementGroupAndIndex = namedtuple("PlacementGroupAndIndex", ["placement_group", "bundle_index"])

class RemoteModuleType(Enum):
    CPUModule = "CPU_ONLY_MODULE"
    ContinuousGPUModule = "CONTINOUS_GPU_MODULE" # one module - one backend actor 
    ExclusiveDiscreteGPUModule = "EXCLUSIVE_DISCRETE_GPU_MODULE" # one module - multiple actors, each actor use 1 exclusive gpu
    ColocateDiscreteGPUModule = "COLOCATE_DISCRETE_GPU_MODULE" #  one module - multiple actors, each actor shares 1 gpu with other actor

class ModuleToActorCallingPolicy(Enum):
    CallAllBackendActors = "ALL"
    CallFirstBackendActor = "FIRST"

class ActorToModuleCollectingPolicy(Enum):
    CollectAllReturnsAsList = "ALL"
    CollectFirstReturnAsItem = "FIRST"


class RemoteModule:
    def __init__(
            self, 
            backend_actor_class,
            placement_groups_and_indices: List[PlacementGroupAndIndex],
            module_name: Optional[str] = None,
            # discrete resource configs
            is_discrete_gpu_module: bool = False, # must be gpu module first, cpu module is not discrete
            resource_reservation_ratio: float = 1.0, # 1.0 means exclusive, <1.0 means to share with other module(s)
            # module-actor policy
            call_policy: str = ModuleToActorCallingPolicy.CallAllBackendActors.value, 
            collect_policy: str = ActorToModuleCollectingPolicy.CollectAllReturnsAsList.value,
            # actor args and envs
            backend_actor_kwargs: Optional[dict] = None, # the init args that pass to the backend actor class
            export_env_var_names: Optional[List[str]] = None,
            do_not_set_cuda_visible_devices: bool = False,
            # actor func register
            skip_private_func: bool = True,
            register_async_call: bool = True,
    ):
        self.backend_actor_class = backend_actor_class
        assert issubclass(self.backend_actor_class, BaseLocalModule)
        if module_name is None:
            module_name = backend_actor_class.__name__ + str(id(self))
        self.module_name = module_name

        self.remote_module_type = None
        
        self.backend_actors = []
        self.backend_actor_reserved_resources = []
        if export_env_var_names is None:
            export_env_var_names = []
        if backend_actor_kwargs is None:
            backend_actor_kwargs = {}
        
        assert 0.0 <= resource_reservation_ratio <= 1.0, "cannot reserve more than 100% or less than 0% of resource"
        if is_discrete_gpu_module is False and resource_reservation_ratio < 1.0:
            resource_reservation_ratio = 1.0
            logger.warning(f"setting resource_reservation_ratio < 1.0 is effective only when is_discrete_gpu_module is True")

        assert call_policy in ModuleToActorCallingPolicy, f"{call_policy} is not a valid module-to-actor call policy"
        assert collect_policy in ActorToModuleCollectingPolicy, f"{collect_policy} is not a valid actor-to-module collect policy"
        self.call_policy = call_policy
        self.collect_policy = collect_policy

        self._create_backend_actors(
            placement_groups_and_indices,
            is_discrete_gpu_module,
            resource_reservation_ratio, 
            backend_actor_kwargs,
            export_env_var_names,
            do_not_set_cuda_visible_devices,    
        )

        self.remote_funcs = []
        self._register_remote_funcs(skip_private_func, register_async_call)
        logger.info("Remote Module Created. Info:\n" + self.format_module_info())
    
    def get_remote_module_type(self):
        return self.remote_module_type
    
    def get_registered_remote_funcs(self):
        return self.remote_funcs
    
    def format_module_info(self):
        return (
            f"Moudle Name: {self.module_name}\nModule Type: {self.remote_module_type}\n"
            f"Backend Actor Count: {len(self.backend_actors)}\nBackend Actor Reserved Resources: {self.backend_actor_reserved_resources}\n"
            f"Call Policy: {self.call_policy}, Collect Policy: {self.collect_policy}\n"
            f"Registered Funcs ({len(self.remote_funcs)}): {self.remote_funcs}\n"
        )
    
    def _create_backend_actors(
            self,
            placement_groups_and_indices: List[PlacementGroupAndIndex], 
            is_discrete_gpu_module: bool,
            resource_reservation_ratio: float,
            backend_actor_kwargs: dict,
            export_env_var_names: List[str],
            do_not_set_cuda_visible_devices: bool,
        ):
        env_vars = {}
        for name in export_env_var_names:
            if name in os.environ:
                env_vars.update(
                    {name: os.environ.get(name)}
                )
            else:
                logger.warning(f"{name} does not exist in environ")
        if do_not_set_cuda_visible_devices is True:
             env_vars.update({"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"})

        if is_discrete_gpu_module is True:
            for pg, idx in placement_groups_and_indices:
                current_bundle_gpu_count = int(pg.bundle_specs[idx].get("GPU"))
                assert current_bundle_gpu_count > 0, f"discrete gpu actor must be created on group with gpu resource"
                current_bundle_cpu_count_per_gpu = float(pg.bundle_specs[idx].get("CPU"))/current_bundle_gpu_count
                if resource_reservation_ratio < 1.0:
                    self.remote_module_type = RemoteModuleType.ColocateDiscreteGPUModule
                else:
                    self.remote_module_type = RemoteModuleType.ExclusiveDiscreteGPUModule
                for _ in range(current_bundle_gpu_count):
                    remote_actor = ray.remote(
                            num_gpus=1 * resource_reservation_ratio,
                            num_cpus=current_bundle_cpu_count_per_gpu * resource_reservation_ratio
                        )(self.backend_actor_class).options(
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg,
                                placement_group_bundle_index=idx,
                        ) , runtime_env={"env_vars": env_vars}
                        ).remote(**backend_actor_kwargs)
                    self.backend_actors.append(remote_actor)
                    self.backend_actor_reserved_resources.append({"GPU": 1 * resource_reservation_ratio, "CPU": current_bundle_cpu_count_per_gpu * resource_reservation_ratio})
                    logger.debug(f"created backend actor ({_+1}/{current_bundle_gpu_count}) of {self.remote_module_type.value} module {self.module_name} (args: {backend_actor_kwargs})" 
                                 f"on {pg.id} idx={idx} with {1 * resource_reservation_ratio} gpu, {current_bundle_cpu_count_per_gpu * resource_reservation_ratio} cpu and environ {env_vars}")

            assert len(self.backend_actors) > 0
            rank_0_actor = self.backend_actors[0]
            module_master_addr = ray.get(rank_0_actor.get_ip_address.remote())
            module_master_port = ray.get(rank_0_actor.get_avaiable_port.remote())
            logger.debug(f"rank 0 backend gives {module_master_addr=}, {module_master_port=}")

            set_environs_futures = []
            for actor_idx, actor in enumerate(self.backend_actors):
                set_environs_futures.append(actor.set_distributed_environs.remote(
                    actor_idx,
                    len(self.backend_actors),
                    module_master_addr,
                    module_master_port
                ))
            ray.get(set_environs_futures)


        else:
            assert len(placement_groups_and_indices) == 1, f"the actor is continuous, should not spread to groups"
            pg, idx = placement_groups_and_indices.pop()
            current_bundle_gpu_count = int(pg.bundle_specs[idx].get("GPU", 0))
            if current_bundle_gpu_count > 0:
                self.remote_module_type = RemoteModuleType.ContinuousGPUModule
            else:
                self.remote_module_type = RemoteModuleType.CPUModule
            current_bundle_cpu_count = int(pg.bundle_specs[idx].get("CPU", 0))
            self.backend_actors.append(
                ray.remote(
                    num_gpus=current_bundle_gpu_count,
                    num_cpus=current_bundle_cpu_count
                )(self.backend_actor_class).options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=idx,
                ), runtime_env={"env_vars": env_vars}
                ).remote(**backend_actor_kwargs)
            )
            self.backend_actor_reserved_resources.append({"GPU": current_bundle_gpu_count, "CPU": current_bundle_cpu_count})
            logger.debug(f"created single backend actor of {self.remote_module_type.value} module {self.module_name} (args={backend_actor_kwargs}) on "
                         f"{pg.id} idx={idx} with {current_bundle_gpu_count} gpu, {current_bundle_cpu_count} cpu and environ {env_vars}")

    

    def _call_func_of_all_remote_actors(self, func_name: str, is_sync_call: bool, *args, **kwargs):
        all_func_return_futures = []
        if self.call_policy == ModuleToActorCallingPolicy.CallAllBackendActors.value:
            for actor in self.backend_actors:
                assert hasattr(actor, func_name)
                all_func_return_futures.append(getattr(actor, func_name).remote(*args, **kwargs))
        elif self.call_policy == ModuleToActorCallingPolicy.CallFirstBackendActor.value:
            actor = self.backend_actors[0]
            assert hasattr(actor, func_name)
            all_func_return_futures.append(getattr(actor, func_name).remote(*args, **kwargs))
        else:
            raise ValueError(f"invalid policy of choice: selecte call policy is {self.call_policy}")
        
        if is_sync_call:
            all_func_returns = ray.get(all_func_return_futures)
        else:
            logger.debug(f"using async call, the result will not be obtained until ray.get")
        
        if self.collect_policy == ActorToModuleCollectingPolicy.CollectAllReturnsAsList.value:
            if is_sync_call:
                return all_func_returns
            else:
                return all_func_return_futures
        elif self.collect_policy == ActorToModuleCollectingPolicy.CollectFirstReturnAsItem.value:
            if is_sync_call:
                return all_func_returns[0]
            else:
                return all_func_return_futures[0]
        else:
            raise ValueError(f"invalid policy of choice: selecte collect policy is {self.collect_policy}")
    
    
    def _register_remote_funcs(self, skip_private_func: bool, register_async_call: bool):
        for name, member in inspect.getmembers(self.backend_actor_class, predicate=inspect.isfunction):
            if not name.startswith("__"): # auto register all non-magic methods
                if name.startswith("_"):
                    if skip_private_func:
                        logger.debug(f"auto detected possible private func: {name}, skip")
                        continue
                self.remote_funcs.append(name)
                setattr(self, name, partial(self._call_func_of_all_remote_actors, name, True))
                logger.debug(f"auto detected and registered remote func (sync call): {name}({member})")
                if register_async_call:
                    async_name = name + "_async"
                    if "async" in name:
                        logger.warning(f"function {name} already contains substring 'async', the mapped func will be {async_name}")
                    self.remote_funcs.append(async_name)
                    setattr(self, async_name, partial(self._call_func_of_all_remote_actors, name, False))
                    logger.debug(f"auto detected and registered remote func (async call): {async_name}({member})")

