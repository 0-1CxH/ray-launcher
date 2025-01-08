import ray
import inspect

from collections import namedtuple
from functools import partial
from loguru import logger
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .base_backend import BaseBackend

PlacementGroupAndIndex = namedtuple("PlacementGroupAndIndex", ["placement_group", "bundle_index"])

class RemoteModule:
    def __init__(
            self, 
            backend_clz,
            placement_groups_and_indices: list[PlacementGroupAndIndex],
            discrete_gpu_actors: bool, # must be gpu actor first, cpu actor is not discrete
            module_name: str = None
    ):
        self.backend_clz = backend_clz
        assert issubclass(self.backend_clz, BaseBackend)
        if module_name is None:
            module_name = backend_clz.__name__ + str(id(self))
        self.module_name = module_name

        self.discrete_gpu_actors = discrete_gpu_actors
        
        self.backend_actors = []
        self._create_backend_actors(placement_groups_and_indices)

        self._register_remote_funcs()

    
    def _create_backend_actors(self, placement_groups_and_indices: list[PlacementGroupAndIndex]):
        if self.discrete_gpu_actors is True:
            for pg, idx in placement_groups_and_indices:
                current_bundle_gpu_count = int(pg.bundle_specs[idx].get("GPU"))
                assert current_bundle_gpu_count > 0, f"discrete gpu actor must be created on group with gpu resource"
                current_bundle_cpu_count_per_gpu = float(pg.bundle_specs[idx].get("CPU"))/current_bundle_gpu_count
                for _ in range(current_bundle_gpu_count):
                    remote_actor = ray.remote(
                            num_gpus=1,
                            num_cpus=current_bundle_cpu_count_per_gpu
                        )(self.backend_clz).options(
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg,
                                placement_group_bundle_index=idx,
                        )# , runtime_env={"env_vars": {"RAY_DEDUP_LOGS": "0"}}
                        ).remote()
                    self.backend_actors.append(remote_actor)
                    logger.debug(f"created remote actor {len(self.backend_actors) - 1} of module {self.module_name} on {pg.id} idx={idx} with 1 gpu and {current_bundle_cpu_count_per_gpu} cpu")
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
            current_bundle_gpu_count = int(pg.bundle_specs[idx].get("GPU"))
            current_bundle_cpu_count = int(pg.bundle_specs[idx].get("CPU"))
            self.backend_actors.append(
                ray.remote(
                    num_gpus=current_bundle_gpu_count,
                    num_cpus=current_bundle_cpu_count
                )(self.backend_clz).options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=idx,
                ) # , runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",}}
                ).remote()
            )
            logger.debug(f"created single remote actor of module {self.module_name} on {pg.id} idx={idx} with {current_bundle_gpu_count} gpu and {current_bundle_cpu_count} cpu")
    

    def _call_func_of_all_remote_actors(self, func_name: str, *args, **kwargs):
        all_func_returns = []
        for actor in self.backend_actors:
            assert hasattr(actor, func_name)
            all_func_returns.append(getattr(actor, func_name).remote(*args, **kwargs))
        if len(all_func_returns) == 1:
            all_func_returns = all_func_returns[0]
        else:
            logger.debug(f"module {self.module_name} contains multiple actors, will return a list of all results")
        return ray.get(all_func_returns)
    
    
    def _register_remote_funcs(self):
        self.remote_funcs = []
        for name, member in inspect.getmembers(self.backend_clz, predicate=inspect.isfunction):
            if not name.startswith("__"): # auto register all non-magic methods
                self.remote_funcs.append(name)
                setattr(self, name, partial(self._call_func_of_all_remote_actors, name))
                logger.debug(f"auto detected and registered remote func: {name}({member})")

