import os
import time
import ray
import traceback
from loguru import logger
from .cluster_status_manager import ClusterStatusManager
from .utils import get_self_addr, run_shell_command, ray_nodes_info_formatter

class ClusterLauncher:

    def __init__(
            self,
            cluster_nodes_count: int,
            head_node_addr: str,
            head_node_ray_port: int = 54321,
            head_node_ray_dashboard_port: int = 54322,
            node_rank: int = None,
            node_ray_manager_port: int = 54323,
            validate_head_node_addr: bool = True,
            export_env_var_names: list = None,
            head_node_spin_wait_interval: int = 3,
            worker_node_spin_wait_interval: int = 3,
    ) -> None:
        
        
        assert isinstance(cluster_nodes_count, int)
        self.cluster_nodes_count = cluster_nodes_count
        self.head_node_addr = head_node_addr
        self.head_node_ray_port = head_node_ray_port
        self.head_node_ray_dashboard_port = head_node_ray_dashboard_port

        # get node rank from arg or environ
        if node_rank is None:
            node_rank_in_env_var = os.environ.get("NODE_RANK")
            if node_rank_in_env_var is not None:
                self.node_rank = int(node_rank_in_env_var)
            else:
                raise ValueError("must specify node rank either by passing to args or exporting to environ")
        else:
            assert isinstance(node_rank, int)
            assert node_rank < cluster_nodes_count and node_rank >= 0
            self.node_rank = node_rank
        
        self.node_ray_manager_port = node_ray_manager_port

        # get node addr and optionally validate it
        self.node_addr = get_self_addr()
        if self.node_rank == 0:
            if validate_head_node_addr:
                assert self.head_node_addr == self.node_addr, (
                    "head node has rank = 0, this is rank 0 node but got"
                    f"ip {self.node_addr}, expect {self.head_node_addr}"
                )
            self.is_head_node = True
        else:
            if validate_head_node_addr:
                assert self.head_node_addr != self.node_addr, (
                    f"head node has rank = 0, this is rank {self.node_rank} node "
                    f"but got head node addr of {self.head_node_addr}"
                )
            self.is_head_node = False
        
        self.cluster_status_manager_handle = None
        self.cluster_status_manager_actor_name = "ClusterStatusManager"
        self.cluster_namespace = "ClusterLauncher"

        self.export_env_var_names = ["PYTHONPATH", "CUDA_VISIBLE_DEVICES"]
        if export_env_var_names is not None:
            self.export_env_var_names.extend(export_env_var_names)
        
        self.head_node_spin_wait_interval = head_node_spin_wait_interval
        self.worker_node_spin_wait_interval = worker_node_spin_wait_interval
            
    
    # context protocol
    def __enter__(self):
        node_started = self._start_ray_node()

        # step1: head node waiting for workers to join the cluster
        if self.is_head_node:
            logger.debug(f"setting up cluster status manager")
            self.cluster_status_manager_handle = \
                ClusterStatusManager.options(
                    name=self.cluster_status_manager_actor_name
                ).remote(self.cluster_nodes_count)
            ray.get(
                self.cluster_status_manager_handle.set_node_validity.remote(self.node_rank, node_started)
            )
        else:
            while True:
                try:
                    self.cluster_status_manager_handle = ray.get_actor(self.cluster_status_manager_actor_name)
                    break
                except Exception as e:
                    logger.debug(f"waiting for cluster status manager ({e})")
                    time.sleep(self.worker_node_spin_wait_interval)
            
            ray.get(self.cluster_status_manager_handle.set_node_validity.remote(self.node_rank, node_started))
        
        # step2: return context on head node, worker nodes spin wait cluster to tear down
        while True:
            
            if self.is_head_node:
                if ray.get(self.cluster_status_manager_handle.is_cluster_ready.remote()):
                    # once the cluster is ready, return self on head node
                    return self
                else:
                    # if is not ready, spin wait
                    time.sleep(self.head_node_spin_wait_interval)
            else:
                if ray.get(self.cluster_status_manager_handle.is_cluster_tearing_down.remote()):
                    # if it is tearing down, stop self ans exit
                    ray.get(self.cluster_status_manager_handle.set_node_validity.remote(self.node_rank, False))
                    self._stop_ray_node()
                    exit(0)
                else:
                    time.sleep(self.worker_node_spin_wait_interval)        

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            s = f"exiting context with exception, exception type: {exc_type}, exception value: {exc_value}, "
            formatted_traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            s += f"exception traceback: {formatted_traceback}"
            logger.error(s)
        else:
            logger.debug("exiting context with no exception")
        
        if self.is_head_node:
            ray.get(self.cluster_status_manager_handle.set_node_validity.remote(self.node_rank, False))
            while True:
                try:
                    if ray.get(self.cluster_status_manager_handle.is_cluster_finished.remote()):
                        break
                    else:
                        logger.debug(f"tearing down the cluster, waiting for worker nodes to exit")
                        time.sleep(self.head_node_spin_wait_interval)
                except:
                    break
            self._stop_ray_node()
            logger.info(f"cluster of {self.cluster_nodes_count} nodes stopped successfully")
        else:
            # only head node will reach here
            raise RuntimeError("worker nodes should not reach here")


    def _start_ray_node(self):
        # step1: run ray start command
        logger.debug(f"runnnig ray start command on node {self.node_rank}")
        if self.is_head_node:
            command = (
                "RAY_prestart_worker_first_driver=0 "
                "ray start --head --include-dashboard=True "
                f"--node-ip-address={self.head_node_addr} --port={self.head_node_ray_port} "
                f"--node-name='ray-head-node-rank-{self.node_rank}-{self.node_addr}' "
                f"--dashboard-port={self.head_node_ray_dashboard_port} --node-manager-port {self.node_ray_manager_port}"
            )
        else:
            command = (f"ray start --address={self.head_node_addr}:{self.head_node_ray_port} "
                    f"--node-name='ray-worker-node-rank-{self.node_rank}-{self.node_addr}' "
                    f"--node-manager-port {self.node_ray_manager_port} "
            )
        ret_code, *_ = run_shell_command(command)
        if ret_code != 0:
            return False
        
        # step2: run ray.init
        env_vars = {}
        for name in self.export_env_var_names:
            if name in os.environ:
                env_vars.update(
                    {name: os.environ.get(name)}
                )
            else:
                logger.warning(f"{name} does not exist in environ")
        logger.debug(f"running ray init on node {self.node_rank} with env: {env_vars}")
        try:
            ray.init(
                runtime_env={"env_vars": env_vars}, 
                namespace=self.cluster_namespace, 
                _node_ip_address=self.head_node_addr, 
                log_to_driver=True
            )
            # must set namespace, or the actor cannot get by name
            logger.debug(f"ray init on node {self.node_rank} finished")
        except Exception as e:
            logger.error(f"ray init on node {self.node_rank} failed due to {e}")
            return False
        
        # step3: head node spin wait for all nodes
        if self.is_head_node:
            alive_nodes = 0
            while alive_nodes < self.cluster_nodes_count:
                nodes = ray.nodes()
                alive_nodes = sum(1 for node in nodes if node["Alive"] == True)
                logger.debug(f"waiting for cluster nodes to start, {alive_nodes}/{self.cluster_nodes_count} ray nodes started")
                time.sleep(self.head_node_spin_wait_interval)
        # _ = run_shell_command("ray status")
        logger.info(ray_nodes_info_formatter(ray.nodes()))

        return True
    
    def _stop_ray_node(self):
        try:
            ray.shutdown()
            logger.debug(f"shutting down ray on node {self.node_rank}")
        except Exception as e:
            logger.debug(f"shutting down ray on node {self.node_rank} failed due to {e}")
        ret_code, *_ = run_shell_command("ray stop")
        if ret_code == 0:
            logger.debug(f"ray stopped on node {self.node_rank}")
