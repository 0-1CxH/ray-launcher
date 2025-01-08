import ray
from enum import Enum
from loguru import logger

class ClusterStatus(Enum):
    INIT = 0
    READY = 1
    TEARING_DOWN = 2
    FINISHED = 3

@ray.remote
class ClusterStatusManager:
    def __init__(self, cluster_nodes_count):
        self.nodes_validity = {}
        self.cluster_nodes_count = cluster_nodes_count
        self.cluster_status = ClusterStatus.INIT
    
    def count_valid_nodes(self):
        return sum(self.nodes_validity.values())
    
    def set_node_validity(self, node_rank, validity):
        if node_rank not in self.nodes_validity:
            self.nodes_validity[node_rank] = False
        self.nodes_validity[node_rank] = validity
        logger.debug(f"setting node {node_rank} validity as {validity}")

        self._cluster_status_update()
    

    def _cluster_status_update(self):
        valid_node_count = self.count_valid_nodes()
        if self.cluster_status == ClusterStatus.INIT:
            logger.debug(f"valid nodes in cluster: {valid_node_count}/{self.cluster_nodes_count}")
            if valid_node_count == self.cluster_nodes_count:
                self.cluster_status = ClusterStatus.READY
                logger.info(f"cluster of {self.cluster_nodes_count} nodes is ready")
        elif self.cluster_status == ClusterStatus.READY:
            if valid_node_count < self.cluster_nodes_count:
                self.cluster_status = ClusterStatus.TEARING_DOWN
                logger.info(f"cluster of {self.cluster_nodes_count} nodes is being torn down")
        elif self.cluster_status == ClusterStatus.TEARING_DOWN:
            if valid_node_count == 0:
                self.cluster_status = ClusterStatus.FINISHED
    
    def is_cluster_ready(self):
        return self.cluster_status == ClusterStatus.READY
    
    def is_cluster_tearing_down(self):
        return self.cluster_status == ClusterStatus.TEARING_DOWN
    
    def is_cluster_finished(self):
        valid_node_count = self.count_valid_nodes()
        if valid_node_count == 0:
            self.cluster_status = ClusterStatus.FINISHED
        return self.cluster_status == ClusterStatus.FINISHED




