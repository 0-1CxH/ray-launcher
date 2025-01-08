import os
import ray
# import socket
import random
from loguru import logger

class BaseBackend:
    def __init__(self):
        self.backend_name = self.__class__.__name__ + str(id(self))

    def get_gpu_ids(self):
        return ray.get_gpu_ids()
    
    def get_ip_address(self):
        return ray.util.get_node_ip_address()
    
    def get_avaiable_port(self):
        port = random.randint(58000, 62000)
        return port
        # while True:
        #     port = random.randint(58000, 62000)
        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #         is_valid = s.connect_ex(('localhost', port)) == 0
        #     if is_valid:
        #         return port


    def set_environ(self, name, val):
        old_val =  os.environ.get(name)
        if old_val:
            logger.debug(f"environ {name} of backend {self.backend_name} is already a non-empty value ({old_val}), now overriding with {val}")
        os.environ[name] = str(val)
    
    def set_distributed_environs(
            self,
            rank,
            world_size,
            master_addr,
            master_port
    ):
        self.set_environ("RANK", rank)
        gpu_ids = self.get_gpu_ids()
        assert len(gpu_ids) == 1
        gpu_id = gpu_ids.pop()
        self.set_environ("LOCAL_RANK", gpu_id)
        self.set_environ("WORLD_SIZE", world_size)
        self.set_environ("MASTER_ADDR", master_addr)
        self.set_environ("MASTER_PORT", master_port)

    def format_distributed_info(self):
        s = "distributed environs: "
        for e_name in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]:
            s += f"{e_name}={os.environ.get(e_name)} "
        return s
