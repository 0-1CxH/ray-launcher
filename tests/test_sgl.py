import os

from torch.distributed.device_mesh import init_device_mesh
from sglang.srt.entrypoints.verl_engine import VerlEngine

from ray_launcher import ClusterLauncher, BaseLocalModule, RemoteModule
import ray



class SGLModule(BaseLocalModule):
    def start(self, model_path, tp_size, pp_size, dp_size):
        self.device_mesh_cpu = init_device_mesh(
            "cpu", mesh_shape=(tp_size, dp_size, pp_size),  
            mesh_dim_names=["tp", "dp", "pp"]
        )
        self.tp_rank = self.device_mesh_cpu.get_local_rank("tp")
        self.dp_rank = self.device_mesh_cpu.get_local_rank("dp")
        print(f"TP {self.tp_rank}/{tp_size}, DP {self.dp_rank}/{dp_size}, IP {self.get_ip_address()}, Devices {self.get_devices_in_environ()}")

        self.fragment = VerlEngine(
            model_path=model_path,
            mem_fraction_static=0.5,
            device_mesh_cpu=self.device_mesh_cpu['tp'],
            base_gpu_id=0, # self.dp_rank,
            gpu_id_step=1, # dp_size,
            dist_init_addr=f"{self.get_ip_address()}:{self.get_avaiable_port()}",
        )
    
    def generate(self, prompts):
        prompt = prompts[self.dp_rank]
        output = self.fragment.generate(
        prompt=prompt,
        sampling_params=dict(max_new_tokens=16, temperature=0.0),
        )
        return output
    
    def shutdown(self):
        self.fragment.shutdown()


with ClusterLauncher(
    cluster_nodes_count=int(os.environ["NNODES"]),
    head_node_addr=os.environ["MASTER_ADDR"],
    export_env_var_names=["RAY_DISABLE_DOCKER_CPU_WARNING"]
) as launcher:
    bundle = [{"GPU": 2, "CPU": 8}]
    pg = ray.util.placement_group(bundle, strategy="PACK")
    print(f"created pg")

    m = RemoteModule(
        SGLModule, [(pg, 0)], 
        is_discrete_gpu_module=True,
        resource_reservation_ratio=1.0,
        call_policy="ALL", collect_policy="ALL",
        backend_actor_kwargs={}
    )

    m.start(**{
            "model_path": "../Qwen2.5-0.5B",
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 2
        })

    print(m.generate([
        ["1+1=2, 1+2=3, 1+3=4, 1+4=", "9-1=8, 8-1=7, 7-1="],
        ["2*1=2, 2*2=4, 2*3=", "8/2=4, 6/2="],
    ]
    ))