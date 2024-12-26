import socket
import subprocess
from loguru import logger

def get_self_addr():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception as e:
        logger.warning(f"failed to get addr by hostname, due to: {e}")
    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning(f"failed to get addr by connect to external, due to {e}")
    finally:
        s.close()
    
    logger.error(f"cannot get current node's addr")
    raise RuntimeError

def run_shell_command(cmd):
    logger.debug(f"executeing command: >>>{cmd}")
    retval = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    if retval.returncode == 0:
        logger.debug(retval.stdout)
    else:
        logger.error(f"command returns non-zero value: {retval.stderr}")
    return  retval.returncode, retval.stdout, retval.stderr




def ray_nodes_info_formatter(ray_nodes):
    
    def _ray_node_formatter(node):
        is_alive = node.get('Alive', False)
        s = f"Node {node.get('NodeID', 'N/A')} ({node.get('NodeName', 'N/A')}) [{'ALIVE' if is_alive else 'DEAD'}]\n"
        s += f"NodeManager: {node.get('NodeManagerAddress', 'N/A')}:{node.get('NodeManagerPort', 'N/A')} ({node.get('NodeManagerHostname', 'N/A')})\n"
        s += f"ObjectStoreSocketName: {node.get('ObjectStoreSocketName', 'N/A')}, ObjectManagerPort: {node.get('ObjectManagerPort', 'N/A')}\n"
        s += f"RayletSocketName: {node.get('RayletSocketName', 'N/A')}\n"
        if is_alive:
            s += f"Resources:\n"
            for resource, value in node.get('Resources', {}).items():
                s += f"  {resource}: {value}\n"
        else:
            s += f"DeathReason: {node.get('DeathReason', 'N/A')}\n"
            s += f"DeathReasonMessage: {node.get('DeathReasonMessage', '')}\n"
        
        s += "\n"
        return s
    
    s = f"ray nodes information: \n"
    for node in ray_nodes:
        s += _ray_node_formatter(node)
    
    return s