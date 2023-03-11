import os
import logging
import pytest
import subprocess

def container_run(runtime_container, cmd):
    container = runtime_container['runtime_container']
    logging.info(cmd)
    ret, output = container.exec_run(
        f'bash -c "{cmd}"',
        tty=True)
    output = output.decode()
    logging.info(f'------>\n{output}')
    m = re.search('(?<=please check ).+\\.log', output)
    if m:
        log_fn = m.group(0).replace('/workspace', '.')
        with open(log_fn) as f:
            logging.info(f'cat {log_fn}\n{f.read()}')

    assert ret == 0

@pytest.mark.runtime
def test_run_nntc(runtime_docker):
    swap_server = os.environ['SWAP_SERVER']
    nntc_model_tar = os.environ['NNTC_MODEL_TAR']
    tar_name, tar_extension = os.path.splitext(nntc_model_tar)
    subprocess.run(
        f'bash -c "wget {os.path.join(swap_server,nntc_model_tar)}"',
        shell=True, check=True)
    subprocess.run(
        f'bash -c "tar -xvf {nntc_model_tar}"',
        shell=True, check=True)
    logging.info(f'test_run_nntc:{nntc_model_tar}')
    container_run(runtime_container, f'python3 -m tpu_perf.run {tar_name}')
