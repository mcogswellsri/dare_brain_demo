#!/usr/bin/env python3
import subprocess
from subprocess import Popen
import time
import tempfile
import socket
import argparse
import sys
import os
import os.path as pth
import random
import yaml
import getpass

import jinja2

#######################################
# Generic code that runs stuff

parser = argparse.ArgumentParser(description=
'''
Generic cluster run script.
''')
parser.add_argument('EXP', type=str, nargs='?', help='experiment to run',
                    default=None)
parser.add_argument('-m', '--mode',
                        help='which command to run',
                        choices=['singlefeats'])

parser.add_argument('-g', '--ngpus', type=int, default=1,
                    help='number of gpus to use')
parser.add_argument('--gid', type=int, default=-1,
                    help='which gpu to use')

parser.add_argument('-s', '--shell', dest='shell', action='store_true',
                    help='Start a shell in the container, but do not '
                         'actually run the job')
parser.add_argument('-l', '--local', action='store_true')

parser.add_argument('--run_dir', type=str, default='./run_data/')

args = parser.parse_args()




def local_setup():
    # Scratch directory for the container
    # TODO: add local_config.yaml.example
    try:
        with open('local_config.yaml', 'r') as f:
            local_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print('Create a local_config.yaml file using local_config.yaml.example '
              'as a template.')
        raise e
    base_data_dir = local_config['mounts']['data']
    Popen(f'mkdir -p {base_data_dir}/experiments/', shell=True).wait()
    uname = getpass.getuser()
    namespace = 'local'
    volumes = []
    for dst, src in local_config['mounts'].items():
        dst = pth.join('/usr', dst) + '/'
        # (local path, container path)
        volumes.append((src, dst))
    return uname, namespace, volumes


def environment_setup(args):
    # working space for this script
    os.makedirs(args.run_dir, exist_ok=True)
    # Places for this script to store meta-data
    Popen(f'mkdir -p {args.run_dir}/cmds/', shell=True).wait()
    Popen(f'mkdir -p {args.run_dir}/pods/', shell=True).wait()
    Popen(f'mkdir -p {args.run_dir}/logs/', shell=True).wait()
    # This needs to be in the directory copied to the container
    Popen(f'mkdir -p {args.run_dir}/tmp/', shell=True).wait()

    # environment dependent variables (try to only use these in runcmd)
    if args.local:
        uname, namespace, volumes = local_setup()
        base_exp_dir = '/usr/data/experiments/'
    else:
        with open('cvt_k8s_config.yaml', 'r') as f:
            cvt_config = yaml.safe_load(f)
        uname = cvt_config['uname']
        namespace = cvt_config['namespace']
        pvcs = cvt_config['pvcs']
        base_exp_dir = f'/usr/data/{uname}/experiments/trojai_gen/'

    return locals()


def runcmd(cmd, external_log=None, work_dir=None):
    '''
    Run cmd, a string containing a command, in a bash shell using gpus.
    '''
    # TODO: use log stuff or remove
    #log_fname = 'data/logs/job_{}_{:0>3d}_{}.log'.format(
    #                int(time.time()), random.randint(0, 999), jobid)
    #if external_log:
    #    if pth.lexists(external_log):
    #        os.unlink(external_log)
    #    link_name = pth.relpath(log_fname, pth.dirname(external_log))
    #    os.symlink(link_name, external_log)
    #jobid += 1
    # write SLURM job id then run the command
    script_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                         dir=f'{args.run_dir}/cmds/', prefix='.', suffix='.docker.sh')
    exp_dir = pth.join(base_exp_dir, args.EXP)
    script_file.write(f'mkdir -p {exp_dir}\n')
    if work_dir is not None:
        script_file.write(f'echo cd {work_dir}\n')
        script_file.write(f'cd {work_dir}\n')
    script_file.write('echo ' + cmd + '\n')
    #script_file.write('echo "host: $HOSTNAME"\n')
    if args.ngpus >= 1:
        script_file.write('nvidia-smi\n')
    script_file.write(cmd)
    script_file.close()
    # use this to restrict runs to current host
    #hostname = socket.gethostname()
    #cmd = ' -w {} bash '.format(hostname) + script_file.name

    ## run it on this machine
    #if args.p == 'debug_local':
    #    if args.shell:
    #        cmd = 'bash '
    #    else:
    #        cmd = 'bash ' + script_file.name
    #    Popen(cmd, shell=True).wait()
    #    return

    # so build the docker image
    sanitized_exp_name = args.EXP.replace('-', 'neg').replace('.', '-')
    docker_tag = f'{uname}-{namespace}-{sanitized_exp_name}'
    Popen(f'docker build -t {docker_tag} .', shell=True).wait()
    command_file = pth.join('/usr/src/app', pth.relpath(script_file.name, os.getcwd()))

    # run it in docker on this machine
    if args.local:
        assert args.ngpus <= 1
        vol_arg = ''
        port_arg = '--network host' #'-P' #'-p 5000:5000'
        #port_arg = '-P' #'-p 5000:5000'
        for local_path, container_path in volumes:
            vol_arg += f'--volume {local_path}:{container_path} '
        if args.shell:
            cmd = f'docker run -it --runtime nvidia {vol_arg} {port_arg} {docker_tag} bash'
            print(f'command: ')
            print(f'/bin/bash {command_file}')
        else:
            cmd = f'bash {command_file}'
            cmd = f'docker run -it --runtime nvidia {vol_arg} {port_arg} {docker_tag} {cmd}'
        print(cmd)
        Popen(cmd, shell=True).wait()

    # run it on the cvt-k8s cluster
    else:

        # push the container to the registry
        # tag
        registry_url = 'open.docker.sarnoff.com'
        tag_cmd = f'docker tag {docker_tag} {registry_url}/{docker_tag}'
        print(tag_cmd)
        Popen(tag_cmd, shell=True).wait()
        # push
        remote_image = f'{registry_url}/{docker_tag}'
        push_cmd = f'docker push {remote_image}'
        print(push_cmd)
        Popen(push_cmd, shell=True).wait()

        # figure out what to run
        if args.shell:
            cmd = 'command: ["/bin/bash"]'
            cmd_args = 'args: ["-c", "--", "while true; do sleep 10; echo hello > /dev/null; done;"]'
            print(f'command: ')
            print(f'/bin/bash {command_file}')
            max_runtime = 14400 # 4 hours
        else:
            cmd = 'command: ["/bin/bash"]'
            cmd_args = f'args: ["{command_file}"]'
            max_runtime = 259200 # 3 days

        # configure the pod
        with open('cvt_k8s_pod_template.yaml', 'r') as f:
            template = jinja2.Template(f.read())
        pod_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                         dir=f'{args.run_dir}/pods/', prefix='.', suffix='.yml')
        pod_name = docker_tag + '-pod'
        pod_file.write(template.render({
            'pod_name': pod_name,
            'container_name': docker_tag + '-container',
            'command': cmd,
            'args': cmd_args,
            'ngpus': args.ngpus,
            'image': remote_image,
            'pvcs': pvcs,
            'max_runtime': max_runtime,
        }))
        pod_file.close()
        print(pod_file.name)

        # run the pod
        run_cmd = f'kubectl apply -n {namespace} -f {pod_file.name}'
        print(run_cmd)
        Popen(run_cmd, shell=True).wait()

        if args.shell:
            exec_cmd = f'kubectl -n {namespace} -it exec {pod_name} -- bash'
            print(f'run this for an interactive shell:\n{exec_cmd}')
            Popen(exec_cmd, shell=True).wait()


#######################################
# Detail experiment configurations

def config(exp, base_exp_dir):
    # generic experiment tracking setup
    if exp is None:
        import warnings
        warnings.warn('Missing experiment code')
        return locals()
    experiment = exp # put it in locals()
    assert exp.startswith('exp')
    exp_dir = pth.join(base_exp_dir, exp)
    log_fname = pth.join(exp_dir, 'log.txt')
    test_log_fname = pth.join(exp_dir, 'evaluate_log.txt')
    exp_vers = list(map(int, exp[3:].split('.')))
    exp_vers += [0] * (100 - len(exp_vers))

    # project specific vars
    fastmri_dicom_dir = '/usr/fastMRI_brain_DICOM/'

    # generate cifar10 models with different kinds of poisoning
    if exp_vers[0] == 0:
        run_mode = 'start_server'

    elif exp_vers[0] == 1:
        run_mode = 'test_segmentation'

    elif exp_vers[0] == 2:
        run_mode = 'generate_thumbnails'

    return locals()

def write_template(template_file, dst_file, data):
    from jinja2 import Template
    with open(template_file, 'r') as f:
        template = Template(f.read())
    with open(dst_file, 'w') as f:
        f.write(template.render(data))

#######################################
# Actually run things

locals().update(environment_setup(args))
locals().update(config(args.EXP, base_exp_dir))



if run_mode == 'start_server':
    # TODO: remove runcmd('python flask_server.py'.format(**locals()))
    runcmd('python brats_demo_server.py'.format(**locals()))

elif run_mode == 'test_segmentation':
    runcmd('python eval.py --model unet --up-mode resize \
    --vis-page 1 --vis-nifti 0 --contest-submission 0 --save-metrics 0 \
    --result-dname seg_test_for_demo \
    --pred-thres 0.5 \
    --max-examples 4 \
    --train-data /usr/brats/NONE/ \
    --val-data /usr/brats/MICCAI_BraTS2020_TrainingData/ \
    --checkpoint-epoch 1000 /usr/monai_model_data/models/unet_resize2/'.format(**locals()),
    work_dir='/usr/src/app/monai_model/')

elif run_mode == 'generate_thumbnails':
    # TODO: remove runcmd('python flask_server.py'.format(**locals()))
    runcmd('python generate_thumbnails.py'.format(**locals()))
