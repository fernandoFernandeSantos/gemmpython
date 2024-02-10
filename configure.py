#!/usr/bin/python3
import argparse
import configparser
import json
import shlex
import subprocess
import sys
from socket import gethostname
import os
from pathlib import Path

CUDA_PATH = "/usr/local/cuda"
CONFIG_FILE = "/etc/radiation-benchmarks.conf"

ALPHA = 1.5
BETA = 0.34
SIZES_AND_LOG_INTERVAL = [
    {"size": 8192, "log_interval": 1},
]

ITERATIONS = int(1e12)
CHECK_INPUT_EXISTENCE = False


def general_configure():
    try:
        config = configparser.RawConfigParser()
        config.read(CONFIG_FILE)
        server_ip = config.get('DEFAULT', 'serverip')
    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))
    hostname = gethostname()
    home = str(Path.home())
    jsons_path = os.path.abspath(f"data/{hostname}_jsons")
    if os.path.isdir(jsons_path) is False:
        os.makedirs(jsons_path, exist_ok=True)
    current_directory = os.path.abspath(os.getcwd())
    return current_directory, home, jsons_path, server_ip


def test_all_jsons(enable_console_logging, timeout):
    hostname = gethostname()
    current_directory = os.getcwd()
    for size_log_config in SIZES_AND_LOG_INTERVAL:
        size = size_log_config["size"]
        default_config = f"size_{size}_tensor_False_cublas_True_precision_float"
        json_file_name = f"{current_directory}/data/{hostname}_jsons/{default_config}.json"
        with open(json_file_name, "r") as fp:
            json_data = json.load(fp)

        for v in json_data:
            exec_str = v["exec"] + (" --verbose" if enable_console_logging else "")
            print("EXECUTING", exec_str)
            try:
                completed_process = subprocess.run(shlex.split(exec_str), check=True, timeout=timeout,
                                                   stdout=sys.stdout, stderr=subprocess.STDOUT)
                completed_process.check_returncode()
            except subprocess.TimeoutExpired:
                print("Timeout raised while executing the command. It should be ok!")
            except Exception as e:
                # Raise the exception if it's not a TimeoutExpired error
                if not isinstance(e, subprocess.TimeoutExpired):
                    raise


def configure():
    current_directory, home, jsons_path, server_ip = general_configure()
    binary_name = "main.py"
    print(f"Generating {binary_name} for CUDA")
    data_dir = f"{current_directory}/data"

    path_to_bin = os.path.abspath(binary_name)

    # gen only for max size, defined on cuda_trip_mxm.cu
    for size_log_config in SIZES_AND_LOG_INTERVAL:
        size = size_log_config["size"]
        default_config = f"size_{size}_tensor_False_cublas_True_precision_float"
        default_path = f'{default_config}.matrix'
        execute = [
                      f"{path_to_bin}",
                      f'--alpha {ALPHA} --beta {BETA}',
                      f'--gold_file {data_dir}/GOLD_{default_path}',
                      f'--iterations {ITERATIONS}',
                  ] + [
                      f"--{cfg_size} {cfg_size_val}" for cfg_size, cfg_size_val in size_log_config.items()
                  ]

        # change mode and iterations for exe
        json_file_name = f"{jsons_path}/{default_config}.json"

        execute_json = [{
            "killcmd": f"pkill -9 -f {binary_name}",
            "exec": ' '.join(execute),
            "codename": binary_name,
            "header": ' '.join(execute[1:])
        }]

        with open(json_file_name, "w") as fp:
            json.dump(execute_json, indent=4, fp=fp)

        generate = execute + ['--generate', '--verbose']
        generate_string = ' '.join(generate)
        if os.system(generate_string) != 0:
            raise SystemExit(f"Could not generate gold using command:\n{generate_string}")


def main():
    parser = argparse.ArgumentParser(description='Configure a setup', add_help=True)
    parser.add_argument('--testjsons', default=0,
                        help="How many seconds to test the jsons, if 0 (default) it does the configure", type=int)
    parser.add_argument('--enableconsole', default=False, action="store_true",
                        help="Enable console logging for testing")

    args = parser.parse_args()

    if args.testjsons != 0:
        test_all_jsons(enable_console_logging=args.enableconsole, timeout=args.testjsons)
    else:
        configure()


if __name__ == "__main__":
    main()
