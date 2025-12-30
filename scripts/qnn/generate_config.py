#!/usr/bin/env python3

# see
# https://github.com/MollySophia/rwkv-qualcomm/blob/2a82c641c90ee130cbd7038ca7449b2fa818de71/utils/htp_devices_config.py
# https://docs.qualcomm.com/bundle/publicresource/topics/80-64748-1/model_prep_linux.html#QNN-HTP-context-binary

import argparse
import json
from pathlib import Path

from device_info import soc_info_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--soc",
        type=str,
        required=True,
        help="SM8850, SA8295, etc",
    )

    parser.add_argument(
        "--graph-name",
        type=str,
        required=True,
        help="Graph name",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save the generated json files",
    )

    parser.add_argument(
        "--qnn-sdk-root",
        type=str,
        required=True,
        help="Path to qnn sdk",
    )

    return parser.parse_args()


def generate_config(
    soc_name: str,
    graph_name: str,
    output_dir: str,
    qnn_sdk_root: str,
):
    if soc_name not in soc_info_dict:
        raise ValueError(
            f"Unsupported SOC {soc_name}. Supported: - {sorted(list(soc_info_dict.keys()))}"
        )
    soc = soc_info_dict[soc_name]

    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    htp_backend_extensions_data = {
        "backend_extensions": {
            "shared_library_path": f"{qnn_sdk_root}/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
            "config_file_path": f"{output_dir}/htp_config.json",
        }
    }

    htp_backend_config_data = {
        "graphs": [
            {
                "vtcm_mb": soc.info.vtcm_size_in_mb,
                "O": 3,
                "graph_names": [graph_name],
            }
        ],
        "devices": [
            {
                "device_id": 0,
                "soc_id": soc.model.value,
                "dsp_arch": soc.info.arch.name,
                "cores": [
                    {
                        "core_id": 0,
                        "perf_profile": "burst",
                        "rpc_control_latency": 200,
                    }
                ],
            }
        ],
    }

    with open(str(output_dir / "htp_backend_extensions.json"), "w") as f:
        json.dump(htp_backend_extensions_data, f, indent=4)

    with open(str(output_dir / "htp_config.json"), "w") as f:
        json.dump(htp_backend_config_data, f, indent=4)


def _test():
    qnn_sdk_root = "/home/fangjun/open-source/qairt/2.40.0.251030"
    generate_config(
        soc_name="SM8850",
        graph_name="model_10_seconds_quantized",
        output_dir="./tmp",
        qnn_sdk_root=qnn_sdk_root,
    )


if __name__ == "__main__":
    #  _test()

    args = get_args()
    print(vars(args))
    generate_config(
        soc_name=args.soc,
        graph_name=args.graph_name,
        output_dir=args.output_dir,
        qnn_sdk_root=args.qnn_sdk_root,
    )

# ./generate_config.py  --soc SM8850 --graph-name abc --output-dir ./tmp2 --qnn-sdk-root $QNN_SDK_ROOT
