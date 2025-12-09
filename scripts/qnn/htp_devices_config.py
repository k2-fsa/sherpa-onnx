#!/usr/bin/env python3

# see
# https://github.com/MollySophia/rwkv-qualcomm/blob/2a82c641c90ee130cbd7038ca7449b2fa818de71/utils/htp_devices_config.py
# https://docs.qualcomm.com/bundle/publicresource/topics/80-64748-1/model_prep_linux.html#QNN-HTP-context-binary
import json

htp_devices = {
    "SM8850": {
        "dsp_arch": "v81",
        "soc_id": 87,
    },
    "SM8750": {
        "dsp_arch": "v79",
        "soc_id": 69,
    },
    "SM8650": {
        "dsp_arch": "v75",
        "soc_id": 57,
    },
    "SM8550": {
        "dsp_arch": "v73",
        "soc_id": 43,
    },
    "SC8380": {
        "dsp_arch": "v73",
        "soc_id": 60,
    },
    "SM8475": {
        "dsp_arch": "v69",
        "soc_id": 42,
    },
    "SM8635": {
        "dsp_arch": "v73",
        "soc_id": 68,
    },
}


def dump_htp_config(
    soc_name: str,
    graph_names: list,
    output_path: str,
    old_qnn: bool = False,
    weights_sharing: bool = True,
):
    if not soc_name in htp_devices.keys():
        raise ValueError(f"Invalid SoC name: {soc_name}")

    if graph_names is None or len(graph_names) == 0:
        raise ValueError("Invalid graph names")

    for i in range(len(graph_names)):
        graph_names[i] = graph_names[i].replace("lib", "").replace("-", "_")

    config = {
        "graphs": {
            "vtcm_mb": 0,
            "O": 3,
            "graph_names": graph_names,
        },
        "devices": [
            {
                "dsp_arch": htp_devices[soc_name]["dsp_arch"],
                "device_id": 0,
                "soc_id": htp_devices[soc_name]["soc_id"],
                "pd_session": "unsigned",
                "cores": [
                    {
                        "core_id": 0,
                        # default, balanced, high_performance, burst
                        "perf_profile": "balanced",
                        "rpc_control_latency": 200,
                    }
                ],
            }
        ],
    }

    if soc_name != "SM8635":
        config["graphs"]["fp16_relaxed_precision"] = 1

    if not old_qnn:
        config["graphs"] = [config["graphs"]]

    if weights_sharing:
        config["context"].update({"weight_sharing_enabled": True})

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)


def dump_htp_link_config(output_path: str, qnn_sdk_root_path: str):
    link = {
        "backend_extensions": {
            "shared_library_path": f"{qnn_sdk_root_path}/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
            "config_file_path": output_path.replace("link.json", "config.json"),
        }
    }
    with open(output_path, "w") as f:
        json.dump(link, f, indent=4)


if __name__ == "__main__":
    qnn_sdk_root = "/home/fangjun/open-source/qairt/2.40.0.251030"
    dump_htp_config(
        soc_name="SM8850",
        graph_names=["model_10_seconds_quantized"],
        output_path="libmodel.so".replace(".so", "_htp_config.json"),
    )
    dump_htp_link_config("libmodel.so".replace(".so", "_htp_link.json"), qnn_sdk_root)
