#!/usr/bin/env python3

# Copyright (c)  2024  Xiaomi Corporation
# Author: Fangjun Kuang
import yaml


def main():
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    print("vocab size", len(config["token_list"]))

    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i, token in enumerate(config["token_list"]):
            f.write(f"{token} {i}\n")
    print("Generate tokens.txt successfully")


if __name__ == "__main__":
    main()
