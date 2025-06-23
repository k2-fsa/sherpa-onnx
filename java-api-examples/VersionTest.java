// Copyright 2025 Xiaomi Corporation

import com.k2fsa.sherpa.onnx.*;

public class VersionTest {
  public static void main(String[] args) {
    System.out.printf("sherpa-onnx version: %s\n", VersionInfo.getVersion());
    System.out.printf("sherpa-onnx gitSha1: %s\n", VersionInfo.getGitSha1());
    System.out.printf("sherpa-onnx gitDate: %s\n", VersionInfo.getGitDate());
  }
}
