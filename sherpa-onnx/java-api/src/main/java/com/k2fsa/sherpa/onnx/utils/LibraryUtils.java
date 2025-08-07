package com.k2fsa.sherpa.onnx.utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import com.k2fsa.sherpa.onnx.core.Core;

public class LibraryUtils {
  // -- supported platform identifiers
  public static final String DARWIN_ARM64 = "darwin_arm64";
  public static final String DARWIN_X64 = "darwin_x64";

  public static final String LINUX_ARM64 = "linux_arm64";
  public static final String LINUX_X64 = "linux_x64";
  public static final String LINUX_X86 = "linux_x86";

  public static final String WIN_ARM64 = "win_arm64";
  public static final String WIN_X64 = "win_x64";
  public static final String WIN_X86 = "win_x86";

  // System property to override native library path
  private static final String NATIVE_PATH_PROP = "sherpa_onnx.native.path";

  public static void load() {
    String libFileName = System.mapLibraryName(Core.NATIVE_LIBRARY_NAME);

    // 1. Try loading from external directory if provided
    String nativePath = System.getProperty(NATIVE_PATH_PROP);
    if (nativePath != null) {
      File nativeDir = new File(nativePath);
      File libInDir = new File(nativeDir, libFileName);
      if (nativeDir.isDirectory() && libInDir.exists()) {
        System.out.println("Loading native lib from external directory: " + libInDir.getAbsolutePath());
        System.load(libInDir.getAbsolutePath());
        return;
      }
    }

    // 2. Fallback to extracting and loading from the JAR
    File libFile = init(libFileName);
    System.out.println("Loading native lib from: " + libFile.getAbsolutePath());
    System.load(libFile.getAbsolutePath());
  }

  private static File init(String libFileName) {
    String osName = System.getProperty("os.name").toLowerCase();
    String osArch = System.getProperty("os.arch").toLowerCase();
    String userHome = System.getProperty("user.home");
    System.out.printf("Detected OS=%s, ARCH=%s, HOME=%s%n", osName, osArch, userHome);

    String archName;
    if (osName.contains("win")) {
      if (osArch.contains("aarch64") || osArch.contains("arm")) {
        archName = WIN_ARM64;
      } else if (osArch.contains("64")) {
        archName = WIN_X64;
      } else {
        archName = WIN_X86;
      }

    } else if (osName.contains("mac")) {
      if (osArch.contains("aarch64") || osArch.contains("arm")) {
        archName = DARWIN_ARM64;
      } else {
        archName = DARWIN_X64;
      }

    } else if (osName.contains("nix") || osName.contains("nux") || osName.contains("aix") || osName.contains("linux")) {
      if (osArch.contains("aarch64") || osArch.contains("arm")) {
        archName = LINUX_ARM64;
      } else if (osArch.contains("64")) {
        archName = LINUX_X64;
      } else {
        archName = LINUX_X86;
      }

    } else {
      throw new UnsupportedOperationException("Unsupported OS: " + osName);
    }

    // Prepare destination directory under ~/lib/<archName>/
    String dstDir = userHome + File.separator + "lib" + File.separator + archName;
    File libFile = new File(dstDir, libFileName);
    File parentDir = libFile.getParentFile();
    if (!parentDir.exists() && !parentDir.mkdirs()) {
      throw new RuntimeException("Unable to create directory: " + parentDir);
    }

    // Extract the native library from JAR
    extractResource("/native/" + archName + "/" + libFileName, libFile);
    return libFile;
  }

  /**
   * Copies a resource file from the jar to the specified destination.
   *
   * @param resourcePath The resource path inside the jar, e.g.:
   *                     /native/linux_x64/libonnxruntime.so
   * @param destination  The destination file on disk
   */
  private static void extractResource(String resourcePath, File destination) {
    try (InputStream in = LibraryUtils.class.getResourceAsStream(resourcePath)) {
      if (in == null) {
        throw new RuntimeException("Resource not found: " + resourcePath);
      }
      Files.copy(in, destination.toPath(), StandardCopyOption.REPLACE_EXISTING);
    } catch (IOException e) {
      throw new RuntimeException("Failed to extract resource " + resourcePath + " to " + destination.getAbsolutePath(),
          e);
    }
  }
}
