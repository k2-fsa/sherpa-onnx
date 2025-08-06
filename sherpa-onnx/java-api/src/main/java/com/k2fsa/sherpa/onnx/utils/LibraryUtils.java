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

  public static void load() {
    String osName = System.getProperty("os.name").toLowerCase();
    String osArch = System.getProperty("os.arch").toLowerCase();
    String userHome = System.getProperty("user.home");
    System.out.printf("Detected OS=%s, ARCH=%s, HOME=%s%n", osName, osArch, userHome);

    String archName;
    String libFileName;

    // --- determine library name + subdirectory
    if (osName.contains("win")) {
      libFileName = Core.WIN_NATIVE_LIBRARY_NAME;
      if (osArch.contains("aarch64") || osArch.contains("arm")) {
        archName = WIN_ARM64;
      } else if (osArch.contains("64")) {
        archName = WIN_X64;
      } else {
        archName = WIN_X86;
      }

    } else if (osName.contains("mac")) {
      libFileName = Core.MACOS_NATIVE_LIBRARY_NAME;
      if (osArch.contains("aarch64") || osArch.contains("arm")) {
        archName = DARWIN_ARM64;
      } else {
        archName = DARWIN_X64;
      }

    } else if (osName.contains("nix") || osName.contains("nux") || osName.contains("aix") || osName.contains("linux")) {
      libFileName = Core.UNIX_NATIVE_LIBRARY_NAME;
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

    // --- prepare destination directory under ~/lib/<archName>/
    String dstDir = userHome + File.separator + "lib" + File.separator + archName;
    File libFile = new File(dstDir, libFileName);
    File parentDir = libFile.getParentFile();
    if (!parentDir.exists() && !parentDir.mkdirs()) {
      throw new RuntimeException("Unable to create directory: " + parentDir);
    }

    // --- extract the native library
    extractResource("/lib/" + archName + "/" + libFileName, libFile);

    // --- finally load the main library
    System.out.println("Loading native lib from: " + libFile.getAbsolutePath());
    System.load(libFile.getAbsolutePath());
  }

  /**
   * Copies a resource file from the jar to the specified destination.
   *
   * @param resourcePath The resource path inside the jar, e.g.:
   *                     /lib/win_x64/onnxruntime.dll
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
