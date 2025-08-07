package com.k2fsa.sherpa.onnx.utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

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

  /* Computes and initializes OS_ARCH_STR (such as linux-x64) */
  private static String initOsArch() {
    String detectedOS = null;
    String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
    if (os.contains("mac") || os.contains("darwin")) {
      detectedOS = "osx";
    } else if (os.contains("win")) {
      detectedOS = "win";
    } else if (os.contains("nux")) {
      detectedOS = "linux";
    } else {
      throw new IllegalStateException("Unsupported os:" + os);
    }
    String detectedArch = null;
    String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH);
    if (arch.startsWith("amd64") || arch.startsWith("x86_64")) {
      detectedArch = "x64";
    } else if (arch.startsWith("x86")) {
      // 32-bit x86 is not supported by the Java API
      detectedArch = "x86";
    } else if (arch.startsWith("aarch64")) {
      detectedArch = "aarch64";
    } else {
      throw new IllegalStateException("Unsupported arch:" + arch);
    }
    return detectedOS + '-' + detectedArch;
  }

  private static File init(String libFileName) {
    String osName = System.getProperty("os.name").toLowerCase();
    String osArch = System.getProperty("os.arch").toLowerCase();
    String userHome = System.getProperty("user.home");
    System.out.printf("Detected OS=%s, ARCH=%s, HOME=%s%n", osName, osArch, userHome);

    String archName = initOsArch();

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
