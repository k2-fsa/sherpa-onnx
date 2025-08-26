package com.k2fsa.sherpa.onnx;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Objects;

/*
# We support the following loading methods

## Method 1 Specify the property sherpa_onnx.native.path

We assume the path contains the libraries sherpa-onnx-jni and onnxruntime.

java \
 -Dsherpa_onnx.native.path=/Users/fangjun/sherpa-onnx/build/install/lib \
 -cp /Users/fangjun/sherpa-onnx/sherpa-onnx/java-api/build/sherpa-onnx.jar
 xxx.java

## Method 2 Specify the native jar library

java \
 -cp /Users/fangjun/sherpa-onnx/sherpa-onnx/java-api/build/sherpa-onnx.jar:/path/to/sherpa-onnx-osx-x64.jar
 xxx.java

Note that you need to replace  : in -cp with ; on windows.

## Method 3 Specify the property java.library.path

We assume the path contains the libraries sherpa-onnx-jni and onnxruntime.

java \
 -Djava.library.path=/Users/fangjun/sherpa-onnx/build/install/lib \
 -cp /Users/fangjun/sherpa-onnx/sherpa-onnx/java-api/build/sherpa-onnx.jar
 xxx.java

 */

public class LibraryUtils {
    // System property to override native library path
    private static final String NATIVE_PATH_PROP = "sherpa_onnx.native.path";
    private static final String LIB_NAME = "sherpa-onnx-jni";

    private static boolean debug = false;

    private static String detectedOS;

    public static void enableDebug() {
        debug = true;
    }

    public static void disableDebug() {
        debug = false;
    }

    public static void load() {
        // 1. Try to load from external directory specified by -Dsherpa_onnx.native.path if provided
        if (loadFromSherpaOnnxNativePath()) {
            return;
        }

        // 2. Load from resources contains in some jar file
        try {
            if (loadFromResourceInJar()) {
                return;
            }
        } catch (IOException e) {
            // pass
        }

        // 3. fallback to -Djava.library.path
        // java -Djava.library.path=C:\mylibs;D:\otherlibs -cp sherpa-onnx.jar xxx.java
        //
        // It throws if it cannot load the lib sherpa-onnx-jni
        System.loadLibrary(LIB_NAME);
    }

    // You specify -Dsherpa_onnx.native.path=/path/to/some/dir
    // where /path/to/some/dir contains the sherpa-onnx-jni and onnxruntime libs
    private static boolean loadFromSherpaOnnxNativePath() {
        String libFileName = System.mapLibraryName(LIB_NAME);
        String nativePath = System.getProperty(NATIVE_PATH_PROP);

        if (nativePath != null) {
            File nativeDir = new File(nativePath);
            File libInDir = new File(nativeDir, libFileName);
            if (nativeDir.isDirectory() && libInDir.exists()) {
                if (debug) {
                    System.out.printf("Loading from: %s\n", libInDir.getAbsolutePath());
                }

                System.load(libInDir.getAbsolutePath());
                return true;
            }
        }

        if (debug) {
            System.out.println("nativePath is null");
        }

        return false;
    }

    private static boolean loadFromResourceInJar() throws IOException {

        String libFileName = System.mapLibraryName(LIB_NAME);
        String sherpaOnnxJniPath = "sherpa-onnx/native/" + getOsArch() + '/' + libFileName;

        Path tempDirectory = null;
        try {

            if (!resourceExists(sherpaOnnxJniPath)) {
                if (debug) {
                    System.out.printf("%s does not exist\n", sherpaOnnxJniPath);
                }

                return false;
            }

            tempDirectory = Files.createTempDirectory("sherpa-onnx-java");

            if (Objects.equals(detectedOS, "osx")) {
                // for macos, we need to first load libonnxruntime.1.17.1.dylib
                String onnxruntimePath = "sherpa-onnx/native/" + getOsArch() + '/' + "libonnxruntime.1.17.1.dylib";
                if (!resourceExists(onnxruntimePath)) {
                    if (debug) {
                        System.out.printf("%s does not exist\n", onnxruntimePath);
                    }

                    return false;
                }

                File tempFile = tempDirectory.resolve("libonnxruntime.1.17.1.dylib").toFile();
                extractResource(onnxruntimePath, tempFile);
                System.load(tempFile.getAbsolutePath());
            } else {
                String onnxLibFileName = System.mapLibraryName("onnxruntime");
                String onnxruntimePath = "sherpa-onnx/native/" + getOsArch() + '/' + onnxLibFileName;
                if (!resourceExists(onnxruntimePath)) {
                    if (debug) {
                        System.out.printf("%s does not exist\n", onnxruntimePath);
                    }

                    return false;
                }

                File tempFile = tempDirectory.resolve(onnxLibFileName).toFile();
                extractResource(onnxruntimePath, tempFile);
                System.load(tempFile.getAbsolutePath());
            }

            File tempFile = tempDirectory.resolve(libFileName).toFile();
            extractResource(sherpaOnnxJniPath, tempFile);
            System.load(tempFile.getAbsolutePath());
        } finally {
            if (tempDirectory != null) {
                cleanUpTempDir(tempDirectory.toFile());
            }
        }

        return true;
    }

    // this method is copied and modified from
    // https://github.com/microsoft/onnxruntime/blob/main/java/src/main/java/ai/onnxruntime/OnnxRuntime.java#L118
    private static String getOsArch() {
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

        String detectedArch;
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

    private static void extractResource(String resourcePath, File destination) {
        if (debug) {
            System.out.printf("Copying from resource path %s to %s\n", resourcePath, destination.toPath());
        }

        try (InputStream in = LibraryUtils.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new RuntimeException("Resource not found: " + resourcePath);
            }
            Files.copy(in, destination.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract resource " + resourcePath + " to " + destination.getAbsolutePath(), e);
        }
    }

    // From ChatGPT:
    // Class.getResourceAsStream(String path) behaves differently than ClassLoader
    //  - No leading slash → relative to the package of LibraryUtils
    //  - Leading slash → absolute path relative to classpath root
    //
    // ClassLoader.getResourceAsStream always uses absolute paths relative to classpath root,
    // no leading slash needed

    private static boolean resourceExists(String path) {
        return LibraryUtils.class.getClassLoader().getResource(path) != null;
    }

    private static void cleanUpTempDir(File dir) {
        if (!dir.exists()) return;

        File[] files = dir.listFiles();
        if (files != null) {
            for (File f : files) {
                f.deleteOnExit(); // schedule each .so for deletion
            }
        }
        dir.deleteOnExit(); // schedule the directory itself
    }
}
