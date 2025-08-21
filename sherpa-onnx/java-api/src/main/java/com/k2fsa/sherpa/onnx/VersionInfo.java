package com.k2fsa.sherpa.onnx;

public class VersionInfo {

    public static String getVersion() {
        LibraryLoader.maybeLoad();
        return getVersionStr2();
    }

    public static String getGitSha1() {
        LibraryLoader.maybeLoad();
        return getGitSha12();
    }

    public static String getGitDate() {
        LibraryLoader.maybeLoad();
        return getGitDate2();
    }

    private static native String getVersionStr2();
    private static native String getGitSha12();
    private static native String getGitDate2();
}
