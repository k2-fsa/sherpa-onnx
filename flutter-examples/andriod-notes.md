# Note about android

Useful commands

```bash
flutter build apk --split-per-abi --release
```

The above commands print the following:

```
✓ Built build/app/outputs/flutter-apk/app-armeabi-v7a-release.apk (94.8MB)
✓ Built build/app/outputs/flutter-apk/app-arm64-v8a-release.apk (96.1MB)
✓ Built build/app/outputs/flutter-apk/app-x86_64-release.apk (96.9MB)
```

Note that it does not generate APK for `x86`.

```
adb install build/app/outputs/flutter-apk/app-arm64-v8a-release.apk
```
