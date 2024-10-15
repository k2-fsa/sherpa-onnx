# Introduction

This document describes how the [tts](./tts) folder is created.


```bash
flutter create --platforms windows,macos,linux,android,ios tts
```

It prints the following:

```
Developer identity "Apple Development: xxx@zzz.com (xxxxxxx)" selected for iOS code signing
Creating project tts...
Resolving dependencies in `tts`... (1.3s)
Downloading packages...
Got dependencies in `tts`.
Wrote 122 files.

All done!
You can find general documentation for Flutter at: https://docs.flutter.dev/
Detailed API documentation is available at: https://api.flutter.dev/
If you prefer video documentation, consider: https://www.youtube.com/c/flutterdev

In order to run your application, type:

  $ cd tts
  $ flutter run

Your application code is in tts/lib/main.dart.
```

```
cd tts
flutter pub get
flutter build macos
flutter run -d macos
```
