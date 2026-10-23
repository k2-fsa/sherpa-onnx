# Android APK signing keystore

The APKs built by CI are signed with this shared keystore so that every build
of an app has the **same signing identity** — otherwise each CI job makes a
fresh debug key and installing a new build over an old one fails on many
OEM installers (e.g. MIUI's "packageInfo is null").

## The keystore itself

The file `ci-apk.keystore` is **not committed** (it is gitignored). It exists
only as the GitHub Actions secret `ANDROID_KEYSTORE_B64`, which is the
base64 of the file. `sign_apk.sh` decodes it to `$RUNNER_TEMP` at signing
time.

- **Alias:** `ci`
- **Store / key password:** `android`
- **Algorithm:** RSA 2048, SHA384withRSA
- **Subject:** `CN=sherpa-onnx CI APK, OU=sherpa-onnx, O=k2-fsa, C=CN`
- **Valid:** 2026-10-23 → 2054-03-10 (10,000 days)

### How it was generated

With [`generate_keystore.sh`](generate_keystore.sh), which runs:

```bash
keytool -genkeypair \
  -keystore ci-apk.keystore \
  -storepass android -key-pass android \
  -alias ci -keyalg RSA -keysize 2048 -validity 10000 \
  -dname "CN=sherpa-onnx CI APK, OU=sherpa-onnx, O=k2-fsa, C=CN"
```

`keytool -list -v -keystore ci-apk.keystore -storepass android -alias ci`
prints the validity dates above.

**This command cannot reproduce the same key.** `keytool -genkeypair` draws
fresh randomness on every run — following it gives you a *functionally
equivalent* keystore (same alias, password, subject and validity) but a
**different signing identity**. Apps signed with your own key cannot upgrade
over apps signed with the official one. The one true identity is the secret;
to verify which key signed an APK:

```bash
apksigner verify --print-certs app.apk | grep SHA-256
```

The official certificate's SHA-256 fingerprint is:

```
48:DE:24:FF:98:34:8D:51:80:2E:CD:2B:8B:0A:A6:6D:3F:90:1A:C2:60:63:BB:0B:EC:37:0B:96:05:58:E0:4D
```

### How the secret is set

```bash
base64 -i ci-apk.keystore | gh secret set ANDROID_KEYSTORE_B64 --repo k2-fsa/sherpa-onnx
```

or copy the base64 to the clipboard and paste it under
**Settings → Secrets and variables → Actions** in the GitHub UI:

```bash
base64 -i ci-apk.keystore | pbcopy
```

Fork PRs do not receive secrets; the signing step fails there with a clear
message instead of shipping unsigned APKs.

## Rotating

See [change-keystore.md](change-keystore.md) for the full procedure. In
short: replacing the secret changes the app's signing identity — users must
uninstall the old app before installing builds signed with the new key.
