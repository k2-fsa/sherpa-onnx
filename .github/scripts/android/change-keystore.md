# How to change (rotate) the APK signing keystore

Do this if the key is compromised, or when you deliberately want a new app
signing identity. See [README.md](README.md) for what the keystore is and how
it is stored.

Rotating changes the **signing identity** of every APK built afterwards:
users must uninstall the old app before installing the new build. Plan the
timing accordingly.

## 1. Generate the new keystore

```bash
cd .github/scripts/android
rm ci-apk.keystore          # the generator refuses to overwrite
./generate_keystore.sh
```

If you use different parameters than the script's defaults (alias `ci`,
password `android`), remember to update `sign_apk.sh` to match
(`--ks-key-alias`, `--ks-pass`, `--key-pass`).

## 2. Put it in the Actions secret

```bash
base64 -i ci-apk.keystore | pbcopy
# then replace ANDROID_KEYSTORE_B64 under
#   Settings → Secrets and variables → Actions
```

or from the command line:

```bash
base64 -i ci-apk.keystore | gh secret set ANDROID_KEYSTORE_B64 --repo k2-fsa/sherpa-onnx
```

This is the **only step CI needs** — the workflows pass the secret to
`sign_apk.sh`, which decodes it at signing time. Nothing else in CI refers to
the key.

## 3. Update the documentation

Replace the SHA-256 fingerprint in [README.md](README.md) with the new one:

```bash
keytool -list -v -keystore ci-apk.keystore -storepass android -alias ci | grep SHA256
```

## 4. Keep local runs in sync

The local fallback file `ci-apk.keystore` (gitignored) is already the new one
after step 1 — local `sign_apk.sh` runs now sign with the new identity.

## 5. Verify

```bash
# with the new secret exported as ANDROID_KEYSTORE_B64:
sign_apk.sh some-input.apk out.apk
apksigner verify --print-certs out.apk | grep SHA-256    # must match step 3
```

## Consequences to expect

- Devices with an app signed by the **old** key must **uninstall** it before
  installing the new build (signature mismatch).
- APKs already published on the release tags keep the **old** signature; they
  cannot be upgraded in-place by newly built APKs.
- If the old key is being rotated because it leaked, note that it remains in
  git history — the new identity simply supersedes it; take down or re-sign
  old artifacts if that matters for your threat model.
