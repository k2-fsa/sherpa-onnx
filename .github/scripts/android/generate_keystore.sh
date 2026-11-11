#!/usr/bin/env bash
# Generate the CI APK signing keystore (see README.md in this directory).
#
# The keystore is NOT committed — it lives as the ANDROID_KEYSTORE_B64 GitHub
# Actions secret. Run this only to create a fresh identity (e.g. after a
# compromise); every rotation makes installed apps require a reinstall.
#
# Usage: generate_keystore.sh [output-path]

set -e

cd "$(dirname "$0")"
out=${1:-ci-apk.keystore}

if [ -f "$out" ]; then
  echo "$out already exists — refusing to overwrite." >&2
  echo "Rotating the key changes the app's signing identity; delete the file" >&2
  echo "yourself if that is really what you want." >&2
  exit 1
fi

keytool -genkeypair \
  -keystore "$out" \
  -storepass android -key-pass android \
  -alias ci -keyalg RSA -keysize 2048 -validity 10000 \
  -dname "CN=sherpa-onnx CI APK, OU=sherpa-onnx, O=k2-fsa, C=CN"

echo
echo "Generated $out:"
keytool -list -v -keystore "$out" -storepass android -alias ci 2>/dev/null |
  grep -E "Alias name|Valid from|SHA256"

echo
echo "Store it as the Actions secret (see README.md):"
echo "  base64 -i $out | pbcopy   # then paste as ANDROID_KEYSTORE_B64"
echo "  base64 -i $out | gh secret set ANDROID_KEYSTORE_B64 --repo k2-fsa/sherpa-onnx"
