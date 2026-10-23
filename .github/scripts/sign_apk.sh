#!/usr/bin/env bash
# Ad-hoc sign an Android APK with the shared CI keystore.
#
# Why not the Gradle debug signature: every CI job generates a fresh
# ~/.android/debug.keystore, so each build of the same app has a different
# signer. Installing a new build over an old one then fails on many OEM
# installers (e.g. MIUI's "packageInfo is null"). This script signs every APK
# with the shared keystore (ANDROID_KEYSTORE_B64 secret) instead, so all
# builds of an app share one identity and upgrades install cleanly. It also
# forces both v1 and v2 signature schemes for older installers (AGP ships
# v2-only when minSdk >= 24).
#
# Usage: sign_apk.sh <input.apk> <output.apk>

# pipefail: a failing apksigner must not be hidden by a successful tail.
set -eo pipefail

if [ $# != 2 ]; then
  echo "Usage: $0 <input.apk> <output.apk>"
  exit 1
fi

in_apk=$1
out_apk=$2

if [ ! -f "$in_apk" ]; then
  echo "$in_apk does not exist"
  exit 1
fi

: "${ANDROID_HOME:?ANDROID_HOME is not set}"

build_tools=$(ls -d "$ANDROID_HOME"/build-tools/* | sort -V | tail -1)
echo "using build-tools: $build_tools"

# The keystore is the base64 of a private key — it lives in the
# ANDROID_KEYSTORE_B64 Actions secret, never in the repo (see
# android/README.md). For manual local runs the gitignored file is used.
keystore=""
if [ -n "${ANDROID_KEYSTORE_B64:-}" ]; then
  keystore="${RUNNER_TEMP:-/tmp}/ci-apk.keystore"
  printf '%s' "$ANDROID_KEYSTORE_B64" | base64 --decode > "$keystore"
  echo "using keystore from ANDROID_KEYSTORE_B64"
elif [ -f "$(dirname "$0")/android/ci-apk.keystore" ]; then
  keystore="$(dirname "$0")/android/ci-apk.keystore"
  echo "using local keystore $keystore"
else
  echo "No signing keystore: set the ANDROID_KEYSTORE_B64 secret" >&2
  echo "(see .github/scripts/android/README.md)" >&2
  exit 1
fi

aligned=$(mktemp -t aligned-XXXXXX).apk
"$build_tools/zipalign" -P 16 -f 4 "$in_apk" "$aligned"
"$build_tools/apksigner" sign \
  --ks "$keystore" \
  --ks-pass pass:android \
  --key-pass pass:android \
  --ks-key-alias ci \
  --v1-signing-enabled true \
  --v2-signing-enabled true \
  --out "$out_apk" "$aligned"
rm -f "$aligned"

"$build_tools/apksigner" verify --verbose "$out_apk" | tail -8
ls -lh "$out_apk"
