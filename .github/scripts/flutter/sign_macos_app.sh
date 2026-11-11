#!/usr/bin/env bash
# Ad-hoc sign a macOS .app bundle so it can be launched on machines other
# than the build machine.
#
# The ffmpeg-based demos are built with CODE_SIGNING_ALLOWED=NO, so every
# nested framework/dylib is unsigned and dyld refuses to load it:
#
#   Library not loaded: @rpath/Ass.framework/Versions/A/Ass
#   Reason: ... Trying to load an unsigned library
#
# Sign nested code inside-out, then the app itself, and verify the result.

set -e
set -o pipefail

if [ $# != 1 ]; then
  echo "Usage: $0 /path/to/xxx.app"
  exit 1
fi

app=$1

if [ ! -d "$app" ]; then
  echo "$app does not exist"
  exit 1
fi

# Detritus (Finder info, resource forks) makes codesign fail with
# "resource fork, Finder information, or similar detritus not allowed"
xattr -cr "$app"

# Sign nested code items, innermost first: a deeper path has more slashes.
# codesign --deep is not used since it is deprecated and it does not handle
# the ffmpeg frameworks reliably.
find "$app/Contents" \( -type d \( -name '*.framework' -o -name '*.bundle' \) -o -type f -name '*.dylib' \) -print |
  awk '{ n = gsub(/\//, "/"); print n "\t" $0 }' |
  sort -rn |
  cut -f2- |
  while IFS= read -r item; do
    echo "signing $item"
    codesign --force --sign - --timestamp=none "$item"
  done

echo "signing $app"
codesign --force --sign - --timestamp=none "$app"

echo "verifying $app"
codesign --verify --deep --strict "$app"

echo "$app is signed and valid"
