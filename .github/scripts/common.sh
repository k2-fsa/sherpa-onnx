#!/usr/bin/env bash
# Common utilities for CI scripts
# Provides robust download functions with retry logic, validation, and automatic cleanup

# Default log function (can be overridden by scripts that define their own)
# Logs messages with timestamp, source file, line number, and function name
log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# --- Extracted-model cleanup ---------------------------------------------
# Test scripts download large ASR models. Each script normally ends with an
# `rm -rf $repo`, but if the script aborts early (e.g. `set -e` when a test
# fails, or a download/extract error) that line never runs and the model is
# left on disk. Across the many test steps in a single CI job this fills the
# runner's disk. To avoid that, download_and_extract registers every extracted
# top-level path here and they are all removed when the script exits, whether
# it succeeded or failed.
_sherpa_cleanup_dirs=()
_sherpa_cleanup_saved_exit_trap=""
_sherpa_cleanup_installed=0

_sherpa_cleanup_on_exit() {
  local d
  for d in ${_sherpa_cleanup_dirs[@]+"${_sherpa_cleanup_dirs[@]}"}; do
    if [ -n "$d" ] && [ -e "$d" ]; then
      log "Cleaning up extracted model: $d"
      rm -rf "$d"
    fi
  done
  # Run any EXIT trap that was installed before ours (e.g. test-rust.sh's
  # diagnostic trap) so we never clobber another script's cleanup.
  if [ -n "$_sherpa_cleanup_saved_exit_trap" ]; then
    eval "$_sherpa_cleanup_saved_exit_trap"
  fi
}

_sherpa_register_cleanup() {
  _sherpa_cleanup_dirs+=("$1")
  if [ "$_sherpa_cleanup_installed" -eq 0 ]; then
    _sherpa_cleanup_installed=1
    # Save whatever EXIT trap is currently installed so ours can chain to it.
    local saved
    saved="$(trap -p EXIT)"
    if [ -n "$saved" ]; then
      # saved looks like: trap -- 'some command' EXIT
      _sherpa_cleanup_saved_exit_trap="${saved#trap -- \'}"
      _sherpa_cleanup_saved_exit_trap="${_sherpa_cleanup_saved_exit_trap%\' EXIT}"
    fi
    trap _sherpa_cleanup_on_exit EXIT
  fi
}

# Download a file with retry logic and validation
# Validates that downloaded files are not empty and not HTML error pages
# Retries with random sleep (1-10 seconds) between attempts
# Usage: download <url> [output_filename] [max_retries]
# Arguments:
#   url - The URL to download from
#   output_filename - Optional. The local filename to save as (default: basename of URL)
#   max_retries - Optional. Maximum number of retry attempts (default: 5)
# Returns: 0 on success, 1 on failure
download() {
  local url="$1"
  local output="${2:-$(basename "$url")}"
  local max_retries="${3:-5}"

  for ((i = 1; i <= max_retries; i++)); do
    log "Download attempt $i/$max_retries: $url"
    if curl -SL --fail -o "$output" "$url"; then
      # Check for empty file
      if [ ! -s "$output" ]; then
        log "Warning: Downloaded file is empty"
        if [ "$i" -lt "$max_retries" ]; then
          local sleep_seconds=$(( RANDOM % 10 + 1 ))
          log "Retrying in $sleep_seconds seconds..."
          sleep "$sleep_seconds"
          continue
        else
          log "Error: Failed to download non-empty file after $max_retries attempts"
          return 1
        fi
      fi

      # Verify the file is not an HTML error page
      # Check for HTML tags in the first few bytes, but allow legitimate text files
      if head -c 1000 "$output" 2>/dev/null | grep -qi "<!doctype\|<html\|<head\|<body"; then
        log "Warning: Downloaded file appears to be an HTML error page"
        if [ "$i" -lt "$max_retries" ]; then
          local sleep_seconds=$(( RANDOM % 10 + 1 ))
          log "Retrying in $sleep_seconds seconds..."
          sleep "$sleep_seconds"
          continue
        else
          log "Error: Failed to download valid file after $max_retries attempts"
          return 1
        fi
      fi
      log "Successfully downloaded: $output"
      return 0
    fi

    if [ "$i" -lt "$max_retries" ]; then
      local sleep_seconds=$(( RANDOM % 10 + 1 ))
      log "Download failed, retrying in $sleep_seconds seconds..."
      sleep "$sleep_seconds"
    fi
  done

  log "Error: Failed to download $url after $max_retries attempts"
  return 1
}

# Download and extract an archive file with retry logic and validation
# Supports multiple archive formats: .tar.bz2, .tar.gz, .tar.xz, .zip
# Automatically removes the archive file after successful extraction
# Validates downloads are not empty and not HTML error pages
# Retries with random sleep (1-10 seconds) between attempts
# Usage: download_and_extract <url> [output_dir] [max_retries]
# Arguments:
#   url - The URL of the archive to download
#   output_dir - Optional. Directory to extract to (default: current directory)
#   max_retries - Optional. Maximum number of retry attempts (default: 5)
# Returns: 0 on success, 1 on failure
download_and_extract() {
  local url="$1"
  local output_dir="${2:-.}"
  local max_retries="${3:-5}"
  local filename=$(basename "$url")

  for ((i = 1; i <= max_retries; i++)); do
    log "Download and extract attempt $i/$max_retries: $url"

    # Download
    if ! download "$url" "$filename" 1; then
      if [ "$i" -lt "$max_retries" ]; then
        local sleep_seconds=$(( RANDOM % 10 + 1 ))
        log "Retrying in $sleep_seconds seconds..."
        sleep "$sleep_seconds"
        continue
      fi
      return 1
    fi

    # Extract based on file extension
    # Compute the top-level name the archive extracts to (release model
    # archives are named after the directory they contain) so we can clean it
    # up automatically when the script exits.
    local extracted="$filename"
    extracted="${extracted%.tar.bz2}"; extracted="${extracted%.tar.bz}"
    extracted="${extracted%.tbz2}"; extracted="${extracted%.tbz}"
    extracted="${extracted%.tar.gz}"; extracted="${extracted%.tgz}"
    extracted="${extracted%.tar.xz}"; extracted="${extracted%.txz}"
    extracted="${extracted%.zip}"
    case "$filename" in
      *.tar.bz2|*.tar.bz|*.tbz2|*.tbz)
        if tar jxvf "$filename" -C "$output_dir" 2>/dev/null; then
          log "Successfully extracted: $filename"
          rm -f "$filename"
          _sherpa_register_cleanup "$output_dir/$extracted"
          return 0
        fi
        ;;
      *.tar.gz|*.tgz)
        if tar zxvf "$filename" -C "$output_dir" 2>/dev/null; then
          log "Successfully extracted: $filename"
          rm -f "$filename"
          _sherpa_register_cleanup "$output_dir/$extracted"
          return 0
        fi
        ;;
      *.tar.xz|*.txz)
        if tar Jxvf "$filename" -C "$output_dir" 2>/dev/null; then
          log "Successfully extracted: $filename"
          rm -f "$filename"
          _sherpa_register_cleanup "$output_dir/$extracted"
          return 0
        fi
        ;;
      *.zip)
        if unzip -q "$filename" -d "$output_dir" 2>/dev/null; then
          log "Successfully extracted: $filename"
          rm -f "$filename"
          _sherpa_register_cleanup "$output_dir/$extracted"
          return 0
        fi
        ;;
      *)
        log "Unknown archive format: $filename"
        return 1
        ;;
    esac

    log "Extraction failed, file may be corrupted"
    rm -f "$filename"

    if [ "$i" -lt "$max_retries" ]; then
      local sleep_seconds=$(( RANDOM % 10 + 1 ))
      log "Retrying in $sleep_seconds seconds..."
      sleep "$sleep_seconds"
    fi
  done

  log "Error: Failed to download and extract $url after $max_retries attempts"
  return 1
}
