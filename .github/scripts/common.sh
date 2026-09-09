#!/usr/bin/env bash
# Common utilities for CI scripts

# Default log function (can be overridden by scripts that define their own)
log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Download a file with retry logic
# Usage: download <url> [output_filename] [max_retries]
download() {
  local url="$1"
  local output="${2:-$(basename "$url")}"
  local max_retries="${3:-5}"

  for ((i = 1; i <= max_retries; i++)); do
    log "Download attempt $i/$max_retries: $url"
    if curl -SL --fail -o "$output" "$url"; then
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

# Download and extract a tarball with retry logic
# Usage: download_and_extract <url> [output_dir] [max_retries]
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
    case "$filename" in
      *.tar.bz2|*.tar.bz|*.tbz2|*.tbz)
        if tar jxvf "$filename" -C "$output_dir" 2>/dev/null; then
          log "Successfully extracted: $filename"
          rm -f "$filename"
          return 0
        fi
        ;;
      *.tar.gz|*.tgz)
        if tar zxvf "$filename" -C "$output_dir" 2>/dev/null; then
          log "Successfully extracted: $filename"
          rm -f "$filename"
          return 0
        fi
        ;;
      *.tar.xz|*.txz)
        if tar Jxvf "$filename" -C "$output_dir" 2>/dev/null; then
          log "Successfully extracted: $filename"
          rm -f "$filename"
          return 0
        fi
        ;;
      *.zip)
        if unzip -q "$filename" -d "$output_dir" 2>/dev/null; then
          log "Successfully extracted: $filename"
          rm -f "$filename"
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
