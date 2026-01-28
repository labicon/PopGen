#!/usr/bin/env bash
set -e

xvfb-run -s "-screen 0 1400x900x24" \
  python "$@"
