#!/usr/bin/env bash
# Prints where cramera_command.sh resolves the console script named in $1, for the tests
# of that helper. $REPO and $PATH are set by the caller.
set -e
source "$REPO/cramera_command.sh"
cramera_command "$1"
