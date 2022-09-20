#!/bin/bash

export WANDBKEY=$(cat .wandbkey)
export SKIP_EXTERN=1

export PYTHONPATH=/app

exec "$@"