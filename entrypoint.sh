#!/bin/bash

export WANDBKEY=$(cat .wandbkey)
export SKIP_EXTERN=1

pip install -v .

exec "$@"