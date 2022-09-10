#!/bin/bash

export WANDBKEY=$(cat .wandbkey)

exec "$@"