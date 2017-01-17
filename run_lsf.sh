#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Need at least one argument: ./run_lsf.sh <program.py> [<program args>*]"
  exit 1
fi

queue=priority
requests=""
if [ "$1" == "run.py" ]; then
  queue="gpu -R rusage[ngpus=1]"
  requests="-R rusage[mem=16000]"
fi
duration="24:0"
program="python -u"
fnc="$@"
if [[ $queue == gpu* ]]; then
  bsub \
    $requests \
    -W $duration \
    -q $queue \
    -o $(date +%Y-%m-%d_%H:%M:%S).out -e $(date +%Y-%m-%d_%H:%M:%S).err \
    'export THEANO_FLAGS="device=gpu`/opt/gpu_env.sh`"; '$program $fnc
else
  bsub \
    $requests \
    -W $duration \
    -q $queue \
    -o $(date +%Y-%m-%d_%H:%M:%S).out -e $(date +%Y-%m-%d_%H:%M:%S).err \
    $program $fnc
fi

