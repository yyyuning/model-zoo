#!/bin/bash

set -eE
trap 'echo "ERROR: $BASH_SOURCE:$LINENO $BASH_COMMAND" >&2' ERR

function quiet_exec()
{
    log=/tmp/log.txt
    echo $@
    $@ 2>&1 | cat > $log
    test ${PIPESTATUS[0]} -eq 0 || {
        cat $log
        echo task failed
        echo $@
        return 1
    }
}

quiet_exec pip3 install -r test/requirements.txt
python3 -m pytest test
