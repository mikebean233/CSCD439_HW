#! /bin/bash

# arg1: n
# arg2: threadsPerBlock
# arg3: kernelNo
function runConfiguration {
    OLDIFS="$IFS"
    IFS="~"
    n=$1
    threadsPerBlock=$2
    kernelNo=$3

    echo "--------- executing configuration: n=$n, threadsPerBlock=$threadsPerBlock, kernelNo=$kernelNo ---------------"
    echo ""
    error=($( { ./reduce "$threadsPerBlock" "$n" "$kernelNo" 1> "n${n}_tpb${threadsPerBlock}_k${kernelNo}.txt" ; } 2>&1 ))
    exitStatus=$?

    if (test "$exitStatus" -ne "0"); then
        echo "There was a problem running the configuration: $error" 1>&2
    fi
    IFS="$OLDIFS"
}

# -------------- n          threadsPerBlock kernelNo
runConfiguration 1048576    1024            2
runConfiguration 1048576    1024            3

runConfiguration 16777216   1024            2
runConfiguration 16777216   1024            3

runConfiguration 67108864   1024            2
runConfiguration 67108864   1024            3

runConfiguration 134217728  1024            2
runConfiguration 134217728  1024            3
