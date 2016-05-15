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
    error=($( { ./jacobi "$threadsPerBlock" 10 "$n" "$n" "$kernelNo" 1> "n${n}_tpb${threadsPerBlock}_k${kernelNo}.txt" ; } 2>&1 ))
    exitStatus=$?

    if (test "$exitStatus" -ne "0"); then
        echo "There was a problem running the configuration: $error" 1>&2
    fi
    IFS="$OLDIFS"
}

# --------------    n threadsPerBlock kernelNo
runConfiguration 1600               0        0
runConfiguration 3200               0        0

runConfiguration 1600              32        0
runConfiguration 1600              32        1

runConfiguration 1600             128        0
runConfiguration 1600             128        1

runConfiguration 1600             256        0
runConfiguration 1600             256        1

runConfiguration 1600             384        0
runConfiguration 1600             384        1

runConfiguration 1600             512        0
runConfiguration 1600             512        1

runConfiguration 1600            1024        0
runConfiguration 1600            1024        1

runConfiguration 3200             256        0
runConfiguration 3200             256        1

runConfiguration 3200             128        0
runConfiguration 3200             128        1