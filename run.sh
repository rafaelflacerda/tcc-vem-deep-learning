#!/bin/bash

clear

OPTIONAL_CMD=${1:-""}

if [ "$OPTIONAL_CMD" == "--clean" ]; then
    echo "===> Cleaning..."
    make clean
    make all
    echo "===> Running..."
    ./bin/polivem
    exit 0
fi

if [ "$OPTIONAL_CMD" == "--build" ]; then
    echo "===> Building..."
    make all
    echo "===> Running..."
    ./bin/polivem
    exit 0
fi

echo "===> Running..."
./bin/polivem