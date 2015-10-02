#!/bin/sh
# source: mpi4py
# https://github.com/mpi4py/mpi4py/blob/master/conf/travis/install-mpi.sh

set -e

MPI_IMPL="$1"
os=`uname`

case "$os" in
    Linux)
        apt-get update -q
        case "$MPI_IMPL" in
            mpich2)
                apt-get install -q gfortran mpich2 libmpich2-3 libmpich2-dev
                ;;
            mpich|mpich3)
                apt-get install -q gfortran libcr0 default-jdk
                wget -q http://www.cebacad.net/files/mpich/ubuntu/mpich-3.1/mpich_3.1-1ubuntu_amd64.deb
                dpkg -i ./mpich_3.1-1ubuntu_amd64.deb
                rm -f ./mpich_3.1-1ubuntu_amd64.deb
                ;;
            openmpi)
                apt-get install -q gfortran openmpi-bin openmpi-common libopenmpi-dev
                ;;
            # TODO: openmpi 1.8 (MPI 3)
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 1
                ;;
        esac
        ;;

    *)
        echo "Unknown operating system: $os"
        exit 1
        ;;
esac
