#!/bin/sh
# source: mpi4py
# https://github.com/mpi4py/mpi4py/blob/master/conf/travis/install-mpi.sh

set -e

MPI_IMPL="$1"
os=`uname`

case "$os" in
    Linux)
        #sudo apt-get update -q
        case "$MPI_IMPL" in
            mpich2)
                if [ ! -d "$HOME/local/$MPI_IMPL/bin" ]; then
                    wget http://www.mpich.org/static/downloads/1.5/mpich2-1.5.tar.gz
                    tar -xzf mpich2-1.5.tar.gz
                    cd mpich2-1.5
                    ./configure --prefix=$HOME/local/$MPI_IMPL --disable-fc --disable-f77 && make && make install
                    cd ../../
                else
                    echo 'Using cached MPICH2 v 1.5 directory';
                fi
            #    sudo apt-get install -q gfortran mpich2 libmpich2-3 libmpich2-dev
                ;;
            mpich3)
                #sudo apt-get install -q gfortran libcr0 default-jdk
                #wget -q http://www.cebacad.net/files/mpich/ubuntu/mpich-3.1/mpich_3.1-1ubuntu_amd64.deb
                #sudo dpkg -i ./mpich_3.1-1ubuntu_amd64.deb
                #rm -f ./mpich_3.1-1ubuntu_amd64.deb
                if [ ! -d "$HOME/local/$MPI_IMPL/bin" ]; then
                    wget http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz
                    tar -xzf mpich-3.1.4.tar.gz
                    cd mpich-3.1.4
                    ./configure --prefix=$HOME/local/$MPI_IMPL --disable-fortran && make && make install
                    cd ../../
                else
                    echo 'Using cached MPICH 3.1.4 directory';
                fi
                ;;
            openmpi16)
                #sudo apt-get install -q gfortran openmpi-bin openmpi-common libopenmpi-dev
                if [ ! -d "$HOME/local/$MPI_IMPL/bin" ]; then
                    wget --no-check-certificate http://www.open-mpi.org/software/ompi/v1.6/downloads/openmpi-1.6.5.tar.bz2
                    tar -xjf openmpi-1.6.5.tar.bz2
                    cd openmpi-1.6.5
                    ./configure --prefix=$HOME/local/$MPI_IMPL && make && make install
                    cd ../../
                else
                    echo 'Using cached OpenMPI 1.6.5 directory';
                fi
                ;;
            openmpi18)
                if [ ! -d "$HOME/local/$MPI_IMPL/bin" ]; then
                    mkdir -p openmpi && cd openmpi
                    wget --no-check-certificate http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.8.tar.bz2
                    tar -xjf openmpi-1.8.8.tar.bz2
                    cd openmpi-1.8.8
                    ./configure --prefix=$HOME/local/$MPI_IMPL && make && make install
                    cd ../../
                    echo 'Using cached OpenMPI 1.8.8 directory';
                fi
                ;;
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
