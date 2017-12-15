#!/bin/sh

#PBS -q swarm
#PBS -l nodes=16:ppn=28
#PBS -l walltime=2:00:00

# set up env
module load gcc/4.9.0
module load mvapich2/2.2
#module load openmpi


# Change to directory from which qsub command was issued
cd $PBS_O_WORKDIR

# Old num nodes and PPN
PPN=$PBS_NUM_PPN
NUM_NODES=$PBS_NUM_NODES
NP=$(expr $NUM_NODES \\* $PPN)

echo "Running with np = $NP, ppn = $PPN, nnodes = $NUM_NODES"

mpirun -np $NP -ppn $PPN ./bin/mxx-benchmark-p2p-bw p2p_bw_${NUM_NODES}nodes.csv
mpirun -np $NP -ppn $PPN ./bin/mxx-benchmark-a2a bm_all2all_${NUM_NODES}nodes.csv
