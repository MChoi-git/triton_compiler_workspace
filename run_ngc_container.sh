#!/bin/bash

# Activate singularity module
module purge
module load singularity-ce/3.8.2

# Run singularity container
singularity run \
	--nv \
	--home /scratch/ssd004/scratch/mchoi/triton_compiler_workspace/workspace \
	--env "PYTHONPATH=/scratch/ssd004/scratch/mchoi/triton_compiler_workspace/workspace/triton/python" \
	/scratch/ssd004/scratch/mchoi/pytorch_23.03-py3.sif
