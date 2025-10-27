#!/bin/bash
#SBATCH --job-name=phase2_heatsink
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/phase2_heatsink_%j.out
#SBATCH --error=logs/phase2_heatsink_%j.err
#SBATCH --mail-user=rameshkumar.b@northeastern.edu
#SBATCH --mail-type=END,FAIL

# Load modules
module load cuda/12.1.1
module load gcc/9.3.0

# Activate conda environment
source ~/.bashrc
conda activate /courses/DS5500.202610/shared/team11/pinn-project/envs/pinn-env

# Create necessary directories
mkdir -p logs models results plots data

# Generate mesh if needed
if [ ! -f "heatsink.msh" ]; then
    echo "Generating heat sink mesh..."
    python src/create_mesh.py
fi

# Create mock data if needed
if [ ! -f "data/fenics_ground_truth.npy" ]; then
    echo "Creating mock heat sink data..."
    python src/create_mock_heatsink_data.py
fi

# Run Phase 2 training
echo "Starting Phase 2 heat sink training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

PYTHONPATH=. python src/train_phase2.py

echo "Training completed at: $(date)"
echo "Job finished successfully!"