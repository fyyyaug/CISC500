eval "$(/opt/conda/bin/conda shell.bash hook)" 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh

conda activate /home/jovyan/conda-envs/clinica

which python
python -V
clinicadl --version