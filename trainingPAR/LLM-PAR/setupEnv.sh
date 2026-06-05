
# 1. Descargar e instalar Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source "$HOME/miniconda/bin/activate"
conda init

# 2. Reiniciar shell (o ejecutar source ~/.bashrc)
source ~/.bashrc

# 3. Crear tu entorno para LLM-PAR
conda env create --file environment.yaml

conda activate llmpar
python -c "import torch; print(f'GPU disponible: {torch.cuda.is_available()}'); print(f'Dispositivos: {torch.cuda.device_count()}')"