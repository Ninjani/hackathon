from pytorch_lightning.cli import LightningCLI
import torch
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Run with python main.py fit -c config.yaml
    Or in an sbatch script with srun python main.py fit -c config.yaml
    """
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(save_config_kwargs={"overwrite": True})

if __name__ == '__main__':
    main()