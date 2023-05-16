from pytorch_lightning.cli import LightningCLI
import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def main():
    """
    Run with python main.py fit -c config.yaml
    Or in an sbatch script with srun python main.py fit -c config.yaml
    """
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI()

if __name__ == '__main__':
    main()