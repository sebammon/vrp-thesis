from pathlib import Path

import torch

MODELS_DIR = Path(__file__).parent.parent / 'models'


def save_model(model, filename='model.pt', overwrite=False):
    """
    Saves model to disk.
    """
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir()

    if (MODELS_DIR / filename).exists():
        if not overwrite:
            raise FileExistsError(f"File {filename} already exists")

    torch.save(model.state_dict(), MODELS_DIR / filename)


def load_model(filename):
    """
    Loads model from disk.
    """
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Directory {MODELS_DIR} does not exist")

    if not (MODELS_DIR / filename).exists():
        raise FileNotFoundError(f"File {filename} does not exist")

    model = torch.load(MODELS_DIR / filename)

    return model
