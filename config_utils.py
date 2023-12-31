from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "n_epochs": 60, #must be beigger than preload if using preload
        "lr": 1e-4,
        "seq_len": 250,
        "d_model": 256, #512 in the paper
        "n_blocks": 6, #6 in the paper
        "n_heads": 8, #8 in the paper
        "d_ff": 1024, #2048 in the paper
        "dropout_rate": 0.05,
        #"datasource": 'indonlp/NusaX-MT', #dataset is too small 
        #"datasource": 'opus_books',
        "datasource": 'opus_euconst',
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tfr_model_",
        "preload": None, #change this to latest epoch number if you want to continue from previous training session
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

# find the path where we will save the weights
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

