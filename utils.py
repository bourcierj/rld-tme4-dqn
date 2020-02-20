

def get_hyperparams_dict(args, ignore=set()):

    hparams = dict()
    for key, value in vars(args).items():
        key = key.replace('_','-')
        if key not in ignore:
            hparams[key] = value
    return hparams


def get_experiment_name(prefix, hparams):
    """Generate a string name for the experiment.
    It is intended to be used for saving experiment information to files.
    The name follows the syntax: runs/CURRENT-DATETIME_HOSTNAME_HYPERPARAMETERS
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    hparams_string = '_'.join([f"{key}={value}" for key, value in hparams.items()])

    return current_time + '_' + socket.gethostname() + prefix + hparams_string

