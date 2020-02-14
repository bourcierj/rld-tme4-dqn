
hparams_cartpole = dict(

    gamma = 0.999,  #@TUNE
    epsilon = 0.01,
    epsilon_decay = 0.99999,
    batch_size = 100,
    replay_memory_capacity = 100000,
    ctarget = 1000,
    layers = [200],
    lr = 0.001,
    lr_decay = 1.,  # @TUNE
    episode_count = 2400,
    update_frequency = 1  # NOT USED
    )


hparams_lunarlander = dict(


)
