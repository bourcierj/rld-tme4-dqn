
hparams_cartpole = dict(

    gamma = 0.999,  #@TUNE
    epsilon = 0.01,
    epsilon_decay = 0.99999,
    batch_size = 100,
    replay_memory_capacity = 100000,
    ctarget = 1000,
    layers = [200],
    lr = 0.001,
    lr_decay = 1.,
    episode_count = 1800,
    update_frequency = 1  # NOT USED
    )


hparams_lunarlander = dict(

    gamma = 0.99,
    epsilon = 0.1,
    epsilon_decay = 0.99999,
    batch_size = 1,
    replay_memory_capacity = 1,
    ctarget = 1000,
    layers = [200],
    lr = 0.0001,
    lr_decay = 1.,
    episode_count = 11000,
    update_frequency = 1  # NOT USED
    )

hparams_gridworld = dict(

    gamma = 0.99,
    epsilon = 0.1,
    epsilon_decay = 0.9999,
    batch_size = 10,
    replay_memory_capacity = 1000000,
    ctarget = 1000,
    layers = [30, 30],
    lr = 0.0001,
    lr_decay = 1.,
    episode_count = 2500,
    update_frequency = 1  # NOT USED
    )
