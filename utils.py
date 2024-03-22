def load_config(config):
    if config['model'] in ['MAE', 'AE']:
        config.update({
            'pred_mode': 'pixel',
            'patch_size': 1
        })

    return config


def update_config(config, study):
    for k, v in study.best_params.items():
        config[k] = v
    config['seed'] = study.best_trial.user_attrs['seed']
    config['save_best_model'] = True
    return config
