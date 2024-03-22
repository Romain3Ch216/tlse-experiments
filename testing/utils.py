from testing.classification_tester import AETester, MAETester

def load_tester(dataset, model, config):
    if config['model'] == 'AE':
        tester = AETester(dataset, model, config)
    elif config['model'] == 'MAE':
        tester = MAETester(dataset, model, config)
    return tester
