from testing.classification_tester import ClassificationTester

def load_tester(dataset, model, config):
    tester = ClassificationTester(dataset, model, config)
    return tester
