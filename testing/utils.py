from testing.classification_tester import ClassificationTester, SemiSupervisedTester, SemiSupervisedVAETester, PatchClassificationTester, SemiSupervisedPatchClassificationTester, PrototypicalTester


def load_tester(dataset, model, config):
    tester = ClassificationTester(dataset, model, config)
    return tester
