from .model import Model
from .evaluator import Evaluator


def get_model_metrics(path_to_checkpoint_file, path_to_val_lmdb_dir):
    model = Model()
    model.cuda()
    step = model.restore(path_to_checkpoint_file)

    evaluator = Evaluator(path_to_val_lmdb_dir)

    test = evaluator.get_model_metrics(model)
    return test
