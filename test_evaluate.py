from model import Model
from .evaluator import Evaluator


def test_model_metrics():
    path_to_checkpoint_file = "./model-5000.pth"
    path_to_val_lmdb_dir = "../../ModelAcademy/dataset/data/train/val.lmdb"

    model = Model()
    model.cpu()
    step = model.restore(path_to_checkpoint_file)

    evaluator = Evaluator(path_to_val_lmdb_dir)

    test = evaluator.get_model_metrics(model)
    return test


print(test_model_metrics())
