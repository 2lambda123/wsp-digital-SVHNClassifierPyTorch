from model import Model
from evaluator import Evaluator




path_to_checkpoint_file = "./model-5000.pth"
path_to_val_lmdb_dir = "../../ModelAcademy/dataset/data/train/val.lmdb"

y_true = ["1", "2", "3", "10", "10", "3"]
y_pred = ["1", "2", "2", "1", "10", "4"]

model = Model()
model.cpu()
step = model.restore(path_to_checkpoint_file)

evaluator = Evaluator(path_to_val_lmdb_dir)

test = evaluator.confusion_matrix(model)
