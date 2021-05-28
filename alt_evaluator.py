import torch
import torch.utils.data
import math
from torchvision import transforms

from .dataset import Dataset
from .alt_train import _loss


class AltEvaluator(object):
    def __init__(self, path_to_lmdb_dir, number_images_to_evaluate):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.dataset = Dataset(path_to_lmdb_dir, transform)
        if number_images_to_evaluate:
            self.dataset = self.dataset[0:int(number_images_to_evaluate)]
        self.batch_size = 1
        self._loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def evaluate(self, model):
        results = []

        with torch.no_grad():

            for batch_idx, (images, length_labels, digits_labels, paths) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)

                print("Evaluating images in batch: ", batch_idx + 1)

                # Calculate loss for batch
                loss = _loss(length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits,
                             length_labels, digits_labels)

                # This only makes sense for batch size of 1
                batch_results = {}
                for image in paths:
                    batch_results[image.decode("utf-8")] = {"loss": loss.item()}

                results.append(batch_results)

        return results

    def evaluate_least_confidence(self, model):
        results = []

        with torch.no_grad():

            for batch_idx, (images, length_labels, digits_labels, paths) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for
                                                                                             digit_labels in
                                                                                             digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(
                    images)

                print("Evaluating images in batch: ", batch_idx + 1)


                image_logits = [max(digit1_logits.tolist()[0]), max(digit2_logits.tolist()[0]), max(digit3_logits.tolist()[0]), max(digit4_logits.tolist()[0]), max(digit5_logits.tolist()[0])]
                least_confidence_for_image = min(image_logits)

                # This only makes sense for batch size of 1
                batch_results = {}
                for image in paths:
                    batch_results[image.decode("utf-8")] = {"loss": least_confidence_for_image}

                results.append(batch_results)

        return results

    def evaluate_margin_sampling(self, model):
        results = []

        with torch.no_grad():

            for batch_idx, (images, length_labels, digits_labels, paths) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for
                                                                                             digit_labels in
                                                                                             digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(
                    images)

                print("Evaluating images in batch: ", batch_idx + 1)

                image_logits = [self.margin(digit1_logits.tolist()[0]), self.margin(digit2_logits.tolist()[0]),
                                self.margin(digit3_logits.tolist()[0]), self.margin(digit4_logits.tolist()[0]),
                                self.margin(digit5_logits.tolist()[0])]
                smallest_margin = min(image_logits)


                # This only makes sense for batch size of 1
                batch_results = {}
                for image in paths:
                    batch_results[image.decode("utf-8")] = {"loss": smallest_margin}

                results.append(batch_results)

        return results

    def margin(self, logits):
        logits.sort(reverse=True)
        return logits[0] - logits[1]