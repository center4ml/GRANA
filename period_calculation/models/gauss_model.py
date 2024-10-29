import pytorch_lightning as pl
import torch
import numpy as np

from period_calculation.models.abstract_model import AbstractModel

from period_calculation.config import model_config  # this is a dictionary with the model configuration

class GaussPeriodModel(AbstractModel):
    def __init__(self,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),

            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.MaxPool2d((2, 1), stride=(2, 1)),

            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.MaxPool2d((2, 1), stride=(2, 1)),

            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.MaxPool2d((2, 1), stride=(2, 1)),

            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.MaxPool2d((2, 1), stride=(2, 1)),

            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(0, 0)),
            torch.nn.MaxPool2d((2, 1), stride=(2, 1)),

            torch.nn.Dropout(0.1)
        )
        self.query = torch.nn.Parameter(torch.empty(1, 2, 32))   #two heads only
        torch.nn.init.xavier_normal_(self.query)

        self.linear1 = torch.nn.Linear(64, 8)
        self.linear2 = torch.nn.Linear(8, 1)

        self.query_sd = torch.nn.Parameter(torch.empty(1, 2, 32))
        torch.nn.init.xavier_normal_(self.query_sd)

        self.linear_sd1 = torch.nn.Linear(64, 8)
        self.linear_sd2 = torch.nn.Linear(8, 1)
        self.relu = torch.nn.ReLU()


    def copy_network_trunk(self, model):
        # https://discuss.pytorch.org/t/copy-weights-from-only-one-layer-of-one-model-to-another-model-with-different-structure/153419
        with torch.no_grad():
            for i, layer in enumerate(model.seq):
                if i%2 == 0 and i!=20:   #convolutional layers are the ones with even indexes with the exeption of the 20th (=dropout)
                    self.seq[i].weight.copy_(layer.weight)
                    self.seq[i].bias.copy_(layer.bias)


    def copy_final_layers(self, model):
        # https://discuss.pytorch.org/t/copy-weights-from-only-one-layer-of-one-model-to-another-model-with-different-structure/153419

        with torch.no_grad():
            self.linear1.weight.copy_(model.linear1.weight)
            self.linear1.bias.copy_(model.linear1.bias)

            self.linear2.weight.copy_(model.linear2.weight)
            self.linear2.bias.copy_(model.linear2.bias)

            self.query.copy_(model.query)

    def duplicate_final_layers(self):
        # https://discuss.pytorch.org/t/copy-weights-from-only-one-layer-of-one-model-to-another-model-with-different-structure/153419

        with torch.no_grad():
            self.linear_sd1.weight.copy_(self.linear1.weight)
            self.linear_sd1.bias.copy_(self.linear1.bias)

            self.linear_sd2.weight.copy_(self.linear2.weight/10)
            self.linear_sd2.bias.copy_(self.linear2.bias/10)

            self.query_sd.copy_(self.query)

    def forward(self, x, neutral=None, return_raw=False):
        #https://www.nature.com/articles/s41598-023-43852-x

        # x is sized                            # batch x 1 x 476 x 476

        preds = self.seq(x)                        # batch x 32 x 5 x 220
        features = torch.flatten(preds, 2)         # batch x 32 x 1100

    # attention
        energy = self.query @ features                           # batch x 2 x 1100
        weights = torch.nn.functional.softmax(energy, 2)    # batch x 2 x 1100
        response = features @ weights.transpose(1, 2) # batch x 32 x 2
        response = torch.flatten(response, 1)         # batch x 64

        preds = self.linear1(response)             # batch x 8
        preds = self.linear2(self.relu(preds))     # batch x 1

    # attention sd

        energy_sd = self.query_sd @ features                         # batch x 2 x 1100
        weights_sd = torch.nn.functional.softmax(energy_sd, 2)  # batch x 2 x 1100
        response_sd = features @ weights_sd.transpose(1, 2)  # batch x 32 x 2
        response_sd = torch.flatten(response_sd, 1)  # batch x 64

        preds_sd = self.linear_sd1(response_sd)  # batch x 8
        preds_sd = self.linear_sd2(self.relu(preds_sd))  # batch x 1

        outputs = [ model_config['receptive_field_height']/(preds[:,0])  , torch.exp(preds_sd[:,0]) ]
        if return_raw:
            outputs.append(preds)
            outputs.append(preds_sd)
            outputs.append(weights)
            outputs.append(weights_sd)

        return tuple(outputs)

    def additional_losses(self):
        """get additional_losses"""
        # additional (orthogonal) loss
        # we multiply the two heads and later the MSE loss (towards zero) sums the result in L2 norm
        # the idea is that the scalar product of two orthogonal vectors is zero
        scalar_product = torch.cat((self.query[0, 0] * self.query[0, 1], self.query_sd[0, 0] * self.query_sd[0, 1]), dim=0)
        orthogonal_loss = torch.nn.functional.mse_loss(scalar_product, torch.zeros_like(scalar_product))
        return orthogonal_loss



    def process_batch_supervised(self, batch):
        """get predictions, losses and mean errors (metrics)"""

        # get predictions
        preds = {}
        preds['period_px'], preds['sd'] = self.forward(batch['image'], batch['neutral'][0], return_raw=False)  # preds: period, sd, orto, preds_raw

        # https://johaupt.github.io/blog/NN_prediction_uncertainty.html
        # calculate losses
        mse_period_px = torch.nn.functional.mse_loss(batch['period_px'],
                                                     preds['period_px'])

        gaussian_nll = torch.nn.functional.gaussian_nll_loss(batch['period_px'],
                                                             preds['period_px'],
                                                             (preds['sd']) ** 2)

        orthogonal_weight = 0.1
        orthogonal_loss = self.additional_losses()
        length_of_the_first_phase = 0
        if self.current_epoch < length_of_the_first_phase:
            #transition from MSE to Gaussian Negative Log Likelihood with sin/cos over first epochs
            angle = torch.tensor((self.current_epoch) / (length_of_the_first_phase) * np.pi / 2)
            total_loss = (gaussian_nll) * torch.sin(angle) + (mse_period_px) * torch.cos(angle) + orthogonal_weight * orthogonal_loss
        else:
            total_loss = gaussian_nll + orthogonal_weight * orthogonal_loss

        losses = {
            'gaussian_nll': gaussian_nll,
            'mse_period_px': mse_period_px,
            'orthogonal': orthogonal_loss,
            'final': total_loss
        }

        # calculate mean errors
        ground_truth_detached = batch['period_px'].detach().cpu().numpy()
        print(ground_truth_detached)
        mean_detached = preds['period_px'].detach().cpu().numpy()
        print(mean_detached)
        sd_detached = preds['sd'].detach().cpu().numpy()
        print("==>", sd_detached)
        px_per_nm_detached = batch['px_per_nm'].detach().cpu().numpy()
        scale_detached = batch['scale'].detach().cpu().numpy()

        period_px_difference = np.mean(abs(
            ground_truth_detached - mean_detached
        ))

        #initiate both with python array with 5 zeros
        true_period_px_difference = [0.0] * 5
        true_period_nm_difference = [0.0] * 5

        for i, dist in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            true_period_px_difference[i] = (np.sum(abs(
                ((ground_truth_detached - mean_detached) / scale_detached) * (sd_detached / scale_detached <dist))) \
                / np.sum(sd_detached / scale_detached < dist)) if np.sum(sd_detached / scale_detached < dist) > 0 else 0

        for i, dist in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            true_period_nm_difference[i] = (np.sum(abs(
                ((ground_truth_detached - mean_detached) / (scale_detached * px_per_nm_detached)) * (sd_detached / scale_detached <dist))) \
                / np.sum(sd_detached / scale_detached < dist)) if np.sum(sd_detached / scale_detached < dist) > 0 else 0


        true_period_px_difference_all = np.mean(abs(
            ((ground_truth_detached - mean_detached) / scale_detached)
        ))

        true_period_nm_difference_all = np.mean(abs(
            ((ground_truth_detached - mean_detached) / (scale_detached * px_per_nm_detached))
        ))

        metrics = {
            'period_px': period_px_difference,
            'true_period_px_1': true_period_px_difference[0],
            'true_period_px_2': true_period_px_difference[1],
            'true_period_px_3': true_period_px_difference[2],
            'true_period_px_4': true_period_px_difference[3],
            'true_period_px_5': true_period_px_difference[4],
            'true_period_px_all': true_period_px_difference_all,

            'true_period_nm_1': true_period_nm_difference[0],
            'true_period_nm_2': true_period_nm_difference[1],
            'true_period_nm_3': true_period_nm_difference[2],
            'true_period_nm_4': true_period_nm_difference[3],
            'true_period_nm_5': true_period_nm_difference[4],
            'true_period_nm_all': true_period_nm_difference_all,

            'count_1': np.sum(sd_detached / scale_detached < 1.0),
            'count_2': np.sum(sd_detached / scale_detached < 2.0),
            'count_3': np.sum(sd_detached / scale_detached < 3.0),
            'count_4': np.sum(sd_detached / scale_detached < 4.0),
            'count_5': np.sum(sd_detached / scale_detached < 5.0),

            'count_all': np.sum(sd_detached > 0.0),
            'mean_sd': np.mean(sd_detached),
            'sd_sd': np.std(sd_detached),
        }

        return preds, losses, metrics


