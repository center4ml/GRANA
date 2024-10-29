import numpy as np
import torch
import pytorch_lightning as pl
import timm

# from hydra.utils import instantiate
from scipy.stats import circmean, circstd
from scipy import ndimage
from skimage.transform import resize

from sampling import get_crop_batch
from granum_utils import get_circle_mask
import image_transforms
from envelope_correction import calculate_best_angle_from_mask
## loss

class ConfidenceScaler:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.data.sort()
    def __call__(self, x):
        return np.searchsorted(self.data,x) / len(self.data)

class PatchedPredictor:
    def __init__(self,
                 model,
                 crop_size=96,
                 normalization=dict(mean=0,std=1),
                 n_samples=32,
                 mask=None,# 'circle', None
                 filter_outliers=True,
                 apply_radon=False, # apply Radon transform
                 radon_size=(128,128), # (int, int) reshape radon transformed image to this shape,
                 angle_confidence_threshold=0,
                 use_envelope_correction=True
                ):
        self.model = model
        self.crop_size = crop_size
        self.normalization = normalization
        self.n_samples = n_samples
        if mask not in [None, 'circle']:
            raise ValueError(f'unknown mask {mask}')
        self.mask = mask
        self.filter_outliers = filter_outliers
        
        self.apply_radon = apply_radon
        self.radon_size = radon_size
        
        self.angle_confidence_threshold = angle_confidence_threshold
        self.use_envelope_correction = use_envelope_correction
        
    @torch.no_grad()
    def __call__(self, img: np.ndarray, mask: np.ndarray):
        pl.seed_everything(44)
        # get crops with different scales and rotation
        crops, angles_tta, scales_tta = get_crop_batch(
            img, mask,
            crop_size=self.crop_size,
            samples_per_scale=self.n_samples,
            use_variance_threshold=True
        )
        if len(crops) == 0:
            return dict(
                est_angle=np.nan,
                est_angle_confidence=0.,
            )
            
        # preprocess batch (normalize, mask, transform)
        batch = self._preprocess_batch(crops)           
        
        # predict for batch - we don't use period and lumen anymore
        preds_direction, preds_period, preds_lumen_width = self.model(batch)
        # # convert to numpy
        # preds_direction = preds_direction.numpy()
        # preds_period = preds_period.numpy()
        # preds_lumen_width = preds_lumen_width.numpy()
        
        # aggregate angles
        est_angles = (preds_direction - angles_tta) % 180
        est_angle = circmean(est_angles, low=-90, high=90) + 90
        est_angle_std = circstd(est_angles, low=-90, high=90)
        est_angle_confidence = self._std_to_confidence(est_angle_std, 10) # confidence 0.5 for std =10 degrees        
        
        if est_angle_confidence < self.angle_confidence_threshold:
            est_angle = np.nan
            est_angle_confidence = 0.
        
        if self.use_envelope_correction and (not np.isnan(est_angle)):           
            angle_correction = -calculate_best_angle_from_mask(
                ndimage.rotate(mask, -est_angle, reshape=True, order=0)
            )            
            est_angle += angle_correction
            
        return dict(
            est_angle=est_angle,
            est_angle_confidence=est_angle_confidence,
        )
        
    def _apply_radon(self, batch): # may reauire circle mask
        crops_radon = image_transforms.batched_radon(batch.numpy())
        crops_radon = np.transpose(resize(np.transpose(crops_radon, (1, 2, 0)), self.radon_size), (2, 0, 1))
        return torch.tensor(crops_radon)
    
    def _preprocess_batch(self, batch):
        if self.mask == 'circle':
            mask = get_circle_mask(batch.shape[1])
            batch[:,mask] = 0
        if self.apply_radon:
            batch = self._apply_radon(batch)
        batch = ((batch/255) - self.normalization['mean'])/self.normalization['std']
        return batch.unsqueeze(1) # add channel dimension

    def _filter_outliers(self, x, qmin=0.25, qmax=0.75):
        x_min, x_max = np.quantile(x, [qmin, qmax])
        return x[(x>=x_min) & (x<=x_max)]
    
    def _std_to_confidence(self, x, x_thr, y_thr=0.5):
        """transform [0, inf] to [1,0], such that f(x_thr)=y_thr"""
        return 1 / (1+x*(1-y_thr)/(x_thr*y_thr))

class CosineLoss(torch.nn.Module):
  def __init__(self, p=1, degrees=False, scale=1):
    super().__init__()
    self.p = p
    self.degrees = degrees
    self.scale = scale
  def forward(self, x, y):
    if self.degrees:
      x = torch.deg2rad(x)
      y = torch.deg2rad(y)
    return torch.mean((1-torch.cos(x-y))**self.p) * self.scale
    
## model
class AngleParser2d(torch.nn.Module):
    def __init__(self, angle_range=180):
        super().__init__()
        self.angle_range = angle_range
    def forward(self, batch):
        # r = torch.linalg.norm(batch, dim=1)
        preds_y_proj = torch.sigmoid(batch[:,0]) - 0.5
        preds_x_proj = torch.sigmoid(batch[:,1]) - 0.5
        preds_direction = self.angle_range/360.*torch.rad2deg(torch.arctan2(preds_y_proj, preds_x_proj))
        return preds_direction

class AngleRegularizer(torch.nn.Module):
  def __init__(self, strength=1.0, scale=1.0, p=2):
    super().__init__()
    self.strength = strength
    self.scale = scale
    self.p = p
  def forward(self, batch):
    r = torch.linalg.norm(batch, dim=1)
    return self.strength * torch.norm(r - self.scale, p=self.p)

class AngleRegularizerLog(torch.nn.Module):
  def __init__(self, strength=1.0, scale=1.0, p=2):
    super().__init__()
    self.strength = strength
    self.scale = scale
    self.p = p
  def forward(self, batch):
    r = torch.linalg.norm(batch, dim=1)
    return self.strength * torch.norm(torch.log(r/self.scale), p=self.p)

class StripsModel(pl.LightningModule):
  def __init__(self, 
              model_name = 'resnet18',
               lr=0.001,
               optimizer_hparams=dict(),
               lr_hparams=dict(classname='MultiStepLR', kwargs=dict(milestones=[100, 150], gamma=0.1)),
               loss_hparams=dict(rotation_weight=10., lumen_fraction_weight=50.),
               angle_hparams=dict(angle_range=180.),
               regularizer_hparams=None,
               sigmoid_smoother=10.
               ):
    super().__init__()
    # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
    self.save_hyperparameters()
    # Create model - implemented in non-abstract classes
    self.model =  timm.create_model(model_name, in_chans=1, num_classes=4) #2 + self.hparams.angle_hparams['ndim'])
    self.angle_parser = AngleParser2d(**self.hparams.angle_hparams)
    self.regularizer = self._get_regularizer(self.hparams.regularizer_hparams)
    self.losses = {
        'direction': CosineLoss(2., True),
        'period': torch.nn.functional.mse_loss,
        'lumen_fraction': torch.nn.functional.mse_loss
    }
    self.losses_weights = {
      'direction': self.hparams.loss_hparams['rotation_weight'],
      'period': 1,
      'lumen_fraction': self.hparams.loss_hparams['lumen_fraction_weight'],
      'regularization': self.hparams.loss_hparams.get('regularization_weight', 0.)
    }
  
  def _get_regularizer(self, regularizer_params):
    if regularizer_params is None:
      return None
    else:
      return instantiate(regularizer_params)
    

  def forward(self, x, return_raw=False):
    """get predictions from image batch"""
    preds = self.model(x) # preds: logit angle_sin, logit angle_cos, period, logit lumen fraction or logit angle, period, logit lumen fraction
    preds_direction = self.angle_parser(preds)
    preds_period = preds[:,-2]
    preds_lumen_fraction = torch.sigmoid(preds[:,-1]*self.hparams.sigmoid_smoother) #lumen fraction is between 0 and 1, so we take sigmoid fo this

    outputs = [preds_direction, preds_period, preds_lumen_fraction]
    if return_raw:
      outputs.append(preds)
      
    return tuple(outputs)

  def configure_optimizers(self):
    # AdamW is Adam with a correct implementation of weight decay (see here
    # for details: https://arxiv.org/pdf/1711.05101.pdf)    
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, **self.hparams.optimizer_hparams)
    # scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_hparams['classname'])(optimizer, **self.hparams.lr_hparams['kwargs'])
    scheduler = instantiate({**self.hparams.lr_hparams, '_partial_': True})(optimizer)
    return [optimizer], [scheduler]

  def process_batch_supervised(self, batch):
    """get predictions, losses and mean errors (MAE)"""

    # get predictions
    preds = {}
    preds['direction'], preds['period'], preds['lumen_fraction'], preds_raw = self.forward(batch['image'], return_raw=True) # preds: angle, period, lumen fraction, raw preds

    # calculate losses
    losses = {
        'direction': self.losses['direction'](2*batch['direction'], 2*preds['direction']),
        'period': self.losses['period'](batch['period'], preds['period']),
        'lumen_fraction': self.losses['lumen_fraction'](batch['lumen_fraction'], preds['lumen_fraction']),
    }
    if self.regularizer is not None:
      losses['regularization'] = self.regularizer(preds_raw[:,:2])
    
    losses['final'] = \
      losses['direction']*self.losses_weights['direction'] + \
      losses['period']*self.losses_weights['period'] + \
      losses['lumen_fraction']*self.losses_weights['lumen_fraction'] + \
      losses.get('regularization', 0.)*self.losses_weights.get('regularization', 0.)

    # calculate mean errors
    period_difference = np.mean(abs(
      batch['period'].detach().cpu().numpy() - \
      preds['period'].detach().cpu().numpy()
    ))

    a1 = batch['direction'].detach().cpu().numpy()
    a2 = preds['direction'].detach().cpu().numpy()
    angle_difference = np.mean(0.5*np.degrees(np.arccos(np.cos(2*np.radians(a2-a1)))))

    lumen_fraction_difference = np.mean(abs(preds['lumen_fraction'].detach().cpu().numpy()-batch['lumen_fraction'].detach().cpu().numpy()))

    mae = {
      'period': period_difference,
      'direction': angle_difference,
      'lumen_fraction': lumen_fraction_difference
    }

    return preds, losses, mae

  def log_all(self, losses, mae, prefix=''):
    self.log(f"{prefix}angle_loss", losses['direction'].item())
    self.log(f"{prefix}period_loss", losses['period'].item())
    self.log(f"{prefix}lumen_fraction_loss", losses['lumen_fraction'].item())
    self.log(f"{prefix}period_difference", mae['period'])
    self.log(f"{prefix}angle_difference", mae['direction'])
    self.log(f"{prefix}lumen_fraction_difference", mae['lumen_fraction'])
    self.log(f"{prefix}loss", losses['final'])
    if 'regularization' in losses:
      self.log(f"{prefix}regularization_loss", losses['regularization'].item())
  
  def training_step(self, batch, batch_idx):
    # "batch" is the output of the training data loader.
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='train_')

    return losses['final'] 
  
  def validation_step(self, batch, batch_idx):
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='val_')
  
  def test_step(self, batch, batch_idx):
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='test_')

    
class StripsModelLumenWidth(pl.LightningModule):
  def __init__(self, 
              model_name = 'resnet18',
               lr=0.001,
               optimizer_hparams=dict(),
               lr_hparams=dict(classname='MultiStepLR', kwargs=dict(milestones=[100, 150], gamma=0.1)),
               loss_hparams=dict(rotation_weight=10., lumen_width_weight=50.),
               angle_hparams=dict(angle_range=180.),
               regularizer_hparams=None,
               sigmoid_smoother=10.
               ):
    super().__init__()
    # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
    self.save_hyperparameters()
    # Create model - implemented in non-abstract classes
    self.model =  timm.create_model(model_name, in_chans=1, num_classes=4) #2 + self.hparams.angle_hparams['ndim'])
    self.angle_parser = AngleParser2d(**self.hparams.angle_hparams)
    self.regularizer = self._get_regularizer(self.hparams.regularizer_hparams)
    self.losses = {
        'direction': CosineLoss(2., True),
        'period': torch.nn.functional.mse_loss,
        'lumen_width': torch.nn.functional.mse_loss
    }
    self.losses_weights = {
      'direction': self.hparams.loss_hparams['rotation_weight'],
      'period': 1,
      'lumen_width': self.hparams.loss_hparams['lumen_width_weight'],
      'regularization': self.hparams.loss_hparams.get('regularization_weight', 0.)
    }
  
  def _get_regularizer(self, regularizer_params):
    if regularizer_params is None:
      return None
    else:
      return instantiate(regularizer_params)

  def forward(self, x, return_raw=False):
    """get predictions from image batch"""
    preds = self.model(x) # preds: logit angle_sin, logit angle_cos, period, logit lumen fraction or logit angle, period, logit lumen fraction
    preds_direction = self.angle_parser(preds)
    preds_period = preds[:,-2]
    preds_lumen_width = preds[:,-1] #lumen fraction is between 0 and 1, so we take sigmoid fo this

    outputs = [preds_direction, preds_period, preds_lumen_width]
    if return_raw:
      outputs.append(preds)
      
    return tuple(outputs)

  def configure_optimizers(self):
    # AdamW is Adam with a correct implementation of weight decay (see here
    # for details: https://arxiv.org/pdf/1711.05101.pdf)    
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, **self.hparams.optimizer_hparams)
    # scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_hparams['classname'])(optimizer, **self.hparams.lr_hparams['kwargs'])
    scheduler = instantiate({**self.hparams.lr_hparams, '_partial_': True})(optimizer)
    return [optimizer], [scheduler]

  def process_batch_supervised(self, batch):
    """get predictions, losses and mean errors (MAE)"""

    # get predictions
    preds = {}
    preds['direction'], preds['period'], preds['lumen_width'], preds_raw = self.forward(batch['image'], return_raw=True) # preds: angle, period, lumen fraction, raw preds

    # calculate losses
    losses = {
        'direction': self.losses['direction'](2*batch['direction'], 2*preds['direction']),
        'period': self.losses['period'](batch['period'], preds['period']),
        'lumen_width': self.losses['lumen_width'](batch['lumen_width'], preds['lumen_width']),
    }
    if self.regularizer is not None:
      losses['regularization'] = self.regularizer(preds_raw[:,:2])
    
    losses['final'] = \
      losses['direction']*self.losses_weights['direction'] + \
      losses['period']*self.losses_weights['period'] + \
      losses['lumen_width']*self.losses_weights['lumen_width'] + \
      losses.get('regularization', 0.)*self.losses_weights.get('regularization', 0.)

    # calculate mean errors
    period_difference = np.mean(abs(
      batch['period'].detach().cpu().numpy() - \
      preds['period'].detach().cpu().numpy()
    ))

    a1 = batch['direction'].detach().cpu().numpy()
    a2 = preds['direction'].detach().cpu().numpy()
    angle_difference = np.mean(0.5*np.degrees(np.arccos(np.cos(2*np.radians(a2-a1)))))

    lumen_width_difference = np.mean(abs(preds['lumen_width'].detach().cpu().numpy()-batch['lumen_width'].detach().cpu().numpy()))
    
    lumen_fraction_pred = preds['lumen_width'].detach().cpu().numpy()/preds['period'].detach().cpu().numpy()
    lumen_fraction_gt = batch['lumen_width'].detach().cpu().numpy()/batch['period'].detach().cpu().numpy()
    lumen_fraction_difference = np.mean(abs(lumen_fraction_pred-lumen_fraction_gt))

    mae = {
      'period': period_difference,
      'direction': angle_difference,
      'lumen_width': lumen_width_difference,
      'lumen_fraction': lumen_fraction_difference
    }

    return preds, losses, mae

  def log_all(self, losses, mae, prefix=''):
    for k, v in losses.items():
        self.log(f'{prefix}{k}_loss', v.item() if isinstance(v, torch.Tensor) else v)
    for k, v in mae.items():
        self.log(f'{prefix}{k}_difference', v.item() if isinstance(v, torch.Tensor) else v)
  
  def training_step(self, batch, batch_idx):
    # "batch" is the output of the training data loader.
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='train_')

    return losses['final'] 

  def validation_step(self, batch, batch_idx):
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='val_')
  
  def test_step(self, batch, batch_idx):
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='test_')

       
    
# class StripsModel(StripsModelGeneral):
#   def __init__(self, model_name, *args, **kwargs):
#     super().__init__( *args, **kwargs)
#     self.model = timm.create_model(model_name, in_chans=1, num_classes=4)
#   def forward(self, x):
#     """get predictions from image batch"""
#     preds = self.model(x) # preds: logit angle_sin, logit angle_cos, period, logit lumen fraction
#     preds_sin = 1. - 2*torch.sigmoid(preds[:,0])
#     preds_cos = 1. - 2*torch.sigmoid(preds[:,1])
#     preds_direction = 0.5*torch.rad2deg(torch.arctan2(preds_sin, preds_cos))
#     preds_period = preds[:,2]
#     preds_lumen_fraction = torch.sigmoid(preds[:,3]) #lumen fraction is between 0 and 1, so we take sigmoid fo this
#     return preds_direction, preds_period, preds_lumen_fraction

# class StripsModelAngle1(StripsModelGeneral):
#   def __init__(self, model_name, *args, **kwargs):
#     super().__init__( *args, **kwargs)
#     self.model = timm.create_model(model_name, in_chans=1, num_classes=3)
#   def forward(self, x):
#     """get predictions from image batch"""
#     preds = self.model(x) # preds: logit angle_sin, logit angle
#     preds_direction = torch.pi * torch.sigmoid(preds[:,0])
#     preds_period = preds[:,1]
#     preds_lumen_fraction = torch.sigmoid(preds[:,2]) #lumen fraction is between 0 and 1, so we take sigmoid fo this
#     return preds_direction, preds_period, preds_lumen_fraction       
        