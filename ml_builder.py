import numpy as np
import random as rd
import xarray as xr
import time as tm

from model.baselines import (
    conv2plus1d, conv3d, encoder_decoder3d,
    mim, predrnn, stconvlstm, stconvs2s
)
from model.ablation import nosocial
from model import slstm, sclstm
 
from tool.train_evaluate import Trainer, Evaluator
from tool.dataset import NetCDFDataset
from tool.loss import RMSELoss, RMSEDownSample, L1LossDownSample
from tool.utils import Util

import torch
from torch.utils.data import DataLoader
from torch.nn import L1Loss

class MLBuilder:

    def __init__(self, config, device):
        
        self.config = config
        self.device = device
        self.dataset_type = 'small-dataset' if (self.config.small_dataset) else 'full-dataset'
        self.step = str(config.step)
        self.dataset_name, self.dataset_file = self.__get_dataset_file()
        self.dropout_rate = self.__get_dropout_rate()
        self.filename_prefix = self.dataset_name + '_step' + self.step
                
    def run_model(self, number):
        self.__define_seed(number)
        validation_split = 0.2
        test_split = 0.2
        # Loading the dataset
        ds = xr.open_mfdataset(self.dataset_file)
        if (self.config.small_dataset):
            ds = ds[dict(sample=slice(0,500))]

        train_dataset = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split)
        val_dataset   = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, is_validation=True)
        test_dataset  = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, is_test=True)
        if self.config.verbose:
            print('[X_train] Shape:', train_dataset.X.shape)
            print('[y_train] Shape:', train_dataset.y.shape)
            print('[X_val] Shape:', val_dataset.X.shape)
            print('[y_val] Shape:', val_dataset.y.shape)
            print('[X_test] Shape:', test_dataset.X.shape)
            print('[y_test] Shape:', test_dataset.y.shape)
            print(f'Train on {len(train_dataset)} samples, validate on {len(val_dataset)} samples')
                        
        params = {'batch_size': self.config.batch, 
                  'num_workers': self.config.workers, 
                  'worker_init_fn': self.__init_seed}

        train_loader = DataLoader(dataset=train_dataset, shuffle=True,**params)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False,**params)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, **params)
        
        models = {
            'stconvs2s-r': stconvs2s.STConvS2S_R,
            'stconvs2s-c': stconvs2s.STConvS2S_C,
            'convlstm': stconvlstm.STConvLSTM,
            'predrnn': predrnn.PredRNN,
            'mim': mim.MIM,
            'conv2plus1d': conv2plus1d.Conv2Plus1D,
            'conv3d': conv3d.Conv3D,
            'enc-dec3d': encoder_decoder3d.Endocer_Decoder3D,
            'vlstm': nosocial.VanillaLSTM_Downsample,
            'slstm': slstm.SocialLSTM_Downsample,
            'sclstm': sclstm.SocialConvLSTM,
        }
        if self.config.model not in models:
            raise ValueError(f'{self.config.model} is not a valid model name. Choose between: {models.keys()}')
            quit()
            
        # Creating the model    
        model_bulder = models[self.config.model]
        model = model_bulder(
            input_size=train_dataset.X.shape, 
            num_layers=self.config.num_layers, 
            hidden_dim=self.config.hidden_dim, 
            kernel_size=self.config.kernel_size, 
            device=self.device, 
            dropout_rate=self.dropout_rate, 
            step=int(self.step)
        )
        model.to(self.device)
        metrics = {
            'rmse': (RMSELoss, RMSEDownSample),
            'mae': (L1Loss, L1LossDownSample)
        }
        if self.config.loss not in metrics:
            raise ValueError(f'{self.config.loss} is not a valid loss function name. Choose between: {models.keys()}')
            quit()
        if self.config.model in ['vlstm', 'slstm']:
            loss = metrics[self.config.loss][1](train_dataset.X.shape)
        else:
            loss = metrics[self.config.loss][0]()
        loss.to(self.device)

        opt_params = {'lr': 0.001, 
                      'alpha': 0.9, 
                      'eps': 1e-6}
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_params)
        util = Util(
            self.config.model, self.dataset_type, 
            self.config.version, self.filename_prefix)
        train_info = {'train_time': 0}
        if self.config.pre_trained is None:
            train_info = self.__execute_learning(
                model, loss, optimizer, train_loader,  val_loader, util) 
                                                 
        eval_info = self.__load_and_evaluate(
            model, loss, optimizer, test_loader, 
            train_info['train_time'], util)

        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()

        return {**train_info, **eval_info}


    def __execute_learning(self, model, loss, optimizer, train_loader, val_loader, util):
        checkpoint_filename = util.get_checkpoint_filename()    
        trainer = Trainer(
            model, loss, optimizer, 
            train_loader, val_loader, self.config.epoch, 
            self.device, util, self.config.verbose, 
            self.config.patience, self.config.no_stop,
            self.config.model)
    
        start_timestamp = tm.time()
        # Training the model
        train_losses, val_losses = trainer.fit(checkpoint_filename, is_chirps=self.config.chirps)
        end_timestamp = tm.time()
        # Learning curve
        util.save_loss(train_losses, val_losses)
        util.plot([train_losses, val_losses], ['Training', 'Validation'], 'Epochs', 'Loss',
                  'Learning curve - ' + self.config.model.upper(), self.config.plot)

        train_time = end_timestamp - start_timestamp       
        print(f'\nTraining time: {util.to_readable_time(train_time)} [{train_time}]')
               
        return {'dataset': self.dataset_name,
                'dropout_rate': self.dropout_rate,
                'train_time': train_time
                }
                
    
    def __load_and_evaluate(self, model, loss, optimizer, test_loader, train_time, util):  
        evaluator = Evaluator(model, loss, optimizer, test_loader, self.device, self.config.model, util, self.step)
        if self.config.pre_trained is not None:
            # Load pre-trained model
            best_epoch, val_loss = evaluator.load_checkpoint(self.config.pre_trained, self.dataset_type, self.config.model)
        else:
            # Load model with minimal loss after training phase
            checkpoint_filename = util.get_checkpoint_filename() 
            best_epoch, val_loss = evaluator.load_checkpoint(checkpoint_filename)
        
        time_per_epochs = 0
        if not(self.config.no_stop): # Earling stopping during training
            time_per_epochs = train_time / (best_epoch + self.config.patience)
            print(f'Training time/epochs: {util.to_readable_time(time_per_epochs)} [{time_per_epochs}]')
        
        test_rmse, test_mae = evaluator.eval(is_chirps=self.config.chirps)
        print(f'Test RMSE: {test_rmse:.4f}\nTest MAE: {test_mae:.4f}')
                        
        return {'best_epoch': best_epoch,
                'val_rmse': val_loss,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'train_time_epochs': time_per_epochs
                }
          
    def __define_seed(self, number):      
        if (~self.config.no_seed):
            # define a different seed in every iteration 
            seed = (number * 10) + 1000
            np.random.seed(seed)
            rd.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic=True
            
    def __init_seed(self, number):
        seed = (number * 10) + 1000
        np.random.seed(seed)
        
    def __get_dataset_file(self):
        dataset_file, dataset_name = None, None
        if (self.config.chirps):
            dataset_file = 'data/dataset-chirps-1981-2019-seq5-ystep' + self.step + '.nc'
            dataset_name = 'chirps'
        else:
            dataset_file = 'data/dataset-ucar-1979-2015-seq5-ystep' + self.step + '.nc'
            dataset_name = 'cfsr'
        
        return dataset_name, dataset_file
        
    def __get_dropout_rate(self):
        dropout_rates = {
            'predrnn': 0.5,
            'mim': 0.5
        }
        if self.config.model in dropout_rates:
            dropout_rate = dropout_rates[self.config.model] 
        else:
            dropout_rate = 0.

        return dropout_rate