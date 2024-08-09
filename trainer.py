import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):

        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
    
        if self._cuda:
            x, y = x.cuda(), y.cuda()
        
        # -propagate through the network
        outputs = self._model(x)
        
        # -calculate the loss
        loss = self._crit(outputs, y.float())
        
        # -compute gradient by backward propagation
        loss.backward()
        
        # -update weights
        self._optim.step()
        
        # -return the loss
        return loss
        
        
    
    def val_test_step(self, x, y):
    
    # predict
    # move x and y to predict
        if self._cuda:
            x, y = x.cuda(), y.cuda()
        
        # set the model to evaluation mode
        self._model.eval()   
        
        # propagate through the network and calculate the loss and predictions    
        outputs = self._model(x)
        loss = self._crit(outputs, y.float())
        predictions = t.round(outputs).int()
        
        # return the loss and predictions
        return loss.item(), predictions
        
    def train_epoch(self):
        # set training mode
        self._model.train()
        
        # initialize loss and batch
        total_loss = 0.0
        num_batches = len(self._train_dl)
        
    #create progress bar
        with tqdm(total=num_batches, desc="Training") as pbar:
            # iterate through the training set
            for x, y in self._train_dl:
            
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
                if self._cuda:
                    x, y = x.cuda(), y.cuda()

        # perform a training step
                loss = self.train_step(x, y)
                
                # calculate the average loss for the epoch and return it
                total_loss += loss.item()

        #update progress bar
                pbar.update(1)

        t.cuda.empty_cache()

        return total_loss / num_batches

    def val_test(self):
    
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        
        # initialize
        total_loss = 0.0
        all_predictions = None
        all_labels = None
        
        
    # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
        
            # iterate through the validation set    
            for x, y in self._val_test_dl:
            
                # transfer the batch to the gpu if given
                if self._cuda:
                    x,y = x.cuda(), y.cuda()
                        
                # perform a validation step
                loss, predictions = self.val_test_step(x, y)
                total_loss += loss

                # save the predictions and the labels for each batch
                if all_predictions is None:
                    all_predictions = predictions.cpu().numpy()
                    all_labels = y.cpu().numpy()
                else:
                    all_predictions = np.append(all_predictions, predictions.cpu().numpy(),axis=0)
                    all_labels = np.append(all_labels, y.cpu().numpy(),axis=0)

    # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        avg_loss = total_loss / len(self._val_test_dl)

        f1_c = f1_score(all_labels[:,0], all_predictions[:,0], average='weighted')
        f1_i = f1_score(all_labels[:,1], all_predictions[:,1], average='weighted')
        accuracy = np.sum(all_labels == all_predictions, axis = 0)
        acc = accuracy/all_predictions.shape[0]

        print(f"Validation/Test Loss: {avg_loss:.4f}")
        print("F1 score cracked: %f" % (f1_c))
        print("F1 score inactiv: %f" % (f1_i))
        print("Accuracy cracks: %f" % acc[0])
        print("Accuracy inactive: %f" % acc[1])

    # return the loss and print the calculated metrics
        return avg_loss#, f1
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        
        t.cuda.empty_cache()

        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        while True:
      
            # stop by epoch number
            if epochs < 0:
                break

        # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            val_loss = self.val_test()

        # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epochs)
                patience_counter = 0
            else:
                patience_counter += 1
                    
                # check whether early stopping should be performed using the early stopping criterion and stop if so
                if patience_counter >= self._early_stopping_patience:
                    print("Early stopping triggered.")
                    break
            epochs -= 1


        t.cuda.empty_cache()

    # return the losses for both training and validation
        return train_losses, val_losses
        
        
        
