from resnet18 import ResNet
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import math
import sys
import imlib as im
from torchvision.utils import save_image
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, loader, config):
        """Initialize configurations."""

        # Data loader.
        self.loader = loader

        # Model configurations.
        self.dim = config.dim

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.r_lr = config.r_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.model_name = config.model_name
        self.img_kind = config.img_kind


        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        # Build the model and tensorboard.
        self.build_model()


    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def build_model(self):
        """Create a classifier."""
        self.ResNet = ResNet(num_classes=self.dim)
        self.r_optimizer = torch.optim.Adam(self.ResNet.parameters(), self.r_lr, [self.beta1, self.beta2])
        self.print_network(self.ResNet, 'ResNet')
        self.ResNet.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained classifier."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        ResNet_path = os.path.join(self.model_save_dir, '{}-ResNet-{}.ckpt'.format(resume_iters,self.model_name))
        self.ResNet.load_state_dict(torch.load(ResNet_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.r_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, dim=4):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(dim):
            c_trg = c_org.clone()
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute mse loss."""
        return F.mse_loss(logit, target,reduction='sum')

    def train(self):
        """Train ResNet within a single dataset."""
        
        
        data_loader = self.loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.dim)

        # Learning rate cache for decaying.
        r_lr = self.r_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):


            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            c_org = label_org.clone()

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.


            # # =================================================================================== #
            # #                             2. Train the classifier                                 #
            # # =================================================================================== #

            #Compute loss with real images.
            out_cls = self.ResNet(x_real)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Backward and optimize.
            r_loss =  d_loss_cls
            loss = r_loss.cpu().detach().numpy()
                
            self.reset_grad()
            r_loss.backward()
            self.r_optimizer.step()

            # Logging.
            loss = {}
            loss['R/loss_cls'] = r_loss.item()
            

            # =================================================================================== #
            #                                 3. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                ResNet_path = os.path.join(self.model_save_dir, '{}-ResNet-{}.ckpt'.format(i+1,self.model_name))
                torch.save(self.ResNet.state_dict(), ResNet_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))


    def test(self):
        f = open("acc_"+self.model_name+".txt","a")

        # Load the trained classifier.
        self.restore_model(self.test_iters)

        # Load the dataset.
        data_loader = self.loader
        test_len = len(data_loader.dataset)# Load the dataset.r.dataset.test_dataset)

        # calculate test acc
        with torch.no_grad():

            res_matrix = np.zeros((21,21))
            sum_sample = np.zeros(21)
            right_sample = np.zeros(21)
            ratio = np.zeros(21)
            y_true = np.zeros(test_len)
            y_pred = np.zeros(test_len)
            num=0

            for i, (x_real, c_org) in enumerate(data_loader):
                x_real = x_real.to(self.device)
                fake_label=self.ResNet(x_real)
                fake_label = fake_label.cpu().numpy()
                c_org = c_org.cpu().numpy()

                for k in range(len(c_org[:,0])):
                    real_class = np.argmax(c_org[k])
                    res_class = np.argmax(fake_label[k])
                    res_matrix[real_class,res_class]+=1
                    y_true[num] = real_class
                    y_pred[num] = res_class
                    num +=1

                for k in range(21):
                    sum_sample[k] = np.sum(res_matrix[k,:])
                    right_sample[k] = res_matrix[k,k]
                    ratio[k] = right_sample[k]/sum_sample[k]*100
                print(np.mean(ratio))
            f.write(str(np.mean(ratio)))
            f.write("\n")




                # plot confusion matrix
                # cm = confusion_matrix(y_true[:num], y_pred[:num])
                # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                # disp.plot()
                # img_name = self.model_name+"2"+self.img_kind+".png"
                # plt.savefig(img_name)


