from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms#, models
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion()   # interactive mode
import argparse
import random
import math
import cv2
from PIL import Image
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score
# from efficientnet_pytorch import EfficientNet
from timm import create_model
from collections import OrderedDict


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class STN(nn.Module):

    def __init__(self, loc_model,device):
        super(STN_N, self).__init__()

        self.f_loc = loc_model
        self.device = device

    def forward(self, x,s1=0.8,s2=0.6): 

        batch_images = x

        theta = self.f_loc(x)

        theta_x_y = theta.view(-1, 4)
        affine_1 = torch.full([theta.size()[0],2,3],0.0)
        #print(theta_x_y)
        #print(affine_1.shape)
        affine_1[:,0,0]=s1
        affine_1[:,1,1]=s1
        affine_1[:,0,2]=theta_x_y[:,0]
        affine_1[:,1,2]=theta_x_y[:,1]
        affine_1 = affine_1.to(self.device)#.cuda()
        
        affine_2 = torch.full([theta.size()[0],2,3],0.0)
        affine_2[:,0,0]=s2
        affine_2[:,1,1]=s2
        affine_2[:,0,2]=theta_x_y[:,2]
        affine_2[:,1,2]=theta_x_y[:,3]
        affine_2 = affine_2.to(self.device)#.cuda()
        
        grid_1 = F.affine_grid(affine_1, batch_images.size())
        rois_1 = F.grid_sample(batch_images, grid_1)
        
        grid_2 = F.affine_grid(affine_2, batch_images.size())
        rois_2 = F.grid_sample(batch_images, grid_2)
              
        return batch_images,rois_1,rois_2,theta_x_y

def Get_STN_model(classes=219,device="cuda:0"):
    #*******************************************************************************************
    model_name = "resnet50"
    STN_local = create_model(model_name, pretrained=False,num_classes=classes)
    #STN_local.load_state_dict(torch.load(pkl_file,map_location=device))
    num_ftrs = STN_local.fc.in_features
    STN_local.fc = torch.nn.Sequential(nn.Linear(num_ftrs, 256),
                                       nn.Tanh(),
                                        nn.Linear(256, 4), 
                                        nn.Tanh(),)
    #*******************************************************************************************
    pkl_file = './save_STN_floc.pkl'
    STN_net = STN(loc_model=STN_local,device = device)
    STN_net.load_state_dict(torch.load(pkl_file,map_location=device))
    STN_net = nn.DataParallel(STN_net)
    STN_net = STN_net.cuda()
    #*******************************************************************************************
    print("STN_net load ok")
    for i,(name, parma) in enumerate(STN_net.named_parameters()):
                  parma.requires_grad = False
                  
    return STN_net
    
    

'''
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
'''
class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
'''
class LabelSmoothingLoss(nn.Module):
    def __init__(self, ignore_index=-1, eps=0.1, reduction="mean"):
        super().__init__()

        self.ignore_index = ignore_index
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        n_class = output.shape[-1]
        output = F.log_softmax(output, -1)

        if self.ignore_index > -1:
            n_class -= 1

        true_dist = torch.full_like(output, self.eps / n_class)
        true_dist.scatter_(
            1, target.data.unsqueeze(1), 1 - self.eps + self.eps / n_class
        )

        if self.ignore_index > -1:
            true_dist[:, self.ignore_index] = 0
            padding_mat = target.data == self.ignore_index
            mask = torch.nonzero(padding_mat, as_tuple=False)

            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        loss = F.kl_div(
            output,
            true_dist.detach(),
            reduction="sum" if self.reduction != "none" else "none",
        )

        if self.reduction == "none":
            loss = loss.sum(1)

        elif self.reduction == "mean":
            if self.ignore_index > -1:
                loss = loss / (target.shape[0] - padding_mat.sum().item())

            else:
                loss = loss / target.shape[0]

        return loss
'''

def load_trained_model(create_model, model_path):
    
    state_dict = torch.load(model_path)        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k]=v

    create_model.load_state_dict(new_state_dict)

    return create_model

def train_model(args, STN,model, criterion, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs=25):
    save_dir = f'./models/{args.id}/'
    path = save_dir + 'epoch_loss_acc_mF1.txt' # write .txt path
    since = time.time()
    best_acc = 0.0
    best_mF1 = 0.0
    lowest_loss = 2.0
    es = 0
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    train_mF1, valid_mF1 = [], []
    lr = []
    for epoch in range(args.restart_epoch, num_epochs):
        epoch_lr = get_lr(optimizer)
        lr.append(epoch_lr)
        print()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        print('learning rate:', epoch_lr)
        f = open(path, 'a')
        print('Epoch {}/{}'.format(epoch+1, num_epochs), file=f)
        print('-' * 10, file=f)       
        print('learning rate:', epoch_lr, file=f)
        f.close()
        
        if epoch > args.restart_epoch + 1 and lr[-1] < lr[-2]:
            print(f'Learning rate decrease from {lr[-2]} to {lr[-1]}!!')
            # load best model weights
            # best_trained_model_path = f'./models/{args.id}/mF1_{best_epoch}_{best_mF1}.pth'
            # model = model.load_state_dict(torch.load(best_trained_model_path))
            model.load_state_dict(best_model_wts)
            # print('load trained model from', best_trained_model_path)
            # model = model.cuda()
            # model = nn.DataParallel(model)
            # print(next(model.parameters()).is_cuda)
            print('Load best model weights!!')
            f = open(path, 'a')
            print(f'Learning rate decrease from {lr[-2]} to {lr[-1]}!!', file=f)
            # print('Please pick the best trained model to keep training!!')
            print('Load best model weights!!', file=f)
            f.close()
            # return 

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            pred_list, label_list = [], []
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            step = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    roi_0,roi_1,roi_2,theta_x_y= STN(inputs)
                    outputs_0 = model(roi_0)
                    outputs_1 = model(roi_1)
                    outputs_2 = model(roi_2)
                    
                    final_ans = F.softmax(outputs_0, dim=1)+F.softmax(outputs_1, dim=1)+F.softmax(outputs_2, dim=1) 
                    _, preds = torch.max(final_ans, 1)
                    #loss = criterion(outputs, labels)
                    loss = (criterion(outputs_0,labels)+criterion(outputs_1,labels)+criterion(outputs_2,labels))/3 
                    
            
                    
                    for label, pred in zip(labels.cpu().detach().numpy().tolist(), preds.cpu().detach().numpy().tolist()):
                        label_list.append(label)
                        pred_list.append(pred)

                # backward + optimize only if in training phase
                if phase == 'train':
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) # contains the loss of entire mini-batch, but divided by the batch size
                running_corrects += torch.sum(preds == labels.data)
                step += 1
                if phase == 'train' and step % 100 == 0:
                    print("Epoch:{}/{} Step:{}/{} Train_loss:{:1.3f} ".format(epoch+1,num_epochs,step+1,len(dataloaders[phase]),loss))
                elif phase == 'trvalain' and step % 100 == 0:
                    print("Epoch:{}/{} Step:{}/{} Val_loss:{:1.3f} ".format(epoch+1,num_epochs,step+1,len(dataloaders[phase]),loss))
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_macro_F1 = get_score(label_list, pred_list)
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_mF1.append(epoch_macro_F1)                
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                valid_mF1.append(epoch_macro_F1)
                scheduler.step(valid_mF1[-1])
                plt.figure(0)
                plt.plot(range(args.restart_epoch+1,epoch+2,1), np.array(train_loss), 'b-')
                plt.plot(range(args.restart_epoch+1,epoch+2,1), np.array(valid_loss), 'r-')
                plt.title(f'Train & Val loss')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.show()
                plt.savefig(f"{save_dir}{args.id}_loss_{args.restart_epoch}.png")
                plt.figure(1)
                plt.plot(range(args.restart_epoch+1,epoch+2,1), np.array(train_acc), 'b-')
                plt.plot(range(args.restart_epoch+1,epoch+2,1), np.array(valid_acc), 'r-')
                plt.title(f'Train & Val accuracy')
                plt.xlabel('epoch')
                plt.ylabel('accuracy')
                plt.show()
                plt.savefig(f"{save_dir}{args.id}_accuracy_{args.restart_epoch}.png")
                plt.figure(2)
                plt.plot(range(args.restart_epoch+1,epoch+2,1), np.array(train_mF1), 'b-')
                plt.plot(range(args.restart_epoch+1,epoch+2,1), np.array(valid_mF1), 'r-')
                plt.title(f'Train & Val Macro F1')
                plt.xlabel('epoch')
                plt.ylabel('Macro F1')
                plt.show()
                plt.savefig(f"{save_dir}{args.id}_macroF1_{args.restart_epoch}.png")
                plt.figure(3)
                plt.plot(range(args.restart_epoch+1,epoch+2,1), np.array(lr), 'b-')
                plt.title(f'Learning Rate')
                plt.xlabel('epoch')
                plt.ylabel('lr')
                plt.show()
                plt.savefig(f"{save_dir}{args.id}_lr_{args.restart_epoch}.png")

            print('{} Loss: {:.6f} Acc: {:.6f} Macro-F1:{:.6f}'.format(phase, epoch_loss, epoch_acc, epoch_macro_F1))
            f = open(path, 'a')
            print('{} Loss: {:.6f} Acc: {:.6f} Macro-F1:{:.6f}'.format(phase, epoch_loss, epoch_acc, epoch_macro_F1), file=f)
            f.close()
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # torch.save(model.state_dict(), save_dir + f'Acc_{epoch + 1}_{epoch_acc}.pth')
                
            if phase == 'val' and epoch_macro_F1 > best_mF1:
                es = 0
                best_mF1 = epoch_macro_F1
                best_epoch = epoch + 1                
                torch.save(model.state_dict(), save_dir + f'mF1_{best_epoch}_{epoch_macro_F1}.pth')
                best_model_wts = copy.deepcopy(model.state_dict())
                # best_model_wts = best_model_wts.cpu()
            elif phase == 'val' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                torch.save(model.state_dict(), save_dir + f'Loss_{epoch + 1}_{epoch_loss}.pth')
            elif phase == 'val':
                torch.save(model.state_dict(), save_dir + f'XXXX_{epoch + 1}_{epoch_loss}.pth')
                
            if phase == 'val' and epoch_macro_F1 < best_mF1:
                es += 1
            
        if es >= args.earlystop:
            print('Early stopping!')
            break
    # # load best model weights
    # model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), save_dir + f'{best_epoch}_eff-{args.efficientnet_bx}_{best_acc}_best.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val epoch(macro_F1): {best_epoch}')
    print(f'Best train loss: {train_loss[best_epoch - 1]}')
    print(f'Best val loss: {valid_loss[best_epoch - 1]}')
    print(f'Best train Acc: {train_acc[best_epoch - 1]}')
    print(f'Best val Acc: {best_acc}')
    print(f'Best train macro-F1: {train_mF1[best_epoch - 1]}')
    print(f'Best val macro-F1: {best_mF1}')
    f = open(path, 'a')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
    print(f'Best val epoch(Acc): {best_epoch}', file=f)
    print(f'Best train loss: {train_loss[best_epoch - 1]}', file=f)
    print(f'Best val loss: {valid_loss[best_epoch - 1]}', file=f)
    print(f'Best train Acc: {train_acc[best_epoch - 1]}', file=f)
    print(f'Best val Acc: {best_acc}', file=f)
    print(f'Best train macro-F1: {train_mF1[best_epoch - 1]}', file=f)
    print(f'Best val macro-F1: {best_mF1}', file=f)
    f.close()
    return model

def get_score(label,predicted):
    Accuracy = accuracy_score(label, predicted)  #(tp+tn)/(tp+fp+fn+tn)
    # Precision = precision_score(label, predicted,average=None)  #tp/(tp+fp)
    # Recall = recall_score(label, predicted,average=None)  #tp/(tp+fn)
    F1 = f1_score(label, predicted,average=None)  #2 / ( (1/ Precision) + (1/ Recall) )
    final_score = 0.5*Accuracy + 0.5*F1.mean()
    return final_score
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mean_std(args):
    data_dir = args.data_dir
    im_size = (args.input_size,args.input_size)
    mean_std_transforms = {
        'train': transforms.Compose([
            transforms.Resize((im_size), interpolation=Image.BICUBIC ),
            transforms.ToTensor(),
        ])
    }
    image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), mean_std_transforms[x]) for x in ['train']}
    train_loader = torch.utils.data.DataLoader(
        image_dataset, batch_size=1, shuffle=False, num_workers=8, )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print(mean)
    print(std)

def cal_mean_std(args, images_dir):
    for i in range(219):
        img_filenames = os.listdir(f'{images_dir}/{i}/')
        # print(img_filenames)
        m_list, s_list = [], []
        for img_filename in img_filenames:
            # print(f'{images_dir}/{i}/{img_filename}')

            img = cv2.imread(f'{images_dir}/{i}/{img_filename}')
            img_resized = cv2.resize(img, (args.input_size, args.input_size), interpolation=cv2.INTER_CUBIC)
            img_resized = img_resized / 255.0
            m, s = cv2.meanStdDev(img_resized)

            m_list.append(m.reshape((3,)))
            s_list.append(s.reshape((3,)))
            #print(m_list)
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    # print('mean: ',m[0][::-1]*255)
    # print('std:  ',s[0][::-1]*255)
    return m , s

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main(args, mean, std):
    data_dir = args.data_dir
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
#     mean = [0.36715875, 0.42368276, 0.48109331]
#     std = [0.21691893, 0.2164323 , 0.2139854]
    # mean_224 = [0.4811, 0.4237, 0.3672]
    # std_224 = [0.2441, 0.2390, 0.2467]
    # mean_384 = [0.4812, 0.4238, 0.3673]
    # std_384 = [0.2462, 0.2414, 0.2489]
    if args.hw_ratio:
        im_size = (int(480 * args.hw_ratio), int(640 * args.hw_ratio))
    else:
        im_size = (args.input_size,args.input_size)
    '''
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((im_size), interpolation=Image.BICUBIC ),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(brightness=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((im_size), interpolation=Image.BICUBIC ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    '''
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.5, saturation=0.5),            
            #transforms.RandomPerspective(0.3, 0.5, Image.BICUBIC),#distortion_scale, p
            transforms.Resize((im_size), interpolation=Image.BICUBIC ),   
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            #transforms.RandomAffine(degrees=20, translate=(0.2, 0.1), resample=Image.BILINEAR),
            transforms.Normalize(mean, std)
            ]),
        'val': transforms.Compose([
            transforms.Resize((im_size), interpolation=Image.BICUBIC ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    ## swin v2 large 192
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ColorJitter(brightness=0.5, saturation=0.5),
    #         transforms.Resize((224,224), interpolation=Image.BICUBIC ),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip(),            
    #         transforms.RandomRotation(degrees=15, expand=True),
    #         transforms.RandomCrop((im_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize((224,224), interpolation=Image.BICUBIC ),
    #         transforms.CenterCrop((im_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #     ]),
    # }
    
    ##beit
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ColorJitter(brightness=0.5, saturation=0.5),
    #         transforms.Resize((256,256), interpolation=Image.BICUBIC ),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip(),
    #         transforms.RandomRotation(degrees=15, expand=True),
    #         transforms.RandomCrop((im_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std),
    #         transforms.RandomErasing(p=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.3))
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize((256,256), interpolation=Image.BICUBIC ),
    #         transforms.CenterCrop((im_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #     ]),
    # }
    
    # swin v1 : strong augmentation
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomRotation(degrees=90, expand=True),#10
#             transforms.RandomCrop((480,480)),
#             transforms.Resize((im_size), interpolation=Image.BICUBIC ),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ColorJitter(brightness=0.5, saturation=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize(mean_384, std_384),
#             transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize((im_size), interpolation=Image.BICUBIC ),
#             transforms.ToTensor(),
#             transforms.Normalize(mean_384, std_384)
#         ]),
#     }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True),
                     'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    # cuda_id = ''
    # for i in range(len(args.device.split(' '))):
    #     id = args.device.split(' ')[i]
    #     if i != len(args.device.split(' ')) - 1:
    #         cuda_id += f'{id},'
    #     else:
    #         cuda_id += f'{id}'
    # device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    # print(f"Using device {device}")
    print(f"batch_size:{args.batch_size}")
    print(f'lr:{args.lr}')
    print(f'dataset_size:{dataset_sizes}\n')
    print(f"data_dir:{args.data_dir}\n")
    print(f'input_size:{im_size}\n')
    #*********************************************************************** STN
    STN = Get_STN_model()
    #***********************************************************************
    pkl_path="~~~"
    #model_ft = create_model(args.model_name, pretrained=True,num_classes=len(class_names))
    model_ft = create_model(args.model_name, pretrained=False,num_classes=len(class_names), checkpoint_path=pkl_path)
    #print(model)
    #model = nn.DataParallel(model)
    #model = model.to(device)
    #***********************************************************************
    
    # model_ft = create_model(args.model_name)
    # model_ft.classifier = nn.Linear(model_ft.classifier.in_features, len(class_names))
    if len(args.model_path) != 0:
        state_dict = torch.load(args.model_path)#, map_location=lambda storage, loc: storage.cuda(0)
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            # print(k)
            if 'module' in k:
                k = k.replace('module.', '')
            new_state_dict[k]=v

        model_ft.load_state_dict(new_state_dict)
        # model_ft.load_state_dict(torch.load(args.model_path))
        print(f'load trained model: {args.model_path}\n')
    # print(model_ft)
    parameter_count = count_parameters(model_ft)
    print(f"#parameters:{parameter_count}")
    model_ft = model_ft.cuda()
    model_ft = nn.DataParallel(model_ft)#,device_ids = [int(i) for i in args.device])
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingLossCanonical(smoothing=args.smoothing)
    optimizer = optim.AdamW(model_ft.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_min = args.min_lr
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, patience=args.patience, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=lr_min, eps=1e-08)
    ## warm_up + step
    # lambda0 = lambda epoch: (epoch / args.warm_up_epoch + lr_min) if epoch < args.warm_up_epoch else (args.lr * args.gamma ** ((epoch - args.warm_up_epoch) // args.step_size) + lr_min) / args.lr
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda0])

    ## warm_up + smooth step(cos)
    # T_max = int(args.epoch - args.warm_up_epoch)* args.t_max_ratio
    # lambda0 = lambda epoch: ((epoch + 1) / args.warm_up_epoch) if epoch < args.warm_up_epoch else (lr_min + 0.5 * (args.lr - lr_min) * (1.0 + math.cos((epoch - args.warm_up_epoch) / (T_max - args.warm_up_epoch) * math.pi))) / args.lr
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda0])
    
    ## no warm_up
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.step_size, eta_min=0.0000001)
    '''
    lr = []
    for epoch in range(0, args.epoch):
        scheduler.step(epoch)
        print(epoch, scheduler.get_last_lr())   
        lr.append(scheduler.get_last_lr())
        # optimizer.step()
    plt.figure(0)
    plt.plot(range(args.epoch), np.array(lr), 'b-', label= 'train')
    plt.title('LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.show()
    plt.savefig(f"./models/{args.id}/{args.id}_lr.png")    
    '''
    model_ft = train_model(args, STN,model_ft, criterion, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs=args.epoch)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()#swinv2_large_window12_192_22k_jsfrc
    parser.add_argument('--id', type=str, default='beit_large_patch16_224_in22k_weight_decay0.05')# patient, weight decay (5e-2 beit), transform, lr(beit->min:1e-6)
    parser.add_argument('--model_name', type=str, default='beit_large_patch16_224_in22k') #swinv2_large_window12_192_22k
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--hw_ratio', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--restart_epoch', type=int, default=0)
    parser.add_argument('--model_path', type=str, default = '')
    parser.add_argument('--data_dir', type=str, default = './orchids_219/training/')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    # parser.add_argument('--step_size', type=int, default=2)
    # parser.add_argument('--t_max_ratio', type=float, default=0.2)
    # parser.add_argument('--warm_up_epoch', type=float, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--earlystop', type=int, default=50)
    parser.add_argument('--device', type=str, default = '1, 0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    save_dir = f'./models/{args.id}/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    
    image_path = args.data_dir + 'train'
    mean, std = cal_mean_std(args, image_path)
    mean = list(mean[0][::-1])
    std = list(std[0][::-1])
    print(mean, std)
    
                
    main(args, mean, std)