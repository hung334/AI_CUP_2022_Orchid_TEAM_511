import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from timm import create_model
from timm.data import create_dataset, create_loader
import torch.nn.functional as F 
from tensorboardX import SummaryWriter
from datetime import datetime
import torchvision.utils as vutils
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score
import warnings


warnings.filterwarnings('ignore')


writer = SummaryWriter('runs_V2_train_STN_two_fold/{}'.format(str(datetime.now())[:-10].replace(":","-")))

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

def load_trained_model(create_model, model_path):
    from collections import OrderedDict
    state_dict = torch.load(model_path)        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k]=v

    create_model.load_state_dict(new_state_dict)

    return create_model

class boundary_Loss(nn.Module):
    def __init__(self, s1 = 0.8,s2=0.6,device="cuda:0"):
        super(boundary_Loss, self).__init__()
        self.s1 = s1
        self.s2 = s2
        self.device = device
        
    def forward(self, input):
        loss = torch.sum(
                   (torch.max(torch.cat((((abs(input[:,0])+self.s1)-1).view(-1,1),torch.zeros(input.size(0)).to(device).view(-1,1)),1),1).values).pow(2)+
                   (torch.max(torch.cat((((abs(input[:,1])+self.s1)-1).view(-1,1),torch.zeros(input.size(0)).to(device).view(-1,1)),1),1).values).pow(2)+
                   (torch.max(torch.cat((((abs(input[:,2])+self.s2)-1).view(-1,1),torch.zeros(input.size(0)).to(device).view(-1,1)),1),1).values).pow(2)+
                   (torch.max(torch.cat((((abs(input[:,3])+self.s2)-1).view(-1,1),torch.zeros(input.size(0)).to(device).view(-1,1)),1),1).values).pow(2))

        return torch.div(loss,input.size(0))

def boundary_loss(input,device,s1=0.8):
    loss = torch.sum(
               (torch.max(torch.cat((((abs(input[:,0])+s1)-1).view(-1,1),torch.zeros(input.size(0)).to(device).view(-1,1)),1),1).values).pow(2)+
               (torch.max(torch.cat((((abs(input[:,1])+s1)-1).view(-1,1),torch.zeros(input.size(0)).to(device).view(-1,1)),1),1).values).pow(2))

    loss = loss/input.size(0)
        
    return loss

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.36715875, 0.42368276, 0.48109331])
    std = np.array([0.21691893, 0.2164323 , 0.2139854 ])
    #mean = [0.36715875, 0.42368276, 0.48109331]
    #std = [0.21691893, 0.2164323 , 0.2139854 ]
    #原先Normalize是對每個channel個別做 減去mean, 再除上std
    inp1 = std * inp + mean

    plt.imshow(inp)
    
def save_im(inp):
    """Imshow for Tensor."""
    #inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.36715875, 0.42368276, 0.48109331])
    std = np.array([0.21691893, 0.2164323 , 0.2139854 ])
    #mean = [0.36715875, 0.42368276, 0.48109331]
    #std = [0.21691893, 0.2164323 , 0.2139854 ]
    #原先Normalize是對每個channel個別做 減去mean, 再除上std
    inp1 = std * inp + mean
    
    return inp1


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



def acc_plt_show(num_epochs,training_accuracy,validation_accuracy,LR,save_file):
    plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
    plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
    plt.title('Train & Val accuracy,epoch:{}'.format(num_epochs))
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("{}/{}_acc.jpg".format(save_file,LR))
    plt.show()

def loss_plt_show(num_epochs,training_loss,validation_loss,LR,save_file):
    plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
    plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
    plt.title('Train & Val loss,epoch:{}'.format(num_epochs))
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    #plt.ylim(0, 1)
    plt.legend()
    plt.savefig("{}/{}_loss.jpg".format(save_file,LR))
    plt.show()
    
def plt_show(image):
    
    if(len(image.shape)==3):
        image = image[:,:,::-1]
    elif(len(image.shape)==2):
        image = image
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()
    
    
if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
        
    BATCH_SIZE = 4
    LR =0.00001
    EPOCHS = 300

    
    #******************************************************************************************
    
    one_fold_train_path = './Datasets/training/train'
    one_fold_dev_path = './Datasets/training/val'
    two_fold_train_path = './Datasets/AI_CUP_fold_complement/train'
    two_ld_dev_path = './Datasets/AI_CUP_fold_complement/val'
    
    train_path = two_fold_train_path
    dev_path = two_ld_dev_path
    

    resize = 384

    mean_384_one_fold = [0.5075337404927281 0.45864544276917535 0.4169235386212412] 
    std_384_one_fold = [0.2125643051799512 0.2385082849964861 0.22386483801695406]
    
    mean_384_two_fold = [0.45625534556274666, 0.4220624936173144, 0.3649616601198825]
    std_384_two_fold = [0.2143212828816861, 0.2210437745632625, 0.2062174242104951]
    
    mean = mean_384_two_fold
    std = std_384_two_fold
    '''
    crop_pct = 1#0.875
    img_size = resize
    scale_size = int(math.floor(img_size / crop_pct))
    print('scale_size',scale_size)
    tfms = transforms.Compose(
       [transforms.Resize(scale_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
    '''
    train_data = datasets.ImageFolder(train_path,transform=transforms.Compose([
        transforms.Resize(size=(resize,resize)),#(h,w)
        #transforms.Resize(resize, interpolation=Image.BICUBIC),
        #transforms.CenterCrop(resize),
        transforms.ColorJitter(brightness=0.5,saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))
    val_data = datasets.ImageFolder(dev_path,transform=transforms.Compose([
        transforms.Resize(size=(resize,resize)),#(h,w)
        #transforms.Resize(resize, interpolation=Image.BICUBIC),
        #transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))
    print(train_data.classes)#获取标签
    print(train_data.class_to_idx)
    train_Loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
    val_Loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    

    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    
    #*******************************************************************************************
    model_name = "resnet50"
    STN_local = create_model(model_name, pretrained=False,num_classes=219)
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
    STN_net = STN_net.to(device)
    #*******************************************************************************************
    
    
    save_file="save_look_0.8_0.6_STN_net_384_test"
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    
        with torch.no_grad():
            STN_net.eval()
            val_loss_reg,total_val,step_count,correct_val =0.0, 0,0,0
            for step, (batch_x,label_y) in enumerate(val_Loader):

                val = Variable(batch_x).cuda()#.to(device)
                labels = Variable(label_y).cuda()#.to(device)
                roi_0,roi_1,roi_2 = net(val)
                #print(label_y.cpu().data[0])
                
                img_display = np.transpose(roi_0.cpu().data[0].numpy(), (1,2,0)) #numpy:CHW, PIL:HWC
                plt.subplot(1,3,1)
                plt.axis('off')
                plt.imshow(img_display)
                
                plt.subplot(1,3,2)
                plt.axis('off')
                img_display = np.transpose(roi_1.cpu().data[0].numpy(), (1,2,0)) #numpy:CHW, PIL:HWC
                plt.imshow(img_display)
                
                plt.subplot(1,3,3)
                plt.axis('off')
                img_display = np.transpose(roi_2.cpu().data[0].numpy(), (1,2,0)) #numpy:CHW, PIL:HWC
                plt.imshow(img_display)
                
                #plt.show()
                plt.axis('off')
                plt.savefig("./{}/{}_CLASS-{}.png".format(save_file,step,train_data.classes[label_y.cpu().data[0]]))
                
                print("{}/{}".format(step,len(val_Loader)))
    