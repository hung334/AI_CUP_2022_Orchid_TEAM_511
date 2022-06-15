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
from PIL import Image

warnings.filterwarnings('ignore')




writer = SummaryWriter('runs_V2_train_STN_two_fold_swin_large/{}'.format(str(datetime.now())[:-10].replace(":","-")))

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
        super(STN, self).__init__()

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
        
    BATCH_SIZE = 8
    LR =0.000001
    EPOCHS = 300

    
    #******************************************************************************************
    
    k_fold = "two"
    
    one_fold_train_path = './Datasets/training/train'
    one_fold_dev_path = './Datasets/training/val'
    two_fold_train_path = './Datasets/AI_CUP_fold_complement/train'
    two_fold_dev_path = './Datasets/AI_CUP_fold_complement/val'
    
    train_path_list = {"one":one_fold_train_path,"two":two_fold_train_path}
    dev_path_list = {"one":one_fold_dev_path,"two":two_fold_dev_path}
    
    train_path = train_path_list[k_fold]
    dev_path = dev_path_list[k_fold]
    

    resize = 384

    mean_384_one_fold = [0.5075337404927281 ,0.45864544276917535 ,0.4169235386212412] 
    std_384_one_fold = [0.2125643051799512 ,0.2385082849964861 ,0.22386483801695406]
    
    mean_384_two_fold = [0.45625534556274666, 0.4220624936173144, 0.3649616601198825]
    std_384_two_fold = [0.2143212828816861, 0.2210437745632625, 0.2062174242104951]
    
    mean_list = {"one":mean_384_one_fold,"two":mean_384_two_fold}
    std_list = {"one":std_384_one_fold,"two":std_384_two_fold}
    
    mean = mean_list[k_fold]
    std = std_list[k_fold]
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
    train_Loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    val_Loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    pkl_one_file = './save/save_resnet50_adam_Smoothing_no_contrast_384/save_resnet50_best_64_0.8413459447706022.pkl'
    pkl_two_file = './save/save_resnet50_adam_Smoothing_no_contrast_384_two_fold/save_resnet50_best_205_0.8390773356526782.pkl'
    
    pkl_list = {"one":pkl_one_file,"two":pkl_two_file}
    
    #*******************************************************************************************
    model_name = "resnet50"
    STN_local = create_model(model_name, pretrained=False,num_classes=219)
    STN_local.load_state_dict(torch.load(pkl_list[k_fold],map_location=device))
    num_ftrs = STN_local.fc.in_features
    STN_local.fc = torch.nn.Sequential(nn.Linear(num_ftrs, 256),
                                       nn.Tanh(),
                                        nn.Linear(256, 4), 
                                        nn.Tanh(),)
    #********************************************************************************************
    model_name = "swin_large_patch4_window12_384"#'swinv2_base_window12to24_192to384_22kft1k'
    pkl_path_one = './save_pth/mF1_59_0.9381713415959991.pth'
    pkl_path_two = './save_pth/two_fold_mF1_253_0.9435420743639922.pth'
    
    pkl_path_list = {'one':pkl_path_one,'two':pkl_path_two}
    
    Local_net = create_model(model_name, pretrained=False,num_classes=len(train_data.classes))#,checkpoint_path=pkl_path)
    Local_net = load_trained_model(Local_net, pkl_path_list[k_fold])
    Local_net = nn.DataParallel(Local_net)
    Local_net = Local_net.to(device)
    #Local_net.requires_grad = False 
    for para in Local_net.parameters():#凍結參數
          para.requires_grad = False
    #*******************************************************************************************
    STN_net_pkl_file = "./save_STN/save_STN_net_best_6.0.00012421664723660797.pkl"
    STN_net = STN(loc_model=STN_local,device = device)
    #STN_net.load_state_dict(torch.load(STN_net_pkl_file,map_location=device))
    #STN_net.f_loc.fc = nn.DataParallel(STN_net.f_loc.fc)
    #STN_net = nn.DataParallel(STN_net)
    STN_net = STN_net.to(device)
    for i,(name, parma) in enumerate(STN_net.named_parameters()):
          if not any(i in name for i in ['f_loc.fc'] ):
                  parma.requires_grad = False
    #******************************************************************************************

    optimizer =torch.optim.Adam(STN_net.f_loc.fc.parameters(), lr=LR, betas=(0.9, 0.99))
    #optimizer =torch.optim.Adam(STN_net.parameters(), lr=LR, betas=(0.9, 0.99))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5, patience=10, verbose=True)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_boundary = boundary_Loss(device=device)
    
    STN_net.train()
    
    training_loss,validation_loss=[],[]
    
    training_accuracy,validation_accuracy =[],[]
    best_acc,best_epoch=0,0
    best_loss_boundary = 100
    
    save_file="save_V2_train_STN_two_fold_swin_large"
    
    aa = 0.8
    bb = 1-aa
    
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    
    for epoch in range(EPOCHS):
        STN_net.train()
        Local_net.eval()
        train_loss_reg,total_train,step_count,correct_train =0.0, 0,0,0
        loss_boundary_reg = 0.0
        
        for step, (batch_x,label_y) in enumerate(train_Loader):

            train = Variable(batch_x).to(device)
            labels = Variable(label_y).to(device)
            batch_images,rois_1,rois_2,theta_x_y = STN_net(train)
            
            
            outputs_1 = Local_net(rois_1)
            outputs_2 = Local_net(rois_2)
            
            train_loss = ((loss_func(outputs_1,labels)+loss_func(outputs_2,labels))/2)*bb + aa*loss_boundary(theta_x_y)
            #train_loss = loss_boundary(theta_x_y)
            #train_loss = loss_func(outputs,labels)
            loss_boundary_reg += loss_boundary(theta_x_y).cpu().data
            
            optimizer.zero_grad()               # clear gradients for this training step
            train_loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            
            final_ans = F.softmax(outputs_1, dim=1)+F.softmax(outputs_2, dim=1)
            
            train_loss_reg +=train_loss.cpu().data
            ans=torch.max(final_ans,1)[1].squeeze()
            correct_train += (ans.cpu() == labels.cpu()).float().sum()
            
            print("Epoch:{}/{} Step:{}/{} Train_loss:{:1.3f} ".format(epoch+1,EPOCHS,step+1,len(train_Loader),train_loss))
            print("Loss_boundary:{}".format(loss_boundary(theta_x_y).cpu().data))

        x = vutils.make_grid(batch_x, normalize=True, scale_each=True)
        writer.add_image('org', x, epoch+1)
        y_1 = vutils.make_grid(rois_1, normalize=True, scale_each=True)
        writer.add_image('crop_0.8', y_1 , epoch+1)
        y_2 = vutils.make_grid(rois_2, normalize=True, scale_each=True)
        writer.add_image('crop_0.6', y_2 , epoch+1)
        
        train_accuracy = 100 * correct_train / float(len(train_data))
        training_accuracy.append(train_accuracy)
        
        #print(step,step_count)
        avg_train_loss = train_loss_reg/len(train_Loader)
        training_loss.append(avg_train_loss)
        
        avg_loss_boundary = loss_boundary_reg/len(train_Loader)
        
        print("{}[Epoch:{}/{}  Avg_train_loss:{:1.3f} Acc_train:{:3.2f}%]".format(("*"*30),epoch+1,EPOCHS,avg_train_loss,train_accuracy))#loss.item()
        print("{}Avg_Loss_boundary:{}".format(("*"*30),avg_loss_boundary))
        
        
        writer.add_scalar('boundary_loss', avg_loss_boundary, epoch+1)
        writer.add_scalar('train_loss', avg_train_loss, epoch+1)
        writer.add_scalar('train_acc', train_accuracy, epoch+1)
        
        with torch.no_grad():
            STN_net.eval()
            Local_net.eval()
            
            val_loss_reg,total_val,step_count,correct_val =0.0, 0,0,0
            loss_boundary_reg_val = 0.0
            for step, (batch_x,label_y) in enumerate(val_Loader):
                              
                val = Variable(batch_x).to(device)
                labels = Variable(label_y).to(device)
                
                batch_images,rois_1,rois_2,theta_x_y = STN_net(val)
                
                outputs_1 = Local_net(rois_1)
                outputs_2 = Local_net(rois_2)
                
                #outputs_1 ,roi ,theta_x_y,outputs_2 = net(val)
                val_loss = ((loss_func(outputs_1,labels)+loss_func(outputs_2,labels))/2)*bb + aa*loss_boundary(theta_x_y)
    
                val_loss_reg +=val_loss.cpu().data
                step_count += 1
                
                loss_boundary_reg_val += loss_boundary(theta_x_y).cpu().data
                
                final_ans = F.softmax(outputs_1, dim=1)+F.softmax(outputs_2, dim=1)
                
                ans=torch.max(final_ans,1)[1].squeeze()
                correct_val += (ans.cpu() == labels.cpu()).float().sum()
            
            val_accuracy = 100 * correct_val / float(len(val_data))
            validation_accuracy.append(val_accuracy)
            
            avg_val_loss = val_loss_reg/len(val_Loader)#step_count
            validation_loss.append(avg_val_loss)
            
            avg_loss_boundary_val = loss_boundary_reg_val/len(val_Loader)
            
            print("{}[Epoch:{}/{}  Avg_val_loss:{:1.3f} Acc_val:{:3.2f}]".format(("*"*30),epoch+1,EPOCHS,avg_val_loss,val_accuracy))
            print("{}Avg_Loss_boundary:{}".format(("*"*30),avg_loss_boundary_val))
            
            
            writer.add_scalar('val_boundary', avg_loss_boundary_val, epoch+1)
            writer.add_scalar('val_loss', avg_val_loss, epoch+1)
            writer.add_scalar('val_acc', val_accuracy, epoch+1)
            
            #if(val_accuracy>=best_acc and avg_loss_boundary_val<=best_loss_boundary  and (epoch+1)>=5):
            if(avg_loss_boundary_val<=best_loss_boundary):
                if(val_accuracy>=best_acc):best_acc=val_accuracy
                best_loss_boundary = avg_loss_boundary_val
                torch.save(STN_net.state_dict(), '{}/save_STN_net_best_{}.{}.pkl'.format(save_file,epoch+1,best_loss_boundary))
                t_fi= open('{}/Best_Acc_{}.txt'.format(save_file,epoch+1),'w')
                t_fi.writelines("best_acc:{}\n".format(best_acc))
                t_fi.writelines("epoch:{} train_acc:{} val_acc:{}\n".format(epoch+1,train_accuracy,val_accuracy))
                t_fi.writelines("train_loss:{} val_loss:{}\n".format(avg_train_loss,avg_val_loss))
                t_fi.writelines("train_boundary:{} val_boundary:{}\n".format(avg_loss_boundary,avg_loss_boundary_val))
                t_fi.close()
            torch.cuda.empty_cache()
        
        print("best_acc:{}%".format(best_acc))
        print("best_loss_boundary:{}".format(best_loss_boundary))
        scheduler.step(val_accuracy)
        lr = optimizer.param_groups[0]['lr']
        print("LR:{}".format(lr))
        loss_plt_show(epoch+1,training_loss,validation_loss,LR,save_file)
        acc_plt_show(epoch+1,training_accuracy,validation_accuracy,LR,save_file)
    
    # #EPOCHS=35
    # loss_plt_show(EPOCHS,training_loss,validation_loss,LR,save_file)
    # acc_plt_show(EPOCHS,training_accuracy,validation_accuracy,LR,save_file)
    torch.save(STN_net.state_dict(), '{}/save_STN_net_{}_{}.pkl'.format(save_file,EPOCHS,avg_loss_boundary_val))
    