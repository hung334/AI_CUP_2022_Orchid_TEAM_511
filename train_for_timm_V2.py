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
import random
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def get_score(label,predicted):
    
    Accuracy = accuracy_score(label, predicted)  #(tp+tn)/(tp+fp+fn+tn)
    Precision = precision_score(label, predicted,average=None)  #tp/(tp+fp)
    Recall = recall_score(label, predicted,average=None)  #tp/(tp+fn)
    F1 = f1_score(label, predicted,average=None)  #2 / ( (1/ Precision) + (1/ Recall) )
    #final_score = 0.5*Accuracy + 0.5*F1.mean()
    
    print("Accuracy:{}".format(Accuracy))
    print("Precision:{}".format(Precision.mean()))
    print("Recall :{}".format(Recall.mean()))
    print("F1:{}".format(F1.mean()))
    #print("Final_score :{}".format(final_score))
    
    return Precision.mean(),Recall.mean(),F1.mean()

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

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

def acc_plt_show(num_epochs,training_accuracy,validation_accuracy,LR,save_file):
    plt.figure()
    plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
    plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
    plt.title('Train & Val accuracy,epoch:{}'.format(num_epochs))
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("{}/{}_acc.jpg".format(save_file,LR))
    plt.show()

def loss_plt_show(num_epochs,training_loss,validation_loss,LR,save_file):
    plt.figure()
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
    
    BATCH_SIZE = 10
    LR =0.0001
    EPOCHS = 150
    
    #******************************************************************************************
    #train_path=r"./Datasets/External_Orchid_Datasets/train-en"
    #dev_path=r"./Datasets/External_Orchid_Datasets/test-en"
    
    #train_path=r"./Datasets/training/train"
    train_path=r"./dataset_2022-06-06/train"
    dev_path=r"./dataset_2022-06-06/validation"
    
    #train_path=r"./Datasets/split_training_6448/train"
    #dev_path=r"./Datasets/split_training_6448/val"
    
    resize = 256
    
    #mean = [0.36715875, 0.42368276, 0.48109331]
    #std = [0.21691893, 0.2164323 , 0.2139854 ]
    
    #mean_384 = [0.4812, 0.4238, 0.3673]
    #std_384 = [0.2462, 0.2414, 0.2489]
    
    mean = [0.5702, 0.5389, 0.4934]
    std = [0.2376, 0.2327, 0.2393]
    
    #mean = [0.26117493, 0.39152832, 0.38936958]
    #std = [0.18968344, 0.19150229, 0.20256462]
    
    train_aug = transforms.Compose([
        transforms.Resize(size=(resize,resize)),#(h,w)
        #transforms.RandomCrop(size=(256,256), padding=5),
        transforms.ColorJitter(brightness=0.5,saturation=0.5),
        #transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        #transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        #transforms.RandomRotation((-45,45)),#隨機角度旋轉 , expand=True
        #transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    
    train_data = datasets.ImageFolder(train_path,transform=train_aug)
    val_data = datasets.ImageFolder(dev_path,transform=transforms.Compose([
        transforms.Resize(size=(resize,resize)),#(h,w)
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))
    print(train_data.classes)#获取标签
    print(train_data.class_to_idx)
    train_Loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
    val_Loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    #*******************************************************************************************

    model_name = "swinv2_tiny_window16_256"
    net = create_model(model_name, pretrained=True,num_classes=len(train_data.classes))
    print(net)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    Cuda_Parallel = 0
    
    device_ids = [0, 1]
    if(Cuda_Parallel):
        net = torch.nn.DataParallel(net,device_ids=device_ids)
        net.cuda()
    else:
        net.to(device)
    

    #************************************************************************************************************

    #optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=6e-4)
    #optimizer =torch.optim.RMSprop(net.parameters(), lr=LR)#, alpha=0.9)
    #optimizer =torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizer =torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    # optimizer = torch.optim.SGD([{'params':net.fc.parameters()},
    #                              {'params':net.layer4[1:3].parameters()}], 
    #                             lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5, patience=10, verbose=True)
    #loss_func = FocalLoss()
    loss_func = LabelSmoothingLossCanonical(smoothing=0.1)#torch.nn.CrossEntropyLoss()
    #loss_func = torch.nn.CrossEntropyLoss()
    #*************************************************************************************************************
    
    training_loss,validation_loss=[],[]
    training_accuracy,validation_accuracy =[],[]
    best_acc,best_epoch,best_score=0,0,0
    
    
    special_case="no_contrast"#"adam_crossen"
    save_file="./save/save_{}_{}".format(model_name,special_case)
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
        
    
    t_fi= open('{}/config.txt'.format(save_file),'w')
    t_fi.writelines("LR:{}\n".format(LR))
    t_fi.writelines("BATCH_SIZE:{}\n".format(BATCH_SIZE))
    t_fi.writelines("transform:{}\n".format(train_aug))
    #t_fi.writelines("resize:{}\n".format(resize))
    #t_fi.writelines("mean:{}\n".format(mean))
    #t_fi.writelines("std:{}\n".format(std))
    t_fi.writelines("train_path:{}\nval_path:{}\n".format(train_path,dev_path))
    t_fi.writelines("model_name:{}\nCuda_Parallel:{}\n".format(model_name,Cuda_Parallel))
    t_fi.writelines("optimizer:{}\n".format(optimizer))
    t_fi.writelines("loss:{}".format(loss_func))
    t_fi.close()
    
    
    for epoch in range(EPOCHS):
        net.train()
        train_loss_reg,total_train,step_count,correct_train =0.0, 0,0,0

        
        for step, (batch_x,label_y) in enumerate(train_Loader):
            
            if(Cuda_Parallel):
                train = Variable(batch_x).cuda()
                labels = Variable(label_y).cuda()
            else:
                train = Variable(batch_x).to(device)
                labels = Variable(label_y).to(device)
            
            outputs = net(train)
            train_loss = loss_func(outputs,labels)
            
            optimizer.zero_grad()               # clear gradients for this training step
            train_loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            
            train_loss_reg +=train_loss.cpu().data
            
            ans=torch.max(outputs,1)[1].squeeze()
            correct_train += (ans.cpu() == labels.cpu()).float().sum()
            
            if(step%100==0):
                print("Epoch:{}/{} Step:{}/{} Train_loss:{:1.3f} ".format(epoch+1,EPOCHS,step+1,len(train_Loader),train_loss))
            
        train_accuracy = 100 * correct_train / float(len(train_data))
        training_accuracy.append(train_accuracy)
        
        avg_train_loss = train_loss_reg/len(train_Loader)
        training_loss.append(avg_train_loss)
        print("{}[Epoch:{}/{}  Avg_train_loss:{:1.3f} Acc_train:{:3.2f}%]".format(("*"*30),epoch+1,EPOCHS,avg_train_loss,train_accuracy))#loss.item()
        
        label = []
        predicted = []
        with torch.no_grad():
            net.eval()
            val_loss_reg,total_val,step_count,correct_val =0.0, 0,0,0
            for step, (batch_x,label_y) in enumerate(val_Loader):
                if(Cuda_Parallel):
                    val = Variable(batch_x).cuda()
                    labels = Variable(label_y).cuda()
                else:
                    val = Variable(batch_x).to(device)
                    labels = Variable(label_y).to(device)
                outputs = net(val)
                val_loss = loss_func(outputs,labels)
    
                val_loss_reg +=val_loss.cpu().data
                step_count += 1
                
                ans=torch.max(outputs,1)[1].squeeze()
                correct_val += (ans.cpu() == labels.cpu()).float().sum()
                

                for i in np.atleast_1d(labels.cpu().data.numpy()):
                    label.append(i)

                for i in np.atleast_1d(ans.cpu().data.numpy()):
                    predicted.append(i)
            
            val_accuracy = 100 * correct_val / float(len(val_data))
            validation_accuracy.append(val_accuracy)
            
            avg_val_loss = val_loss_reg/len(val_Loader)#step_count
            validation_loss.append(avg_val_loss)
            
            print("{}[Epoch:{}/{}  Avg_val_loss:{:1.3f} Acc_val:{:3.2f}%]".format(("*"*30),epoch+1,EPOCHS,avg_val_loss,val_accuracy))
            
            Precision,Recall,F1 = get_score(label,predicted)
            #print("final_score:{}".format(final_score))
            
            if(val_accuracy>=best_acc ):#and final_score>=best_score):
                best_acc=val_accuracy
                #best_score=final_score
                torch.save(net.state_dict(), '{}/save_{}_best_{}_{}.pkl'.format(save_file,model_name,epoch+1,val_accuracy))
                t_fi= open('{}/Best_Acc_{}_{}.txt'.format(save_file,epoch+1,val_accuracy),'w')
                t_fi.writelines("epoch:{}/{} train_acc:{} val_acc:{}\n".format(epoch+1,EPOCHS,train_accuracy,best_acc))
                t_fi.writelines("train_loss:{} val_loss:{}\n".format(avg_train_loss,avg_val_loss))
                t_fi.writelines("Precision:{}\n".format(Precision))
                t_fi.writelines("Recall:{}\n".format(Recall))
                t_fi.writelines("F1:{}".format(F1))
                t_fi.close()
            #torch.cuda.empty_cache()
        
        print("best_acc:{}%".format(best_acc))
        #print("best_score:{}%".format(best_score))
        scheduler.step(val_accuracy)
        lr = optimizer.param_groups[0]['lr']
        print("LR:{}".format(lr))
        loss_plt_show(epoch+1,training_loss,validation_loss,LR,save_file)
        acc_plt_show(epoch+1,training_accuracy,validation_accuracy,LR,save_file)
        plt.close('all')
    # #EPOCHS=35
    # loss_plt_show(EPOCHS,training_loss,validation_loss,LR,save_file)
    # acc_plt_show(EPOCHS,training_accuracy,validation_accuracy,LR,save_file)
    torch.save(net.state_dict(), '{}/save_{}_{}_{}.pkl'.format(save_file,model_name,EPOCHS,val_accuracy))