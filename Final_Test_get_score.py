from PIL import Image
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F 
import os
import shutil
from timm import create_model
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore')
import math
import pandas as pd

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

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def Get_STN(pkl_file,num_classes=219,device="cuda"):
    model_name = "resnet50"
    STN_local = create_model(model_name, pretrained=False,num_classes=num_classes)
    num_ftrs = STN_local.fc.in_features
    STN_local.fc = torch.nn.Sequential(nn.Linear(num_ftrs, 256),
                                       nn.Tanh(),
                                        nn.Linear(256, 4), 
                                        nn.Tanh(),)
    STN_net = STN(loc_model=STN_local,device = device)
    STN_net.load_state_dict(torch.load(pkl_file,map_location=device))
    return STN_net

def Get_Model(model_name,pkl_path,num_classes=219):
    model = create_model(model_name, pretrained=False,num_classes=num_classes, checkpoint_path=pkl_path)
    return model

if __name__ == '__main__':

    #template = pd.read_csv('./submission_template.csv')
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    val_path = r"./Datasets/training/val"
    #val_path = r"./Datasets/AI_CUP_fold_complement/val"
    
    orchid_test_set = val_path
    
    #******************************************************************************************************************************
    BATCH_SIZE = 1
    Num_workers = 2
    
    resize = 384
    mean_384_one_fold = [0.5075337404927281 ,0.45864544276917535 ,0.4169235386212412] 
    std_384_one_fold = [0.2125643051799512 ,0.2385082849964861 ,0.22386483801695406]
    mean_384_two_fold = [0.45625534556274666, 0.4220624936173144, 0.3649616601198825]
    std_384_two_fold = [0.2143212828816861, 0.2210437745632625, 0.2062174242104951]
    
    mean = mean_384_one_fold
    std = std_384_one_fold
    
    tfm = transforms.Compose([
        transforms.Resize(size=(resize,resize)),#(h,w)
        transforms.ToTensor(),
    ])
    tfm_c = transforms.Compose([
        transforms.Resize(resize, interpolation=Image.BICUBIC),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
    ])
    
    
    tfm_one = transforms.Normalize(mean_384_one_fold, std_384_one_fold)
    tfm_two = transforms.Normalize(mean_384_two_fold, std_384_two_fold)
    
    orchid_test_path = ImageFolderWithPaths(orchid_test_set,transform=tfm_c)
    orchid_test_Loader_path = torch.utils.data.DataLoader(dataset=orchid_test_path,batch_size=BATCH_SIZE,shuffle=False,num_workers=Num_workers)
    
    val_data = datasets.ImageFolder(val_path)
    print(val_data.classes)
    #*************************************************************************************************************************************************
    one_fold_STN_pkl = "./pytorch-image-models/save_STN/one_fold_save_STN_net_best_37.3.454599573160522e-05.pkl"
    two_fold_STN_pkl = "./pytorch-image-models/save_STN/two_fold_save_STN_net_best_18.2.972046786453575e-05.pkl"
    
    model_name = "swinv2_base_window12to24_192to384_22kft1k"
    one_fold_Network_pkl = "./pytorch-image-models/output/0531_0310_swinv2_base_window12to24_192to384_22kft1k_official_aug_v0/model_best.pth.tar"
    two_fold_Network_pkl = "./pytorch-image-models/output/0603_0131_swinv2_base_window12to24_192to384_22kft1k_complement_final_bacbbone_95.5927_219epoch/model_best.pth.tar"
    
    one_fold_STN_Network_pkl = "./pytorch-image-models/output/STN_backbone_official_86_new_V2/checkpoint-37.pth.tar"
    two_fold_STN_Network_pkl = ""
    
    Swin_L_pkl = "./pytorch-image-models/output/0__swin_large__official_95.2887/model_best.pth.tar"
    Swin_L_model = Get_Model('swin_large_patch4_window12_384_in22k',pkl_path=Swin_L_pkl)
    Swin_L_model = nn.DataParallel(Swin_L_model)
    Swin_L_model = Swin_L_model.cuda()
    
    
    STN_one_model = Get_STN(pkl_file=one_fold_STN_pkl)
    STN_two_model = Get_STN(pkl_file=two_fold_STN_pkl)
    
    Network_one_model = Get_Model(model_name,pkl_path=one_fold_Network_pkl)
    Network_two_model = Get_Model(model_name,pkl_path=two_fold_Network_pkl)
    
    STN_Network_one_model = Get_Model(model_name,pkl_path=one_fold_STN_Network_pkl)
    
    STN_one_model = nn.DataParallel(STN_one_model)
    STN_one_model = STN_one_model.to(device)#.cuda()
    STN_two_model = nn.DataParallel(STN_two_model)
    STN_two_model = STN_two_model.to(device)#.cuda()
    
    Network_one_model = nn.DataParallel(Network_one_model)
    Network_one_model = Network_one_model.to(device)#.cuda()
    Network_two_model = nn.DataParallel(Network_two_model)
    Network_two_model = Network_two_model.to(device)#.cuda()
    
    STN_Network_one_model = nn.DataParallel(STN_Network_one_model)
    STN_Network_one_model = STN_Network_one_model.to(device)#.cuda()
    
    #*************************************************************************************************************************************************
    label = []
    predicted = []
    ok_count = 0
    correct_count,error_count = 0,0
    with torch.no_grad():
            STN_one_model.eval()
            STN_two_model.eval()
            
            Network_one_model.eval()
            Network_two_model.eval()
            
            STN_Network_one_model.eval()
            
            Swin_L_model.eval()
            for step, (batch_x,label_y,path) in enumerate(orchid_test_Loader_path):
                img_file = os.path.split(path[0])[1]
                print(img_file)
                test_one = Variable(tfm_one(batch_x)).to(device)#.cuda()
                test_two = Variable(tfm_two(batch_x)).to(device)#.cuda()
                
                roi_0_one,roi_1_one,roi_2_one,_ = STN_one_model(test_one)
                #roi_0_two,roi_1_two,roi_2_two,_ = STN_two_model(test_two)
                
                outpust_0_one = Network_one_model(roi_0_one)
                outpust_1_one = STN_Network_one_model(roi_1_one)
                outpust_2_one = STN_Network_one_model(roi_2_one)
                
                #outpust_0_two = Network_two_model(roi_0_two)
                #outpust_1_two = Network_two_model(roi_1_two)
                #outpust_2_two = Network_two_model(roi_2_two)
                
                #output_L = Swin_L_model(test_one)
                
                #final_outputs = F.softmax(output_L, dim=1)
                
                #final_outputs = F.softmax(outpust_1_one, dim=1)+F.softmax(outpust_2_one, dim=1)
                
                final_outputs = F.softmax(outpust_0_one, dim=1)+F.softmax(outpust_1_one, dim=1) + F.softmax(outpust_2_one, dim=1)
                
                #final_outputs = F.softmax(outpust_0_one, dim=1)+F.softmax(outpust_1_one, dim=1)+F.softmax(outpust_2_one, dim=1)
                #            F.softmax(outpust_0_two, dim=1)+F.softmax(outpust_1_two, dim=1)+F.softmax(outpust_2_two, dim=1)
                            
                ans = torch.max(final_outputs,1)[1].squeeze()
                final_ans = val_data.classes[int(ans)]
                #if(test_data.classes[int(ans)] == class_orchid):
                label.append(int(val_data.classes[int(label_y)]))
                predicted.append(int(ans))
                if(int(ans)== int(val_data.classes[int(label_y)])):
                    correct_count+=1
                else:
                    error_count+=1
                    
    print('correct_count:{},  {:.2f}%'.format(correct_count,(correct_count/len(val_data)*100)))
    print('error_count:{},    {:.2f}%'.format(error_count,(error_count/len(val_data)*100)))

    Accuracy = accuracy_score(label, predicted)#(tp+tn)/(tp+fp+fn+tn)
    Precision = precision_score(label, predicted,average=None) #tp/(tp+fp)
    Recall = recall_score(label, predicted,average=None)#tp/(tp+fn)
    F1 = f1_score(label, predicted,average=None)#2 / ( (1/ Precision) + (1/ Recall) )
    
    print("Accuracy:{}".format(Accuracy))
    print("Precision:{}".format(Precision.mean()))
    print("Recall :{}".format(Recall.mean()))
    print("F1:{}".format(F1.mean()))
    final_score = 0.5*Accuracy + 0.5*F1.mean()
    print("Final_score :{}".format(final_score))
    
    #img_file = "dmlg79vsu0.jpg"
    #model_ans = 1
    #template.loc[template['filename']==img_file,'category']=model_ans
    #template.to_csv("outpust.csv", index=False)