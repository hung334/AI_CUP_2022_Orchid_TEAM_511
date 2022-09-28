# AI_CUP_2022_Orchid


先將 orchid_private_set.zip 和 orchid_public_set.zip 解壓縮至
./AI_CUP_2022_Orchid_TEAM_511/AI_CUP_2022_test/orchid_set/內做存放

**File Structure**

    ├── orchid_set
	    	├──orchid_public_set
		│     ├──*.jpg
		│     │
		│     │  ...
		│     │
		│     ├──*.jpg
		│    
		└── orchid_private_set
			  ├──*.jpg
			  │
			  │  ...
			  │
			  ├──*.jpg



**1.在Colab上掛載Google Drive**

``` {.python}
from google.colab import drive
drive.mount("/content/drive")
```
**2.進入到文件所在的目錄**

``` {.python}
import os
path = "/content/drive/My Drive/AI_CUP_2022_Orchid_TEAM_511"
os.chdir(path)
os.listdir(path)
```

**3.需安裝套件:** 

`pip install timm==0.6.2.dev0`

``` {.python}
pip install timm==0.6.2.dev0
```


**4.執行 Final_Test.py 即可輸出答案，答案會儲存在(./Ans)**

`!python Final_Test.py`

*＃注意
本隊伍在實驗階段時有發現到Pillow的版本若非當初訓練時的版本(8.3.1)，在輸出結果上會有些許誤差。*

``` {.python}
!python Final_Test.py
```


**5.如何執行 train.py**

**5-1 進入pytorch-image-models 目錄下**
``` {.python}
%cd pytorch-image-models
```
模型訓練的權重皆存放在
(./AI_CUP_2022_Orchid_TEAM_511/pytorch-image-models/output)

**5-2 訓練 swin large**

--experiment 儲存訓練實驗的檔案夾名稱

`!CUDA_LAUNCH_BLOCKING=1 python train.py ../Datasets/training --train-split train --val-split val --model swin_large_patch4_window12_384_in22k --pretrained --num-classes 219 --mean 0.5075337404927281 0.45864544276917535 0.4169235386212412 --std 0.2125643051799512 0.2385082849964861 0.22386483801695406 -b 4 -vb 4 --opt adamw --weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 300 --warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --recount 1 --remode pixel --cutmix 0.4 --train-interpolation bicubic --drop-path 0.1 -j 4 --save-images --output output --experiment 0`

``` {.python}
!CUDA_LAUNCH_BLOCKING=1 python train.py ../Datasets/training --train-split train --val-split val --model swin_large_patch4_window12_384_in22k --pretrained --num-classes 219 --mean 0.5075337404927281 0.45864544276917535 0.4169235386212412 --std 0.2125643051799512 0.2385082849964861 0.22386483801695406 -b 4 -vb 4 --opt adamw --weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 300 --warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --recount 1 --remode pixel --cutmix 0.4 --train-interpolation bicubic --drop-path 0.1 -j 4 --save-images --output output --experiment 0
```

**5-2-1 Fine tune swin large**

`!CUDA_LAUNCH_BLOCKING=1 python train.py ../Datasets/training --train-split train --val-split val --model swin_large_patch4_window12_384_in22k --initial-checkpoint ./model/0/model_best.pth.tar --num-classes 219 --mean 0.5075337404927281 0.45864544276917535 0.4169235386212412 --std 0.2125643051799512 0.2385082849964861 0.22386483801695406 -b 2 -vb 2 --opt adamw --weight-decay 0.01 --sched step --lr 0.00001 --decay-rate 1.0 --no-aug --epochs 100 --warmup-epochs 0 --color-jitter 0.0 --reprob 0.0 --recount 1 --remode pixel --cutmix 0.0 --train-interpolation bicubic --drop-path 0.0 --log-interval 30 -j 4 --save-images --output output --experiment 0_no_aug_no_weight_decay`

**5-3-1 訓練Swin v2 base (A-fold)**

`
!CUDA_LAUNCH_BLOCKING=1 python train.py ../Datasets/training
--train-split train --val-split val --model
swinv2_base_window12to24_192to384_22kft1k --pretrained --num-classes
219 --img-size 384 --mean 0.5075337404927281 0.45864544276917535
0.4169235386212412 --std 0.2125643051799512 0.2385082849964861
0.22386483801695406 -b 2 -vb 2 --opt adamw --weight-decay 0.01
--layer-decay 0.65 --sched cosine --lr 0.0001 --lr-cycle-limit 1
--warmup-lr 1e-5 --min-lr 1e-5 --epochs 300 --warmup-epochs 5
--color-jitter 0.5 --reprob 0.5 --recount 1 --remode pixel --cutmix
0.4 --train-interpolation bicubic --drop-path 0.1 -j 4 --save-images
--output output --experiment
0531_0310_swinv2_base_window12to24_192to384_22kft1k_official_aug_v0_new`


**5-3-2 訓練Swin v2 base (B-fold)**


`!CUDA_LAUNCH_BLOCKING=1 python train.py
../Datasets/AI_CUP_fold_complement --train-split train
--val-split val --model swinv2_base_window12to24_192to384_22kft1k
--pretrained --num-classes 219 --mean 0.45625534556274666
0.4220624936173144 0.3649616601198825 --std 0.2143212828816861
0.2210437745632625 0.2062174242104951 -b 2 -vb 2 --opt adamw
--weight-decay 0.01 --layer-decay 0.65 --sched cosine --lr 0.0001
--lr-cycle-limit 2 --warmup-lr 1e-5 --min-lr 1e-5 --epochs 300
--warmup-epochs 5 --color-jitter 0.5 --reprob 0.5 --recount 1
--remode pixel --cutmix 0.4 --train-interpolation bicubic
--drop-path 0.1 -j 4 --save-images --output output --experiment
0603_0131_swinv2_base_window12to24_192to384_22kft1k_complement_final_bacbbone_new
--mixup-off-epoch 300`

**指定單張GPU**

`CUDA_VISIBLE_DEVICES=1 bash ./distributed_train.sh 1`

**指定多張特定GPU**

如:有3張時，只用其中的0，2

`CUDA_VISIBLE_DEVICES=0,2 bash ./distributed_train.sh 3`
