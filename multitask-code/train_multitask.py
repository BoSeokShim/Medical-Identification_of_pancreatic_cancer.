import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from utils.util import *
from sklearn.metrics import roc_curve
# import apex
# from apex import amp
from dataset_multitask import get_df_stone, get_transforms, MMC_ClassificationDataset
from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC, MMC_Model, Multitask_MMC_Model

import pandas as pd

Precautions_msg = '(주의사항) ---- \n'


'''
- train.py

모델을 학습하는 전과정을 담은 코드

#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python train.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

pycharm의 경우: 
Run -> Edit Configuration -> train.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

*** def parse_args(): 실행 파라미터에 대한 모든 정보가 있다.  
*** def run(): 학습의 모든과정이 담긴 함수. 이곳에 다양한 trick을 적용하여 성능을 높혀보자. 
** def main(): fold로 나뉜 데이터를 run 함수에 분배해서 실행
* def train_epoch(), def val_epoch() : 완벽히 이해 후 수정

 MMCLab, 허종욱, 2020
 
 기본
 python train.py --kernel-type baseline --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16
 
 세포 그룹화
 python train.py --kernel-type grouping --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --GROUPING
 
 meta 데이터 사용
 python train.py --kernel-type meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --use-meta
 
 세포 그룹화 & meta 데이터 사용
 python train.py --kernel-type grouping_meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --use-meta --GROUPING
 
 모든 데이터 사용
 
 baseline densenet121
 python train.py --kernel-type baseline_densenet121_grouping --k-fold 4 --data-folder original_stone/ --enet-type densenet121 --n-epochs 20 --image-size 256 --batch-size 16 --GROUPING

train_acc_check 
 python train.py --kernel-type baseline_train_acc_check --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 

'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    # kernel_type : 실험 세팅에 대한 전반적인 정보가 담긴 고유 이름


    parser.add_argument('--data-dir', type=str, default='./data/') #'./data/' ./adddataset/
    # base 데이터 폴더 ('./data/')

    parser.add_argument('--data-folder', type=str, required=True)  #'processing/H6c7/
    # 데이터 세부 폴더 예: 'original_stone/'
    # os.path.join(data_dir, data_folder, 'train.csv')

    parser.add_argument('--image-size', type=int, default='256')
    # 입력으로 넣을 이미지 데이터 사이즈

    parser.add_argument('--enet-type', type=str, required=True)
    # 학습에 적용할 네트워크 이름
    # {resnest101, seresnext101,
    #  tf_efficientnet_b7_ns,
    #  tf_efficientnet_b6_ns,
    #  tf_efficientnet_b5_ns...}

    parser.add_argument('--use-amp', action='store_true')
    # 'A Pytorch EXtension'(APEX)
    # APEX의 Automatic Mixed Precision (AMP)사용
    # 기능을 사용하면 속도가 증가한다. 성능은 비슷
    # 옵션 00, 01, 02, 03이 있고, 01과 02를 사용하는게 적절
    # LR Scheduler와 동시 사용에 버그가 있음 (고쳐지기전까지 비활성화)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2309

    parser.add_argument('--use-meta', action='store_true')
    # meta데이터 (사진 외의 나이, 성별 등)을 사용할지 여부

    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    # meta데이터 사용 시 중간레이어 사이즈

    parser.add_argument('--out-dim', type=int, default=2)
    # 모델 출력 output dimension

    parser.add_argument('--DEBUG', action='store_true')
    # 디버깅용 파라미터 (실험 에포크를 5로 잡음)

    parser.add_argument('--GROUPING', action='store_true')
    # 세포 그룹화를 하는지 확인

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    # 학습에 사용할 GPU 번호

    parser.add_argument('--k-fold', type=int, default=4)
    # data cross-validation
    # k-fold의 k 값을 명시

    parser.add_argument('--log-dir', type=str, default='./logs')
    # Evaluation results will be printed out and saved to ./logs/
    # Out-of-folds prediction results will be saved to ./oofs/
    # 분할 했다가 다시 합친 결과

    parser.add_argument('--accumulation_step', type=int, default=2)
    # Gradient accumulation step
    # GPU 메모리가 부족할때, 배치를 잘개 쪼개서 처리한 뒤 합치는 기법
    # 배치가 30이면, 60으로 합쳐서 모델 업데이트함num

    parser.add_argument('--model-dir', type=str, default='./weights')
    # weight 저장 폴더 지정
    # best :

    parser.add_argument('--use-ext', action='store_true')
    # 원본데이터에 추가로 외부 데이터를 사용할지 여부

    parser.add_argument('--batch-size', type=int, default=2) # 배치 사이즈
    parser.add_argument('--num-workers', type=int, default=2) # 데이터 읽어오는 스레드 개수
    parser.add_argument('--init-lr', type=float, default=3e-5) # 초기 러닝 레이트. pretrained를 쓰면 매우 작은값
    parser.add_argument('--n-epochs', type=int, default=100) # epoch 수

    args, _ = parser.parse_known_args()
    return args


def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    PROBS = []
    TARGETS = []
    bar = tqdm(loader)
    for i, (data, target, target2,image_name) in enumerate(bar):

        optimizer.zero_grad()
        
        if args.use_meta:
            data, meta = data
            data, meta, target, target2 = data.to(device), meta.to(device), target.to(device), target2.to(device)
            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
            logits, logits2 = model(data, meta)
            probs += logits.softmax(1)
        else:
            data, target = data.to(device), target.to(device)
            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)

            logits = model(data)

            probs += logits.softmax(1)


        loss = 0.8 * criterion(logits, target) + 0.2 * criterion(logits2, target2)

        if not args.use_amp:
            loss.backward()


        if args.image_size in [896,576]:
            # 그라디언트가 너무 크면 값을 0.5로 잘라준다 (max_grad_norm=0.5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # gradient accumulation (메모리 부족할때)
        if args.accumulation_step:
            if (i + 1) % args.accumulation_step == 0:
                optimizer.step()
                #optimizer.zero_grad()
        else:
            optimizer.step()
            #optimizer.zero_grad()

        loss_np = loss.detach().cpu().numpy()
        PROBS.append(probs.detach().cpu())
        TARGETS.append(target.detach().cpu())
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    train_acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    train_loss = np.mean(train_loss)
    return train_loss, train_acc


def val_epoch(model, loader, target_idx, is_ext=None, n_test=1, get_output=False):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation

    '''

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    PATIENT_ID = []

    with torch.no_grad():
        for (data, target, patient_id, image_name,cancer_id) in tqdm(loader):  # ,patient_id

            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l, _ = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)

                patient_id = list(patient_id)
                # change = np.zeros((len(target), args.out_dim))
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)

                # for i in range(len(target)):
                #     change[i, :] = (list(format(target[i], 'b').zfill(5)))
                #
                # target = torch.tensor(change).float().to(device)

                for I in range(n_test):
                    l, _ = model(get_trans(data, I))
                    logits += l

                    # probs += m(l)
                    probs += l.softmax(1)

            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            # PATIENT_ID.append(patient_id.detach().cpu())
            patient_id = list(patient_id)
            PATIENT_ID.append(patient_id)

            # loss = criterion(m(logits), target)
            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    PATIENT_ID = sum(PATIENT_ID, [])
    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    # PATIENT_ID = torch.cat(PATIENT_ID).numpy()
    PATIENT_ID = np.array(PATIENT_ID)
    #
    #
    unique_idx = np.unique(PATIENT_ID)  # .astype(np.int64)

    if args.GROUPING:
        for u_id in unique_idx.tolist():
            res_list = np.where(PATIENT_ID == u_id)
            save_prob = PROBS[res_list]
            mean_prob = save_prob.mean(axis=0)
            PROBS[res_list] = mean_prob

    if get_output:
        return LOGITS, PROBS
    else:
        # accuracy : 정확도
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

        # AUC : ROC 아래 넓이
        auc = roc_auc_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx])

        fpr, tpr, threshold = roc_curve((TARGETS == target_idx).astype(float), PROBS[:, target_idx])

        # acc = ((PROBS > 0.5).astype(np.float64) == TARGETS).mean() * 100.
        # acc = (PROBS.argmax(1) == TARGETS).mean() * 100
        return val_loss, acc, auc, auc,  fpr, tpr, threshold

def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, target_idx, meta_columns):
    '''
    학습 진행 메인 함수

    :param fold: cross-validation에서 valid에 쓰일 분할 번호
    :param df: DataFrame 학습용 전체 데이터 목록
    :param meta_features, n_meta_features: 이미지 외 추가 정보 사용 여부
    :param transforms_train, transforms_val: 데이터셋 transform 함수
    :param target_idx:
    '''

    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
        # batch_normalization에서 배치 사이즈 1인 경우 에러 발생할 수 있으므로, 데이터 한개 버림
        if len(df_train) % args.batch_size == 1:
            df_train = df_train.sample(len(df_train)-1)
        if len(df_valid) % args.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid)-1)


    # 데이터셋 읽어오기
    dataset_train = MMC_ClassificationDataset(df_train, 'train', meta_features, transform=transforms_train, meta_columns=meta_columns)
    dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', meta_features, transform=transforms_val, meta_columns=meta_columns)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim,
        pretrained=True
    )

    model = model.to(device)
    
    

    auc_max = 0.
    auc_no_ext_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
    model_file2 = os.path.join(args.model_dir, f'{args.kernel_type}_best_no_ext_fold{fold}.pth')
    model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)

    if DP:
        model = nn.DataParallel(model)

    # amp를 사용하면 버그 (use_amp 비활성화)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')

        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc, auc_no_ext,  fpr, tpr, threshold = val_epoch(model, valid_loader, target_idx, is_ext=df_valid['is_ext'].values)

        if args.use_ext:
            content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train acc: {train_acc:.4f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, Acc: {(acc):.4f}, AUC: {(auc):.6f}, AUC_no_ext: {(auc_no_ext):.6f}.'
        else:
            content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train acc: {train_acc:.4f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, Acc: {(acc):.4f}, AUC: {(auc):.6f}.'

        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()    
        if epoch == 2:
            scheduler_warmup.step() # bug workaround
            
        if auc > auc_max:
            print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            torch.save(model.state_dict(), model_file)
            auc_max = auc

        if auc < 0.05:
            dict = {'fpr': fpr, 'tpr': tpr}
            df_dict = pd.DataFrame(dict)
            df_dict.to_csv(f'./auc={auc}.csv')


        # 외부데이터를 사용할 경우, 외부데이터를 제외한 모델을 따로 저장한다.
        if args.use_ext:
            if auc_no_ext > auc_no_ext_max:
                print('auc_no_ext_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_no_ext_max, auc_no_ext))
                torch.save(model.state_dict(), model_file2)
                auc_no_ext_max = auc_no_ext

    #torch.save(model.state_dict(), model_file3)


def main():

    '''
    ####################################################
    # stone data 데이터셋 : dataset.get_df_stone
    ####################################################
    '''
    df_train, df_test, meta_features, n_meta_features, target_idx, meta_columns = get_df_stone(
        k_fold = args.k_fold,
        out_dim = args.out_dim,
        data_dir = args.data_dir,
        data_folder = args.data_folder,
        use_meta = args.use_meta,
        use_ext = args.use_ext
    )

    # 모델 트랜스폼 가져오기
    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = range(args.k_fold)

    for fold in folds:
        run(fold, df_train, meta_features, n_meta_features, transforms_train, transforms_val, target_idx, meta_columns)


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # argument값 만들기
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # 네트워크 타입 설정
    if args.enet_type == 'resnest101':
        ModelClass = Resnest_MMC
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_MMC
    elif 'efficientnet' in args.enet_type:
        ModelClass = Multitask_MMC_Model
    elif args.enet_type == 'densenet121':
        ModelClass = Effnet_MMC
    elif args.enet_type == 'vgg19':
        ModelClass = Effnet_MMC
    elif args.enet_type == 'resnet101':
        ModelClass = Effnet_MMC
    elif args.enet_type == 'vit':
        ModelClass = MMC_Model
    else:
        raise NotImplementedError()

    # GPU가 여러개인 경우 멀티 GPU를 사용함
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # 실험 재현을 위한 random seed 부여하기
    set_seed(2359)
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    # 메인 기능 수행
    main()