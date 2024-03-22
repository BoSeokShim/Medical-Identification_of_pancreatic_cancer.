import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
import torch.nn as nn
from dataset_multitask import get_df_stone, get_transforms, MMC_ClassificationDataset
from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC, MMC_Model, Multitask_MMC_Model
from utils.util import *
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from ptflops import get_model_complexity_info
from utils.torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from flopco import FlopCo
from itertools import groupby
import datetime
import shutil

Precautions_msg = ' '

'''
- evaluate.py

학습한 모델을 평가하는 코드
Test셋이 아니라 학습때 살펴본 validation셋을 활용한다. 
grad-cam 실행한다. 


#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python evaluate.py --kernel-type 4fold_b3_300_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30

3d project 경우 밑 코드 실행
python evaluate.py --kernel-type 4fold_b3_10ep_alb01 --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --n-epochs 10 --image-size 300

pycharm의 경우: 
Run -> Edit Configuration -> evaluate.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b5_ns --n-epochs 30 --k-fold 4



edited by MMCLab, 허종욱, 2020

 기본
 python evaluate.py --kernel-type baseline --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16
 
 세포 그룹화
 python evaluate.py --kernel-type grouping --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --GROUPING
 
 meta 데이터 사용
 python evaluate.py --kernel-type meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --use-meta
 
 세포 그룹화 & meta 데이터 사용
 python evaluate.py --kernel-type grouping_meta --k-fold 4 --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 20 --image-size 256 --batch-size 16 --use-meta --GROUPING
 
 모든 데이터 사용
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, default='original_stone/')
    parser.add_argument('--image-size', type=int, default = 256)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--use-ext', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--oof-dir', type=str, default='./oofs')

    parser.add_argument('--k-fold', type=int, default=4)
    parser.add_argument('--eval', type=str, choices=['best', 'best_no_ext', 'final'], default="best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--thresholds', type=float, default=0.5)  # PROBS에서 양성 판단 임계값 (acc 측정할때 사용)
    parser.add_argument('--select', type=int, default=300)  # sorting된 confidence csv에서 가져올 이미지 개수
    parser.add_argument('--GROUPING', action='store_true')  # 세포 그룹화를 하는지 확인

    args, _ = parser.parse_known_args()
    return args



def confidence_score(PROBS, TARGETS, PATIENT_ID, IMAGE_NAME, optimal_threshold):

    confidence_dict = {'image_name': list(np.concatenate(IMAGE_NAME)), 'patient_id': list(np.concatenate(PATIENT_ID)),
                       'prob': np.concatenate(PROBS)[:, 1],
                       'target': np.concatenate(TARGETS)
                       }

    # 정답을 맞춘 부분만 probs 표기, 이 외에는 모두 0
    # 1만 맞췄을때, 0과 1 모두 맞췄을 때 두가지 경우 존재 (한 level만 맞아도 정답으로 취급할땐, ==대신 or 연산자 사용)
    confidence_dict['correct_prob'] = (confidence_dict['prob'] > optimal_threshold) * confidence_dict['target'] * confidence_dict['prob']

    confidence_dict['predict'] = (confidence_dict['prob'] > optimal_threshold).astype(int)
    confidence_dict['true/false'] = (confidence_dict['predict']==confidence_dict['target']).astype(int)

    # dict로 dataframe 만들고 confidence_sum기준으로 sorting하여 저장
    confidence_csv = pd.DataFrame(confidence_dict)
    # confidence_csv = confidence_csv.sort_values(by=['correct_prob'], axis=0, ascending=False)
    os.makedirs(f'./confidence/{args.kernel_type}_confidence', exist_ok=True)
    confidence_csv.to_csv(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_confidence.csv')





def sorted_confidence_save(select):
    os.makedirs(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_high_confidence_image', exist_ok=True)
    # csv 만듬
    confidence_csv = pd.read_csv(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_confidence.csv')
    confidence_csv[:select].to_csv(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_high_confidence.csv', index=False)
    high_confidence = confidence_csv[['image_name']][:select].squeeze().tolist()

    for img_name in high_confidence: # 사진 저장
        shutil.copyfile(f'./data/original_stone/train/{img_name}', f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_high_confidence_image/{img_name}')



def plot_roc_curve(fper, tper, fold, optimal_threshold,auc):
    os.makedirs(f'./roc_curve_image/{args.kernel_type}_roc_curve_image', exist_ok=True)
    plt.figure()
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'fold: {fold}, Opt_t: {optimal_threshold:.5f}, auc: {auc:.4f}')
    plt.legend()

    plt.savefig(f'./roc_curve_image/{args.kernel_type}_roc_curve_image/{args.kernel_type}_{fold}.png')
    # plt.show()

    if fold == "All":
        tpr_fpr_dict = {'fpr': fper, 'tpr': tper,
                           'optimal_threshold': optimal_threshold,
                           }
        tpr_fpr_csv = pd.DataFrame(tpr_fpr_dict)
        tpr_fpr_csv.to_csv(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_tpr_fpr.csv')

def val_epoch(model, loader, target_idx, fold, is_ext=None, n_test=1, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    PATIENT_ID = []
    IMAGE_NAME = []
    CANCER = []

    with torch.no_grad():
        for (data, target, patient_id, image_name, cancer_id) in tqdm(loader):

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
                #change = np.zeros((len(target), args.out_dim))
                patient_id = list(patient_id)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)

                # for i in range(len(target)):
                #     change[i, :] = (list(format(target[i], 'b').zfill(5)))
                #
                # target = torch.tensor(change).float().to(device)

                for I in range(n_test):
                    l, _ = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
                    # probs += m(l)

            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            patient_id = list(patient_id)
            PATIENT_ID.append(patient_id)
            IMAGE_NAME.append(image_name)
            CANCER.append(cancer_id)

            # loss = criterion(m(logits), target)
            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    PATIENT_ID = sum(PATIENT_ID, [])
    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    PATIENT_ID = np.array(PATIENT_ID)
    IMAGE_NAME = np.concatenate(IMAGE_NAME)
    CANCER = np.concatenate(CANCER)
    #PATIENT_ID = torch.cat(PATIENT_ID).numpy()

    unique_idx = np.unique(PATIENT_ID) #.astype(np.int64)

    if args.GROUPING:
        for u_id in unique_idx.tolist():
            res_list = np.where(PATIENT_ID == u_id)
            save_prob = PROBS[res_list]
            mean_prob = save_prob.mean(axis=0)
            PROBS[res_list] = mean_prob


    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    # roc-curve
    fpr, tpr, threshold = roc_curve((TARGETS == target_idx).astype(float), PROBS[:, target_idx])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]

    # f1-score
    f1 = f1_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx] > optimal_threshold)

    # recall-score
    recall = recall_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx] > optimal_threshold)

    # precision
    precision = precision_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx] > optimal_threshold)

    # auc
    auc = roc_auc_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx])

    # roc plot
    plot_roc_curve(fpr, tpr, fold, optimal_threshold, auc)

    return LOGITS, PROBS, TARGETS, PATIENT_ID, IMAGE_NAME, CANCER, val_loss, acc, f1, recall, precision, auc, optimal_threshold



def main():
    '''
    ####################################################
    # stone data 데이터셋 : dataset.get_df_stone
    ####################################################
    '''
    df_train, df_test, meta_features, n_meta_features, target_idx, meta_columns = get_df_stone(
        k_fold=args.k_fold,
        out_dim=args.out_dim,
        data_dir=args.data_dir,
        data_folder=args.data_folder,
        use_meta=args.use_meta,
        use_ext=args.use_ext
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    LOGITS = []
    PROBS = []
    TARGETS = []
    PATIENT_ID = []
    IMAGE_NAME = []
    CANCER = []
    STD_LIST = []
    STD_AUC_LIST = []

    folds = range(args.k_fold)
    for fold in folds:
        print(f'Evaluate data fold{str(fold)}')
        df_valid = df_train[df_train['fold'] == fold]

        # batch_normalization에서 배치 사이즈 1인 경우 에러 발생할 수 있으므로, 데이터 한개 버림
        if len(df_valid) % args.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid) - 1)

        if args.DEBUG:
            df_valid = pd.concat([
                df_valid[df_valid['target'] == target_idx].sample(args.batch_size * 3),
                df_valid[df_valid['target'] != target_idx].sample(args.batch_size * 3)
            ])

        dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', meta_features, transform=transforms_val, meta_columns=meta_columns)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
        elif args.eval == 'best_no_ext':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_no_ext_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
        model = model.to(device)


        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)

        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        '''
        ####################################################
        # data를 위한 평가함수 : val_epoch
        ####################################################
        '''
        this_LOGITS, this_PROBS, this_TARGETS, this_PATIENT_ID, this_IMAGE_NAME, this_CANCER, val_loss, acc, f1, recall, precision, auc, optimal_threshold = val_epoch(model, valid_loader, target_idx, fold, is_ext=df_valid['is_ext'].values, n_test=1, get_output=True)
        LOGITS.append(this_LOGITS)
        PROBS.append(this_PROBS)
        TARGETS.append(this_TARGETS)
        PATIENT_ID.append(this_PATIENT_ID)
        IMAGE_NAME.append(this_IMAGE_NAME)
        CANCER.append(this_CANCER)
        STD_LIST.append((this_PROBS.argmax(1) == this_TARGETS).mean() * 100.)
        STD_AUC_LIST.append(roc_auc_score((this_TARGETS == target_idx).astype(float), this_PROBS[:, target_idx]))


    # 전체 fold 끝
    # 전체 데이터에 대한 confidence 구하고, confidence sum 기준으로 sorting
    confidence_score(PROBS, TARGETS, PATIENT_ID, IMAGE_NAME, optimal_threshold)

    #sorting 된 confidence_csv에서 상위 'select'개의 사진 저장
    sorted_confidence_save(args.select)

    PROBS = np.concatenate(PROBS)
    TARGETS = np.concatenate(TARGETS)
    PATIENT_ID = np.concatenate(PATIENT_ID)
    IMAGE_NAME = np.concatenate(IMAGE_NAME)
    CANCER = np.concatenate(CANCER)

    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
    fpr, tpr, threshold = roc_curve((TARGETS == target_idx).astype(float), PROBS[:, target_idx])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]



    # f1-score
    f1 = f1_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx] > optimal_threshold)

    # recall-score
    recall = recall_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx] > optimal_threshold)

    # precision
    precision = precision_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx] > optimal_threshold)

    # auc
    auc = roc_auc_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx])

    AUC = []

    plot_roc_curve(fpr, tpr, 'All', optimal_threshold, auc)


    print('-------------------------------------최종 결과---------------------------------')
    print('validation loss : ', f'{val_loss:.6f}')
    print('Accuracy : ', f'{acc:.4f}')
    print(" -- std ACC score: ", np.std(np.array(STD_LIST), ddof=1))
    print('f1-score:', f'{f1:.4f}')
    print('recall-score:', f'{recall:.4f}')
    print('precision-score:', f'{precision:.4f}')
    print('auc: ', f'{auc:.4f}')
    print(" -- std AUC score: ", np.std(np.array(STD_AUC_LIST), ddof=1))
    print(classification_report(TARGETS, PROBS[:, target_idx] > optimal_threshold))


    cancer_dict = {'PROBS': PROBS.argmax(1), 'TARGETS': TARGETS, 'PATIENT_ID': PATIENT_ID, 'IMAGE_NAME': IMAGE_NAME,
                   'CANCER': CANCER}
    df_cancer = pd.DataFrame(cancer_dict)

    cancer_acc = []

    for i in df_cancer["CANCER"].unique():
        acc = (df_cancer[df_cancer["CANCER"] == i]['TARGETS'] == df_cancer[df_cancer["CANCER"] == i][
            'PROBS']).mean() * 100
        print(f'{i} acc = {acc}')
        cancer_acc.append(acc)

    print(f'acc = {np.mean(np.array(cancer_acc))}')


    Test_dict = {'fpr': fpr, 'tpr': tpr}
    df_Test = pd.DataFrame(Test_dict)
    df_Test.to_csv(f'./test_roc/{args.kernel_type}.csv')


def visualization(self, confusion_matrix, obj=False):
    label = list(self.target2idx)
    obj_label = list(self.model2idx)


    mpl.style.use('seaborn')

    if obj:
        confusion_matrix = pd.DataFrame(confusion_matrix.numpy(), index=[self.img_type], columns=obj_label)
    else:
        total = np.sum(confusion_matrix.numpy(), axis=1)
        confusion_matrix = confusion_matrix / total[:, None]
        confusion_matrix = pd.DataFrame(confusion_matrix.numpy(), index=label, columns=label)

    fig = plt.figure(figsize=(12, 9))
    plt.clf()

    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    cmap = sns.cubehelix_palette(light=0.95, dark=0.08, as_cmap=True)
    ax = sns.heatmap(confusion_matrix, annot=False, annot_kws={"size": 10}, fmt='.2f', linewidths=0.3, cmap=cmap, clip_on=False)

    if obj:
        plt.xticks(np.arange(len(obj_label)) + 0.5, obj_label, size=25, rotation='vertical')
        plt.yticks(np.arange(len([self.img_type])) + 0.5, [self.img_type], size=25, rotation='horizontal')
    else:
        plt.xticks(np.arange(len(label)) + 0.5, label, size=25, rotation='vertical')
        plt.yticks(np.arange(len(label)) + 0.5, label, size=25, rotation='horizontal')

    ax.axhline(y=0, color='k', linewidth=1)
    ax.axhline(y=len(confusion_matrix), color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.axvline(x=len(confusion_matrix), color='k', linewidth=2)


    plt.savefig(os.path.join('./graph',
                             f'final_{self.weight_num}_({self.task_type}+{self.side_task_type})_{self.img_type}.png'),
                dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()






if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')
    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)
    # if args.use_gradcam:
    #     os.makedirs(os.path.join(args.gradcam_dir, args.kernel_type), exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_MMC
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_MMC
    elif 'efficientnet' in args.enet_type:
        ModelClass = Multitask_MMC_Model
    elif args.enet_type == 'densenet121':
        ModelClass = Effnet_MMC
    elif args.enet_type == 'resnet101':
        ModelClass = Effnet_MMC
    elif args.enet_type == 'vit':
        ModelClass = MMC_Model
    elif args.enet_type == 'vgg19':
        ModelClass = Effnet_MMC
    elif args.enet_type == 'inception_resnet_v2':
        ModelClass = Effnet_MMC
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()
    m = nn.Sigmoid()
    main()