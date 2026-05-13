import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#CPU:-1 GPU:0

import pandas as pd
import numpy as np
import csv

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from sklearn import metrics  
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix  

from tensorflow.keras import layers, models

feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','TINN','LF','HF','ULF','VLF','LFHF',
         'total_power','lfp','hfp','SD1','SD2','pA','pQ','ApEn','shanEn','D2','subject','label']
WINDOW_SIZE = '120'

NOISE = ['bp_time_ens']

subjects = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

# +
from collections import Counter



def read_csv(path, feats, testset_num):
    print("testset num: ",testset_num)
    df = pd.read_csv(path, index_col = 0)
    
    df = df[feats]

    train_df = df.loc[df['subject'] != testset_num]
    test_df =  df.loc[df['subject'] == testset_num]

    del train_df['subject']
    del test_df['subject']
    del df['subject']

    
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values   
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values    
    
    return df, X_train, y_train, X_test, y_test
# -



# # Machine learning



# +
def DT_model(X_train, y_train, X_test, y_test):
    
    model = tree.DecisionTreeClassifier(random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)

    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    return AUC, F1, ACC


def RF_model(X_train, y_train, X_test, y_test):
    
    model = RandomForestClassifier(max_depth=4, random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
   
    return AUC, F1, ACC

def AB_model(X_train, y_train, X_test, y_test):
    
    model = AdaBoostClassifier(random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    return AUC, F1, ACC

def KN_model(X_train, y_train, X_test, y_test):
    
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4

    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    return AUC, F1, ACC

def LDA_model(X_train, y_train, X_test, y_test):
    
    model = LinearDiscriminantAnalysis()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    
    return AUC, F1, ACC

def SVM_model(X_train, y_train, X_test, y_test):
    
    model = svm.SVC()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    
    return AUC, F1, ACC


def GB_model(X_train, y_train, X_test, y_test):
    
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    ACC = accuracy_score(y_test, y_pred)
    
    fpr,tpr, roc_auc = dict(), dict(), dict()
    n_classes = 4
    
    y_pred = np.eye(n_classes)[y_pred]
    y_test = np.eye(n_classes)[y_test]  # one-hot-vector
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    AUC = np.array(list(roc_auc.values())).mean()
    F1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')  
    
    
    return AUC, F1, ACC

def build_baseline_model():
    model = models.Sequential([
        # 使用 Input 層定義 (樣本, 24, 1)
        layers.Input(shape=(24, 1)),
        
        # 第一組：24 -> 12
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # 第二組：12 -> 6
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # 第三組：6 -> 3 (這裡建議停止池化，或者只做卷積)
        layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2), 
        
        # 第四組：僅卷積，不再池化（避免維度歸零）
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        
        # 展平與全連接層
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')  # 4 classes
    ])
    
    return model

def cnn1d_model(X_train, y_train, X_test, y_test):
    model = build_baseline_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    y_train = np.eye(4)[y_train]  # one-hot-vector
    y_test = np.eye(4)[y_test]  # one-hot-vector
    
    model.fit(X_train, y_train, epochs=30, batch_size=32)
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_classes = np.eye(4)[predicted_classes]  # one-hot-vector
    accuracy = accuracy_score(y_test, predicted_classes)
    F1 = f1_score(y_test, predicted_classes, average='macro')
    Precision = precision_score(y_test, predicted_classes, average='macro')
    Recall = recall_score(y_test, predicted_classes, average='macro')
   
    
    return accuracy, F1, Precision, Recall

def MLP_model(X_train, y_train, X_test, y_test):
    model = models.Sequential([
        layers.Input(shape=(24,1)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    y_train = np.eye(4)[y_train]  # one-hot-vector
    y_test = np.eye(4)[y_test]  # one-hot-vector
    
    model.fit(X_train, y_train, epochs=30, batch_size=32)
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_classes = np.eye(4)[predicted_classes]  # one-hot-vector
    accuracy = accuracy_score(y_test, predicted_classes)
    F1 = f1_score(y_test, predicted_classes, average='macro')
    Precision = precision_score(y_test, predicted_classes, average='macro')
    Recall = recall_score(y_test, predicted_classes, average='macro')
    
    return accuracy, F1, Precision, Recall

from tensorflow.keras import regularizers,optimizers
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

def RNN_model(X_train, y_train, X_test, y_test):
    model = models.Sequential([
        layers.Input(shape=(24, 1)),
        
        # LSTM 層：比 SimpleRNN 更能捕捉 sequential dependencies
        layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), # 穩定數據分佈，加速收斂
        layers.Dropout(0.3),
        
        # 3. 第二層 LSTM：高度抽象化序列特徵
        # return_sequences=False 代表只輸出最後一個總結向量
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.3),
        
        # 4. 全連接層 (Dense)：進行非線性映射
        layers.Dense(16, activation='relu'),
        
        # 5. 輸出層：多分類使用 Softmax
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    y_train = np.eye(4)[y_train]  # one-hot-vector
    y_test = np.eye(4)[y_test]  # one-hot-vector
    model.fit(X_train, y_train, epochs=30, batch_size=64)
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_classes = np.eye(4)[predicted_classes]  # one-hot-vector
    accuracy = accuracy_score(y_test, predicted_classes)
    F1 = f1_score(y_test, predicted_classes, average='macro')
    Precision = precision_score(y_test, predicted_classes, average='macro')
    Recall = recall_score(y_test, predicted_classes, average='macro')
    
    return accuracy, F1, Precision, Recall


# +
feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','TINN','LF','HF','ULF','VLF','LFHF',
         'total_power','SD1','SD2','pA','pQ','ApEn','shanEn','D2','subject','label']       

for n in NOISE:
    
    #path = '27_features_ppg_9/data_merged_' + n + WINDOW_SIZE + '.csv'
    #result_path_all = 'result/BGM2/all_features_' + n + WINDOW_SIZE + '.csv'

    path = '27_features_ppg_test_4/data_merged_' + n + WINDOW_SIZE + '.csv'
    result_path_all = 'result_quad.csv'
    X=None
    DT_AUC, DT_F1, DT_ACC = [], [], []
    RF_AUC, RF_F1, RF_ACC= [], [], []
    AB_AUC, AB_F1, AB_ACC = [], [], []
    KN_AUC, KN_F1, KN_ACC = [], [], []
    LDA_AUC, LDA_F1, LDA_ACC = [], [], []
    SVM_AUC, SVM_F1, SVM_ACC = [], [], []
    GB_AUC, GB_F1, GB_ACC = [], [], []
    ACC_CNN, F1_CNN, Precision_CNN, Recall_CNN = [], [], [], []
    ACC_MLP, F1_MLP, Precision_MLP, Recall_MLP = [], [], [], []
    ACC_RNN, F1_RNN, Precision_RNN, Recall_RNN = [], [], [], []
    for sub in subjects:
    
        df, X_train, y_train, X_test, y_test = read_csv(path, feats, sub)
        df.fillna(0)
        # Normalization
        X=X_train
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test)  
        
    
        auc_dt, f1_dt, acc_dt = DT_model(X_train, y_train, X_test, y_test)
        auc_rf, f1_rf, acc_rf = RF_model(X_train, y_train, X_test, y_test)
        auc_ab, f1_ab, acc_ab = AB_model(X_train, y_train, X_test, y_test)
        auc_kn, f1_kn, acc_kn = KN_model(X_train, y_train, X_test, y_test)
        auc_lda, f1_lda, acc_lda = LDA_model(X_train, y_train, X_test, y_test)
        auc_svm, f1_svm, acc_svm = SVM_model(X_train, y_train, X_test, y_test)
        auc_gb, f1_gb, acc_gb = GB_model(X_train, y_train, X_test, y_test)
        acc_cnn, f1_cnn, precision_cnn, recall_cnn = cnn1d_model(X_train, y_train, X_test, y_test)
        acc_mlp, f1_mlp, precision_mlp, recall_mlp = MLP_model(X_train, y_train, X_test, y_test)
        acc_rnn, f1_rnn, precision_rnn, recall_rnn = RNN_model(X_train, y_train, X_test, y_test)
        DT_AUC.append(auc_dt)
        DT_F1.append(f1_dt)
        DT_ACC.append(acc_dt)
        RF_AUC.append(auc_rf)
        RF_F1.append(f1_rf)
        RF_ACC.append(acc_rf)
        AB_AUC.append(auc_ab)
        AB_F1.append(f1_ab)
        AB_ACC.append(acc_ab)
        KN_AUC.append(auc_kn)
        KN_F1.append(f1_kn)
        KN_ACC.append(f1_kn)
        LDA_AUC.append(auc_lda)
        LDA_F1.append(f1_lda)
        LDA_ACC.append(acc_lda)
        SVM_AUC.append(auc_svm)
        SVM_F1.append(f1_svm)
        SVM_ACC.append(acc_svm)
        GB_AUC.append(auc_gb)
        GB_F1.append(f1_gb)
        GB_ACC.append(acc_gb)
        ACC_CNN.append(acc_cnn)
        F1_CNN.append(f1_cnn)
        Precision_CNN.append(precision_cnn)
        Recall_CNN.append(recall_cnn)
        ACC_MLP.append(acc_mlp)
        F1_MLP.append(f1_mlp)
        Precision_MLP.append(precision_mlp)
        Recall_MLP.append(recall_mlp)
        ACC_RNN.append(acc_rnn)
        F1_RNN.append(f1_rnn)
        Precision_RNN.append(precision_rnn)
        Recall_RNN.append(recall_rnn)

    

    with open(result_path_all, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['subject','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17','total'])
        writer.writerow(['DT_AUC'] + DT_AUC + [np.mean(DT_AUC)])
        writer.writerow(['RF_AUC'] + RF_AUC + [np.mean(RF_AUC)])
        writer.writerow(['AB_AUC'] + AB_AUC + [np.mean(AB_AUC)])
        writer.writerow(['KN_AUC'] + KN_AUC + [np.mean(KN_AUC)])
        writer.writerow(['LDA_AUC'] + LDA_AUC + [np.mean(LDA_AUC)])
        writer.writerow(['SVM_AUC'] + SVM_AUC + [np.mean(SVM_AUC)])
        writer.writerow(['GB_AUC'] + GB_AUC + [np.mean(GB_AUC)])
        writer.writerow(['DT_F1'] + DT_F1 + [np.mean(DT_F1)])
        writer.writerow(['RF_F1'] + RF_F1 + [np.mean(RF_F1)])
        writer.writerow(['AB_F1'] + AB_F1 + [np.mean(AB_F1)])
        writer.writerow(['KN_F1'] + KN_F1 + [np.mean(KN_F1)])
        writer.writerow(['LDA_F1'] + LDA_F1 + [np.mean(LDA_F1)])
        writer.writerow(['SVM_F1'] + SVM_F1 + [np.mean(SVM_F1)])
        writer.writerow(['GB_F1'] + GB_F1 + [np.mean(GB_F1)])
        writer.writerow(['DT_ACC'] + DT_ACC + [np.mean(DT_ACC)])
        writer.writerow(['RF_ACC'] + RF_ACC + [np.mean(RF_ACC)])
        writer.writerow(['AB_ACC'] + AB_ACC + [np.mean(AB_ACC)])
        writer.writerow(['KN_ACC'] + KN_ACC + [np.mean(KN_ACC)])
        writer.writerow(['LDA_ACC'] + LDA_ACC + [np.mean(LDA_ACC)])
        writer.writerow(['SVM_ACC'] + SVM_ACC + [np.mean(SVM_ACC)])
        writer.writerow(['GB_ACC'] + GB_ACC + [np.mean(GB_ACC)])
        writer.writerow(['CNN_ACC'] + ACC_CNN + [np.mean(ACC_CNN)])
        writer.writerow(['CNN_F1'] + F1_CNN + [np.mean(F1_CNN)])
        writer.writerow(['CNN_Precision'] + Precision_CNN + [np.mean(Precision_CNN)])
        writer.writerow(['CNN_Recall'] + Recall_CNN + [np.mean(Recall_CNN)])
        writer.writerow(['MLP_ACC'] + ACC_MLP + [np.mean(ACC_MLP)])
        writer.writerow(['MLP_F1'] + F1_MLP + [np.mean(F1_MLP)])
        writer.writerow(['MLP_Precision'] + Precision_MLP + [np.mean(Precision_MLP)])
        writer.writerow(['MLP_Recall'] + Recall_MLP + [np.mean(Recall_MLP)])
        writer.writerow(['RNN_ACC'] + ACC_RNN + [np.mean(ACC_RNN)])
        writer.writerow(['RNN_F1'] + F1_RNN + [np.mean(F1_RNN)])
        writer.writerow(['RNN_Precision'] + Precision_RNN + [np.mean(Precision_RNN)])
        writer.writerow(['RNN_Recall'] + Recall_RNN + [np.mean(Recall_RNN)])
        file.close()
    print("DONE: ",n)
# -
# #### 

