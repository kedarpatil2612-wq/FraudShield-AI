import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
FEATURE_COLUMNS=['Time']+[f'V{i}' for i in range(1,29)]+['Amount']
def prepare_data(df):
    required=FEATURE_COLUMNS+['Class']; missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    clean=df[required].replace([np.inf,-np.inf],np.nan).dropna()
    if clean.empty or clean.Class.nunique()!=2: raise ValueError('Dataset must contain both Class 0 and Class 1.')
    return clean[FEATURE_COLUMNS].astype(float),clean.Class.astype(int)
def train_model(X,y):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=1000,class_weight='balanced',solver='liblinear',random_state=42))])
    model.fit(Xtr,ytr); p=model.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int); fpr,tpr,_=roc_curve(yte,p)
    m={'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'cm':confusion_matrix(yte,pred,labels=[0,1]),'fpr':fpr.tolist(),'tpr':tpr.tolist(),'auc':roc_auc_score(yte,p)}
    return model,m
def predict_one(model,values):
    row=pd.DataFrame([{c:float(values[c]) for c in FEATURE_COLUMNS}],columns=FEATURE_COLUMNS); p=float(model.predict_proba(row)[0,1])
    return {'prediction':int(p>=.5),'probability':p,'risk':'HIGH' if p>=.7 else 'MEDIUM' if p>=.35 else 'LOW'}
