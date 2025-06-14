import pandas as pd
import streamlit as st
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier


st.title("MODEL BUILDING FOR CLASSIFICATION")

col1, col2 = st.columns(2)
with col1:
    x_file= st.file_uploader("upload 'X' data",type=['csv'])
with col2:   
    y_file = st.file_uploader("upload 'Y' data",type=['csv'])





 #logistic ression
def logistic_r (train_x,train_y,test_x,test_y):
    le = LogisticRegression()
    model_lr = le.fit(train_x,train_y)

    lr_train_predict = model_lr.predict(train_x)
    lr_test_predict = model_lr.predict(test_x)

    lr_train_acc = accuracy_score(train_y,lr_train_predict)*100
    lr_test_acc = accuracy_score(test_y,lr_test_predict)*100

    return lr_train_acc,lr_test_acc,model_lr

#Random forest
def random_forest (train_x, train_y,test_x,test_y):
    kfold = KFold(n_splits=10, random_state=5,shuffle=True)
    n_estimators = np.array(range(10,50)) 
    max_feature = [2,3]
    param_grid = dict(n_estimators =n_estimators,max_features=max_feature)

    model_rfc = RandomForestClassifier()
    grid_rfc = GridSearchCV(estimator=model_rfc, param_grid=param_grid)
    grid_rfc.fit(train_x, train_y)

    RFC_Model = RandomForestClassifier(grid_rfc.best_params_['n_estimators'])
    RFC_Model.fit(train_x,train_y)

    RFC_train_predict = RFC_Model.predict(train_x)
    RFC_test_predict = RFC_Model.predict(test_x)

    rfc_train_acc = accuracy_score(train_y,RFC_train_predict)*100
    rfc_test_acc = accuracy_score(test_y,RFC_test_predict)*100

    return rfc_train_acc,rfc_test_acc, model_rfc

 #support vector clasifer

#Support Vector clasiffiers
def svc(train_x,train_y,test_x,test_y):
    clf = SVC()
    param_grid_svc = [{'kernel':['rbf','sigmoid','poly'],'gamma':[0.5,0.1,0.005],'C':[25,20,10,0.1,0.001] }]
    svc= RandomizedSearchCV(clf,param_grid_svc,cv=10)
    svc.fit(train_x,train_y)

    svc_train_predict = svc.predict(train_x)
    svc_test_predict = svc.predict(test_x)

    svc_train_acc = accuracy_score(train_y,svc_train_predict)*100
    svc_test_acc = accuracy_score(test_y,svc_test_predict)*100

    return svc_train_acc,svc_test_acc,svc

#bagging
def bagging(train_x,train_y,test_x,test_y):
    cart = DecisionTreeClassifier()

    model_bag = BaggingClassifier(estimator=cart, n_estimators= 10, random_state=6)
    model_bag.fit(train_x,train_y)

    bag_train_predict = model_bag.predict(train_x)
    bag_test_predict = model_bag.predict(test_x)

    bag_train_acc = accuracy_score(train_y,bag_train_predict)*100
    bag_test_acc = accuracy_score(test_y,bag_test_predict)*100

    return bag_train_acc,bag_test_acc,model_bag

#xgb
def xgb(train_x,train_y,test_x,test_y):
    n_estimators =np.array(range(10,80,10))
    xgb_model = XGBClassifier(n_estimators=70,max_depth=5)
    xgb_model.fit(train_x,train_y)

    xgb_train_predict = xgb_model.predict(train_x)
    xgb_test_predict = xgb_model.predict(test_x)

    xgb_train_acc = accuracy_score(train_y,xgb_train_predict)*100
    xgb_test_acc = accuracy_score(test_y,xgb_test_predict)*100

    return xgb_train_acc,xgb_test_acc,xgb_model

#LGBM
def lgbm(train_x,train_y,test_x,test_y):

    params = {}
    params['learning_rate'] = 1
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 5
    params['min_data'] = 10
    params['max_depth'] = 5

    lgbm_model = lgb.LGBMClassifier()
    lgbm_model.fit(train_x,train_y)

    lgbm_train_predict = lgbm_model.predict(train_x)
    lgbm_test_predict = lgbm_model.predict(test_x)

    lgbm_train_acc = accuracy_score(train_y,lgbm_train_predict)*100
    lgbm_test_acc = accuracy_score(test_y,lgbm_test_predict)*100

    return lgbm_train_acc,lgbm_test_acc,lgbm_model

#NaiveByaes
def NB(train_x,train_y,test_x,test_y):
    nb_model = GaussianNB()
    nb_model.fit(train_x, train_y)

    nb_train_predict=nb_model.predict(train_x)
    nb_test_predict=nb_model.predict(test_x)

    nb_train_acc = accuracy_score(train_y,nb_train_predict)*100
    nb_test_acc = accuracy_score(test_y,nb_test_predict)*100

    return nb_train_acc,nb_test_acc,nb_model


#KNN
def knn(train_x,train_y,test_x,test_y):

    n_neighbors = np.array(range(2,30))
    param_grid = dict(n_neighbors=n_neighbors)

    model = KNeighborsClassifier()
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid.fit(train_x, train_y)

    knn_model = KNeighborsClassifier(grid.best_params_['n_neighbors'])
    knn_model.fit(train_x, train_y)

    knn_train_predict=knn_model.predict(train_x)
    knn_test_predict=knn_model.predict(test_x)

    knn_train_acc = accuracy_score(train_y,knn_train_predict)*100
    knn_test_acc = accuracy_score(test_y,knn_test_predict)*100

    return knn_train_acc,knn_test_acc,knn_model

def df(train_x,train_y,test_x,test_y):

    list= [logistic_r (train_x,train_y,test_x,test_y), 
    random_forest (train_x, train_y,test_x,test_y),
    svc(train_x,train_y,test_x,test_y),
    bagging(train_x,train_y,test_x,test_y),
    xgb(train_x,train_y,test_x,test_y),
    lgbm(train_x,train_y,test_x,test_y),
    NB(train_x,train_y,test_x,test_y),
    knn(train_x,train_y,test_x,test_y) ]

    acc_data = pd.DataFrame(list,columns=('Train accuracy','Test accuracy','Model'),index=['logistic','Random_forest','SVC','Bagging','XGB','LGBM','NB',"KNN"])

    return acc_data


if st.button("ENTER"):

    start_time= time.time()

    if x_file and y_file is not None:
        

        x=pd.read_csv(x_file,index_col=0)
        y=pd.read_csv(y_file,index_col=0)
       
        train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=10)

        data = df(train_x,train_y,test_x,test_y)

        st.dataframe(data)

        for model_name in data.index:
            model = data.loc[model_name, 'Model']
            file_name = f"{model_name}_model.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(model, f) 

            with open(file_name, "rb") as f:
                st.download_button(
                    label=f"📥 Download {model_name.capitalize()} Model",
                    data=f.read(),
                    file_name=file_name,
                    mime="application/octet-stream",key=f"download_{model_name}"
                ) 

    end_time= time.time()

    time_taken = end_time-start_time

    st.success(f"Task complited in {time_taken:.2f} seconds")

    
       
     
       

