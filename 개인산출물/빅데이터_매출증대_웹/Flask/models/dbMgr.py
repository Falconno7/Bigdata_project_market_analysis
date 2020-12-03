import pandas as pd 
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Data Set Learning & Modeling
def modeling():
    result = None
    try:
        # Load Train Data 
        print("=========Modeling========")
        df1 = pd.read_excel('2016_health_screenings_data.xlsx')
        print("학습 데이터 row / column 수 : ",df1.shape)

        # Data Preprocessing (100,000)
        df2 = df1.iloc[0:100000]

        cond1 = (df2['식전혈당(공복혈당)'] >= 126)
        df2.loc[cond1, 'Target']= 1
        df2.loc[~cond1, 'Target']= 0

        # Select Target
        X = df2[['트리글리세라이드', '연령대코드(5세단위)', '수축기혈압', '감마지티피','이완기혈압', '체중(5Kg단위)']]
        Y =df2['Target']
        print("설명 변수 데이터 row/column 수 : ",X.shape)
        print("목표 변수 데이터 row/column 수 : ",Y.shape)

        print("=========Split Data========")
        # Split Data Set
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1234)
        print("Train X 데이터 row/column 수 :  ",X_train.shape)
        print("Test X 데이터 row/column 수 :  ",X_test.shape)
        print("Train Y 데이터 row/column 수 :  ",Y_train.shape)
        print("Test Y 데이터 row/column 수 :  ",Y_test.shape)

        # Fitting Model
        model = GradientBoostingClassifier()
        result = model.fit(X_train,Y_train)

    except Exception as e :
        print("========MODELING ERROR========")    
    finally:
        pass
    return result