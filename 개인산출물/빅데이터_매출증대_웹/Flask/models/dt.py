import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Data Set Learning & Modeling
def modeling():
    result = None
    try:
        # Load Train Data
        print("=========Modeling========")
        df1 = pd.read_csv('df_raw_x.csv', encoding='utf-8')
        print("학습 데이터 row / column 수 : ",df1.shape)

        df2 = pd.read_csv('df_raw_y.csv', encoding='ecu-kr')

        # Fitting Model
        model = DecisionTreeClassifier()
        result = model.fit(df1,df2)

    except Exception as e :
        print("========MODELING ERROR========")
    finally:
        pass
    return result