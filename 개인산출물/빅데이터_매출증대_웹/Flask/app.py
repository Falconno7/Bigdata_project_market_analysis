#app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.api as sm
import warnings
import itertools
from statsmodels.tsa.arima_model import ARIMA

# Flask Class 객체를 선언한다 .
app = Flask(__name__)

# Main Page
@app.route('/') # Local Host 
def index():
        return render_template("index.html")

# Data Model Page
@app.route('/data_model_form')
def data_model():
        return render_template('data_model_form.html')

@app.route('/data_model_form2')
def data_model2():
        return render_template('data_model_form_2.html')

@app.route('/modeling2',methods=['POST'])
def predict_data2():
        품목 = str(request.form['name'])
        date = str(request.form['date'])

        df_raw = pd.read_csv("product_data_취합.csv",encoding='utf-8')
        df_raw = pd.DataFrame(df_raw)
        df_b = df_raw[['공급일자','물품대분류']]
        df_b['cnt'] = 1
        df_b['공급일자'] = df_b['공급일자'].astype('datetime64[ns]')
        df_b = df_b.groupby(['공급일자', '물품대분류']).sum()
        df_b.reset_index(inplace=True)

        df_free = df_b[df_b['물품대분류'] == 품목]
        df_a = df_free.reset_index()
        del df_a['index']
        df_a['work_week'] = df_a['공급일자'].dt.week
        df_a['month'] = df_a['공급일자'].dt.month
        df_a['day_of_week'] = df_a['공급일자'].dt.day_name()
        data1 = df_a.set_index('공급일자')

        y = data1['cnt'].resample('1w').mean()
        mod = sm.tsa.statespace.SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_invertibility=False, enforce_stationarity=False)
        result2 = mod.fit()
        result2.get_prediction()
        pred = result2.get_prediction(start = pd.to_datetime('2018-09-30'), dynamic = False)
        pred_ci = pred.conf_int()

        y_forecasted = pred.predicted_mean
        y_forecasted = pd.DataFrame(data=y_forecasted)
        y_forecasted.reset_index(inplace=True)
        y_forecasted.rename(columns={0:'매출량'}, inplace=True)
        p = y_forecasted[y_forecasted['공급일자'] == date].predicted_mean
        _string = str(p)
        print(_string)
        string = _string.split(' ')
        print(string)
        _temp = string[4]
        _temp = _temp[:-6]
        ret = ':  매출량 = ' + _temp
        return render_template('data_model_form_2.html', predict=ret)


# Data Predict & Insert Data Base
@app.route('/modeling',methods=['POST'])
def predict_data():
        data0 = request.form['data0'] #연령
        data1 = request.form['data1'] #성별 남, radio_
        data2 = request.form['data2']
        data3 = request.form['data3']
        data4 = request.form['data4']
        data5 = request.form['data5']

        # print(data0,data1,data2,data3,data3,data4,data5)
#연령	주당구매액	성별_남	성별_여	배송서비스신청여부_미신청	배송서비스신청여부_신청
# 모바일알람여부_.	모바일알람여부_수신	구_광주	구_기타	구_기흥구	구_분당구	구_서울 강남구
# 구_서울 송파구	구_수원 권선구	구_수원 영통구	구_수원 장안구	구_수원 팔달구	구_수정구
# 구_수지구	구_중원구	구_처인구	구_하남	구_화성

        df1 = pd.read_csv('df_x.csv',engine='python')
        df1['연령'] = int(data0)
        df1['주당구매액'] = float(data5)
        if data1 == '남' :
                df1['성별_남'] = 1
        else:
                df1['성별_여'] = 1

        if data2 == '예':
                df1['배송서비스신청여부_신청'] = 1
        else:
                df1['배송서비스신청여부_미신청'] = 1

        if data3 == '예':
                df1['모바일알람여부_수신'] = 1
        else:
                df1['모바일알람여부_.'] = 1

        df1[data4] = 1
        # df1.drop('Unnamed: 0',axis=1,inplace=True)
        df1 = pd.DataFrame(df1)
        print(df1)
        # scaler = joblib.load('Standard_Scaler.pkl')
        train_x = pd.read_csv('df_raw_x.csv', engine='python')
        train_x.drop('Unnamed: 0',axis=1,inplace=True)
        scaler = StandardScaler()
        scaler.fit(train_x)
        x_scaled = scaler.transform(df1)
        print(x_scaled)
        DecisionTreeClassifier = joblib.load('Decision_tree_model.pkl')
        predict = DecisionTreeClassifier.predict(x_scaled)
        # Input Data
        # data = [[data1,data2_1,data2_2,data3_1,data3_2,data4]]
        # Predict Y 
        # result = dt.modeling()
        # # data = pd.DataFrame(data=df1)
        # predict = result.predict(df1)

        # return render_template('data_model_form.html', predict=predict.round(3))

        print(predict)

        if predict == 0:
                p = ' : 일반 회원'
        else :
                p = ' : VIP'
        return render_template('data_model_form.html',predict = p)
# Finished Code 
if __name__ == '__main__' :
    app. debug = True
    app.run(debug=True)