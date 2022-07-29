
from flask import Flask
from flask import Flask, make_response, jsonify, request, render_template
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import warnings
import datetime as dt
warnings.filterwarnings('ignore')
from varname import nameof
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from scipy.optimize import minimize



import pandas as pd
import numpy as np


import requests
from bs4 import BeautifulSoup
import urllib.request


import json
from urllib import parse
import pandas as pd
import xmltodict


from konlpy.tag import Komoran
import re


from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re


import cufflinks as cf
import plotly
import plotly.express as px


import folium
from folium.plugins import MarkerCluster



import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
sns.set()
plt.rc('font', family='Malgun Gothic')

# #-------------------------------------------- 차트 관련 : 한글폰트 설정용
import matplotlib.font_manager as fm
def fontcheck():
    f = [f.name for f in fm.fontManager.ttflist]
    # print(f)
    plt.rc('font', family='Malgun Gothic')




app = Flask(__name__) # 신경 안써도 될 부분
session_cookie_samesite=app.config["SESSION_COOKIE_SAMESITE"]
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=None,
)








def kodex200_info():
    # 국내 ETF 첫번째꺼 뽑아오기  - KODEX 200
    def top_etf():
        import FinanceDataReader as fdr
        df_etf = fdr.StockListing('ETF/KR')
        code = df_etf['Symbol'][0]
        etf_1 = fdr.DataReader(code, '2019')
        return etf_1

    etf_1 = top_etf()

    # 이동평균선 구하는 함수
    def make_ma(df):
        ma5 = df['Close'].rolling(window=5).mean()  # 5일 이동평균선
        ma20 = df['Close'].rolling(window=20).mean()  # 20일 이동평균선
        ma60 = df['Close'].rolling(window=60).mean()  # 60일 이동평균선
        ma120 = df['Close'].rolling(window=120).mean()  # 120일 이동평균선
        df['ma5'] = ma5
        df['ma20'] = ma20
        df['ma60'] = ma60
        df['ma120'] = ma120
        df = df.dropna()
        df = df.reset_index()

        # 삭제할 인덱스 저장 - 2019년 삭제할거임
        tmp = []
        for i in range(len(df)):
            if df['Date'][i].year == 2019:
                tmp.append(i)
        df = df.drop(tmp)
        return df

    etf_1 = make_ma(etf_1)


    fig = px.line(etf_1, x='Date', y=['Close', 'ma5', 'ma20', 'ma60', 'ma120'])
    fig.update_layout(title='KODEX ETF 200',
                      xaxis_title='DATE',
                      yaxis_title='Moving Average')
    res_str = fig.to_json()
    return res_str







def etf_line():
    ESGA = pd.read_csv("./data/20_High.csv")
    ESGD = pd.read_csv("./data/20_Low.csv")
    sin_stock = pd.read_csv("./data/sin_stock.csv")
    ETF_3 = pd.DataFrame()
    ETF_3['날짜'] = ESGA['날짜']
    ETF_3['ESG_high'] = ESGA['ETF_지수']
    ETF_3['ESG_low'] = ESGD['ETF_지수']
    ETF_3['sin_stock'] = sin_stock['ETF_지수']
    fig = px.line(ETF_3, x='날짜', y=['ESG_high', 'ESG_low', 'sin_stock'])
    res_str = fig.to_json()
    return res_str


def volume():
    ESGA = pd.read_csv("./data/20_High.csv")
    ESGD = pd.read_csv("./data/20_Low.csv")
    sin_stock = pd.read_csv("./data/sin_stock.csv")
    ETF_4 = pd.DataFrame()
    ETF_4['날짜'] = ESGA['날짜']
    ETF_4['ESG_high'] = ESGA['거래량']
    ETF_4['ESG_low'] = ESGD['거래량']
    ETF_4['sin_stock'] = sin_stock['거래량']
    ETF_4
    fig = px.line(ETF_4, x='날짜', y=['ESG_high', 'ESG_low', 'sin_stock'])
    res_str = fig.to_json()  # ---tohtml
    return res_str








def row4넣을차트1(): # 오 이거는 그냥 이미지 반환했네.

    return "row4넣을차트1.png"

def row4넣을차트2(): # 오 이거는 그냥 이미지 반환했네.

    return "row4넣을차트2.png"

def row4넣을차트3(): # 오 이거는 그냥 이미지 반환했네.

    return "row4넣을차트3.png"

def polio():
    return "polio_img.png"

def pie_chart():
    import plotly.graph_objects as go
    colors = ["#013848", "#0085AF", "#00A378"]

    labels = ['A+', 'A', 'B+', 'B', 'C', 'D']
    values = ['12', '179', '285', '153', '300', '21']

    fig = go.Figure(data=[go.Pie(labels=labels,
                                 values=values,
                                 textinfo='label+percent',
                                 insidetextorientation='radial',
                                 sort=False,
                                 marker=dict(colors=colors,
                                             line=dict(color='white', width=1)
                                             )
                                 )
                          ])
    res = fig.to_json()
    return res



def naver_news(news_rows=8):
    import requests
    from bs4 import BeautifulSoup
    import time

    news_list_list = [['http://www.wikileaks-kr.org/news/articleView.html?idxno=128676', 'GS건설, ESG 활동 담은 2022년 지속가능경영보고서 발간', '7분 전', 'https://search.pstatic.net/common/?src=https%3A%2F%2Fimgnews.pstatic.net%2Fimage%2Forigin%2F5622%2F2022%2F07%2F25%2F64980.jpg&type=ff264_180&expire=2&refresh=true' ],
                      ['http://www.ekoreanews.co.kr/news/articleView.html?idxno=61931', "위니아에이드 임직원, '플로깅 데이' 진행", '1시간 전','https://search.pstatic.net/common/?src=https%3A%2F%2Fimgnews.pstatic.net%2Fimage%2Forigin%2F5562%2F2022%2F07%2F25%2F24004.jpg&type=ff264_180&expire=2&refresh=true'],
                      ['http://www.fnnews.com/news/202207251116062409', 'SK네트웍스 "코로나 어려움 속 지난해 환경성과 252억원 창출"', '2시간 전',"https://search.pstatic.net/common/?src=https%3A%2F%2Fimgnews.pstatic.net%2Fimage%2Forigin%2F014%2F2022%2F07%2F25%2F4872448.jpg&type=ff264_180&expire=2&refresh=true"],
                      ['http://www.fnnews.com/news/202207251134499724', '"지구와 함께 달립니다"...위니아에이드, \'플로깅데이\' 실시', '2시간 전','https://search.pstatic.net/common/?src=https%3A%2F%2Fimgnews.pstatic.net%2Fimage%2Forigin%2F025%2F2022%2F07%2F25%2F3211908.jpg&type=ff264_180&expire=2&refresh=true'],
                      ['http://www.metroseoul.co.kr/article/20220725500189', '롯데칠성, 플라스틱 다이어트…용기 무게 10% 경량화', '3시간 전','https://search.pstatic.net/common/?src=https%3A%2F%2Fimgnews.pstatic.net%2Fimage%2Forigin%2F5286%2F2022%2F07%2F25%2F455081.jpg&type=ff264_180&expire=2&refresh=true'],
                      ['https://www.enetnews.co.kr/news/articleView.html?idxno=6326', "중진공, '수해·태풍 피해기업' 대상\xa0재해신속지원단 운영", '3시간 전','https://search.pstatic.net/common/?src=https%3A%2F%2Fimgnews.pstatic.net%2Fimage%2Forigin%2F5866%2F2022%2F07%2F25%2F585.jpg&type=ff264_180&expire=2&refresh=true']]

    return news_list_list[:news_rows]








# 포트폴리오 시각화
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import warnings
import datetime as dt
warnings.filterwarnings('ignore')
from varname import nameof
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
fin_df = pd.read_csv('data/fin_dataset2.csv', encoding = 'utf-8')
df_kodex_kospi = fdr.DataReader('226490', '2021-01-01', '2022-03-31') #
df_arirang_esg = fdr.DataReader('278420 ', '2021-01-01', '2022-03-31')

# 회귀 그래프를 그리는 그래프
def reg_plot(df):
    fig = px.scatter(df, x="Market Returns", y="VAR Return", trendline="ols")
    fig.update_layout(
                    title = 'Beta',
                    xaxis_title = 'Market Returns',
                    yaxis_title = 'VAR Return',
                    font = dict(
                                # family = 'Courier New, monospace',
                                size = 12,
                                color = 'Black'
                            ),
                    autosize=False,
                    width=550,
                    height=500,
                    margin=dict(
                                    l=50,
                                    r=50,
                                    b=100,
                                    t=100,
                                    pad=4
                                ),


                    )
    res = fig.to_json()
    return res



# 상관계수를 산출하는 함수
def corr_plot(df1,df2):
    df = pd.DataFrame({'Market Returns':df1['Change'], 'VAR Return':df2['Change']})
    cov = np.cov(df['Market Returns'], df['VAR Return'])[0, 1]  # 공분산
    xsd = np.std(df['Market Returns'], ddof=1)   # x의 표본표준편차
    ysd = np.std(df['VAR Return'], ddof=1)   # y의 표본표준편차
    corr = cov / ( xsd * ysd )
    return reg_plot(df)

# 데이터를 ETF 기준으로 분리하기 위한 함수
def df_transfomer(df, standard = 'High', agg = 'sum'):

    if standard.lower() == 'sin_stock':
        df = df.groupby('ESG_기준').get_group(standard.lower()).groupby('날짜').sum().reset_index()
        df['기준'] = 'sin_stock'
    else:
        df_20 = df.groupby('ESG_기준').get_group('20_'+standard).groupby('날짜').sum().reset_index()
        df_20 = df_20[df_20['날짜'].apply(lambda x : dt.datetime.strptime(x, '%Y-%m-%d').year==2020)]
        df_20['기준'] = '20_'+standard
        df_21 = df.groupby('ESG_기준').get_group('21_'+standard).groupby('날짜').sum().reset_index()
        df_21 = df_21[df_21['날짜'].apply(lambda x : dt.datetime.strptime(x, '%Y-%m-%d').year>=2021)]
        df_21['기준'] = '21_'+standard
        df = pd.concat([df_20,df_21],axis=0 )
    return df

# 직접 정의한 ETF의 지수를 구하기 위한 함수
def etf_index(df, date_list):

    etf_idx=[]
    for date in date_list:
        start = np.sum(df[df['날짜']=='2020-01-02']['종가']\
            *df[df['날짜']=='2020-01-02']['유동주식'])
        end = np.sum(df[df['날짜']==date]['종가']\
            *df[df['날짜']==date]['유동주식'])
        etf_idx.append((end/start)*100)
    df['ETF_지수'] = etf_idx
    name = df['기준'][0]
    return df


sin_stock = etf_index(df_transfomer(fin_df, 'sin_stock'),fin_df['날짜'].unique())
etf_High = etf_index(df_transfomer(fin_df, 'High'),fin_df['날짜'].unique())
etf_Low = etf_index(df_transfomer(fin_df, 'Low'),fin_df['날짜'].unique())
# 21년도의 데이터만 사용
esg_High_21 = etf_High[(pd.to_datetime(etf_High['날짜']).dt.year>=2021)& (pd.to_datetime(etf_High['날짜'])<='2022-03-31')].reset_index()
esg_Low_21 =   etf_Low[(pd.to_datetime(etf_Low['날짜']).dt.year>=2021) & (pd.to_datetime(etf_Low['날짜'])<='2022-03-31')].reset_index()
sin_stock =   sin_stock[(pd.to_datetime(sin_stock['날짜']).dt.year>=2021) & (pd.to_datetime(sin_stock['날짜'])<='2022-03-31')].reset_index()

# 변환율을 계산하여 칼럼에 추가
esg_High_21['Change'] = (esg_High_21['ETF_지수'] - esg_High_21['ETF_지수'].shift(1)) / esg_High_21['ETF_지수'].shift(1)
esg_Low_21['Change'] = (esg_Low_21['ETF_지수'] - esg_Low_21['ETF_지수'].shift(1)) / esg_Low_21['ETF_지수'].shift(1)
sin_stock['Change'] = (sin_stock['ETF_지수'] - sin_stock['ETF_지수'].shift(1)) / sin_stock['ETF_지수'].shift(1)

# kodex 데이터의 인덱스 초기화
df_kodex_kospi.reset_index(inplace=True)
esg_High_21['Change'][0]=0
esg_Low_21['Change'][0]=0
sin_stock['Change'][0]=0

df_arirang_esg.reset_index(inplace=True)

def stock_20_21():
    import plotly.graph_objs as go
    from chart_studio import plotly
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    init_notebook_mode(connected=True)
    esg_high_21 = fin_df.groupby('ESG_기준').get_group('21_High').groupby('날짜').sum().reset_index()
    esg_low_21 = fin_df.groupby('ESG_기준').get_group('21_Low').groupby('날짜').sum().reset_index()
    sin_stock = fin_df.groupby('ESG_기준').get_group('sin_stock').groupby('날짜').sum().reset_index()
    stock_list = [esg_high_21, esg_low_21, sin_stock]
    data = []
    for df in stock_list:
        data.append(go.Candlestick(x=df['날짜'],
                                   open=df['시가'],
                                   high=df['최고가'],
                                   low=df['최저가'],
                                   close=df['종가'])
                    )
    dict_list = [dict(label='ESG High',
                      method='update',
                      args=[{'visible': [True, False, False]},
                            {'title': 'ESG High' + ' Stock'}]),
                 dict(label='ESG Low',
                      method='update',
                      args=[{'visible': [False, True, False]},
                            {'title': 'ESG Low' + ' Stock'}]),
                 dict(label='Sin Stock',
                      method='update',
                      args=[{'visible': [False, False, True]},
                            {'title': 'Sin' + ' Stock'}]),
                 dict(label='Reset',
                      method='update',
                      args=[{'visible': [True, True, True]},
                            {'title': '전체'}])]
    updatemenus = list([
        dict(type="buttons",
             active=-1,
             buttons=dict_list
             ,
             )
    ])

    layout = dict(title='2021-2022 Stock', showlegend=True,
                  updatemenus=updatemenus)

    fig = go.Figure(data=data, layout=layout)

    ## offline notebook 용으로 iplot
    iplot(fig, filename='base-bar')
    res_str = fig.to_json()  # ---tohtml
    # print(res_str)
    return res_str

def abcdsin():
    esg_df = pd.read_html(
        "http://www.cgs.or.kr/business/esg_tab04.jsp?pg=1&pp=1005&skey=&svalue=&sfyear=2021&styear=2021&sgtype=&sgrade=#ui_contents")[
        0]
    esg_df.set_index('NO', inplace=True)
    esg_df.sort_index(inplace=True)
    esg_df = esg_df.iloc[:, :-1]
    esg_fin = pd.merge(esg_df, fin_df, left_on='기업코드 첨부내용', right_on='종목번호', how='right')
    esg_fin_df = esg_fin[
        ['기업명_x', 'ESG등급', '환경', '사회', '지배구조', '날짜', '종목번호', 'ESG_기준', '시가', '최고가', '최저가', '종가', '거래량', '일일주가변동률',
         '유동주식']]

    # import plotly.plotly as py
    import plotly.graph_objs as go
    from chart_studio import plotly
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    init_notebook_mode(connected=True)
    ESG_Ap_df = esg_fin_df.groupby('ESG등급').get_group('A+').groupby('날짜').sum().reset_index()
    ESG_A_df = esg_fin_df.groupby('ESG등급').get_group('A').groupby('날짜').sum().reset_index()
    ESG_Bp_df = esg_fin_df.groupby('ESG등급').get_group('B+').groupby('날짜').sum().reset_index()
    ESG_B_df = esg_fin_df.groupby('ESG등급').get_group('B').groupby('날짜').sum().reset_index()
    ESG_C_df = esg_fin_df.groupby('ESG등급').get_group('C').groupby('날짜').sum().reset_index()
    ESG_D_df = esg_fin_df.groupby('ESG등급').get_group('D').groupby('날짜').sum().reset_index()
    ESG_G_df = [ESG_Ap_df, ESG_A_df, ESG_Bp_df, ESG_B_df, ESG_C_df, ESG_D_df]
    data = []
    for df in ESG_G_df:
        data.append(go.Candlestick(x=df['날짜'],
                                   open=df['시가'],
                                   high=df['최고가'],
                                   low=df['최저가'],
                                   close=df['종가'])
                    )
    dict_list = [dict(label='GRADE A+',
                      method='update',
                      args=[{'visible': [True, False, False, False, False, False]},
                            {'title': 'ESG GRADE A+'}]),
                 dict(label='GRADE A',
                      method='update',
                      args=[{'visible': [False, True, False, False, False, False]},
                            {'title': 'ESG GRADE A'}]),
                 dict(label='GRADE B+',
                      method='update',
                      args=[{'visible': [False, False, True, False, False, False]},
                            {'title': 'ESG GRADE B+'}]),
                 dict(label='GRADE B',
                      method='update',
                      args=[{'visible': [False, False, False, True, False, False]},
                            {'title': 'ESG GRADE B'}]),
                 dict(label='GRADE C',
                      method='update',
                      args=[{'visible': [False, False, False, False, True, False]},
                            {'title': 'ESG GRADE C'}]),
                 dict(label='GRADE D',
                      method='update',
                      args=[{'visible': [False, False, False, False, False, True]},
                            {'title': 'ESG GRADE D'}]),
                 dict(label='Reset',
                      method='update',
                      args=[{'visible': [True, True, True, True, True, True]},
                            {'title': '전체'}])]
    updatemenus = list([
        dict(type="buttons",
             active=-1,
             buttons=dict_list
             ,
             )
    ])

    layout = dict(title='2021-2022 Stock', showlegend=False,
                  updatemenus=updatemenus)

    fig = go.Figure(data=data, layout=layout)

    ## offline notebook 용으로 iplot
    iplot(fig, filename='base-bar')
    res_str = fig.to_json()  # ---tohtml
    # print(res_str)
    return res_str

def 차트중간에넣을꺼1():
    m = esg_High_21['Change'].mean() * 365
    st = esg_High_21['Change'].std() * np.sqrt(365)
    rf = 1.45 * 0.01
    cal_df = pd.DataFrame({'ER': [rf, m], 'Stdev': [0, st]})
    fig = px.line(cal_df, x='Stdev', y='ER', title='Capital Allocation Line of ESG Top Index')
    # fig.update_layout(autosize = False )
    res_str = fig.to_json()  # ---tohtml
    # print(res_str)
    return res_str

def 차트중간에넣을꺼2():
    m = esg_Low_21['Change'].mean() * 365
    st = esg_Low_21['Change'].std() * np.sqrt(365)
    rf = 1.45 * 0.01
    cal_df = pd.DataFrame({'ER': [rf, m], 'Stdev': [0, st]})
    fig = px.line(cal_df, x='Stdev', y='ER', title='Capital Allocation Line of Anti-ESG Index')
    res_str = fig.to_json()  # ---tohtml
    # print(res_str)
    return res_str

def 차트중간에넣을꺼3():
    m = sin_stock['Change'].mean() * 365
    st = sin_stock['Change'].std() * np.sqrt(365)
    rf = 1.45 * 0.01
    cal_df = pd.DataFrame({'ER': [rf, m], 'Stdev': [0, st]})
    fig = px.line(cal_df, x='Stdev', y='ER', title='Capital Allocation Line of Sin Index')
    res_str = fig.to_json()  # ---tohtml
    # print(res_str)
    return res_str

def 차트중간에넣을꺼4():
    m = df_kodex_kospi['Change'].mean() * 365
    st = df_kodex_kospi['Change'].std() * np.sqrt(365)
    rf = 1.45 * 0.01
    cal_df = pd.DataFrame({'ER': [rf, m], 'Stdev': [0, st]})
    fig = px.line(cal_df, x='Stdev', y='ER', title='Capital Market Line (KOSPI)')
    res_str = fig.to_json()  # ---tohtml
    # print(res_str)
    return res_str


@app.route('/') # 주소가 있다
def index():
    news_list_list = naver_news()
    esg_ratio = pie_chart()
    kodex200_dic = kodex200_info()
    line_chart_str = etf_line()
    line_chart_str2 = volume()
    stock_2021_2021 = stock_20_21()
    gradeabcd = abcdsin()
    polio_img_path = polio()

    # tupl str dict array...  list는 전송불가..
    return render_template("index.html" # 결과를 보내줘라
                            , STOCK_20_21 =stock_2021_2021
                           , GRADE_ABCD= gradeabcd
                            , NEWS_LIST_LIST  = news_list_list # return된것 그릇 담아주기
                            , KODEX200_DICT = kodex200_dic
                            , LINE_CHART_STR  = line_chart_str
                            , LINE_CHART_STR2 = line_chart_str2
                            ,PIE_CHART_STR = esg_ratio
                           ,POLIO_STR = polio_img_path

                           )


@app.route("/index2") # 주소가 있다.
def index2():
    esg_hi_mvp = row4넣을차트1()
    esg_low_mvp = row4넣을차트2()
    sin_mvp = row4넣을차트3()
    esg_hi_kodex = corr_plot(esg_High_21, df_kodex_kospi )
    esg_hiari = corr_plot(esg_High_21, df_arirang_esg)
    esg_lowko= corr_plot(esg_Low_21, df_kodex_kospi)
    esg_lowari = corr_plot(esg_Low_21, df_arirang_esg)
    sin_kodex= corr_plot(sin_stock, df_kodex_kospi)
    sin_ari = corr_plot(sin_stock, df_arirang_esg)
    row1중간 = 차트중간에넣을꺼1()
    row2중간 = 차트중간에넣을꺼2()
    row3중간 = 차트중간에넣을꺼3()
    row4중간 = 차트중간에넣을꺼4()

    return render_template("index2.html"
                           ,ESG_HIGH_STR = esg_hi_kodex
                           , ESG_HI_MVP =esg_hi_mvp
                           ,ESG_LOW_MVP =esg_low_mvp
                           ,SIN_MVP = sin_mvp
                            , ROW1_MED=row1중간
                           ,ROW2_MED =row2중간
                           ,ROW3_MED = row3중간
                           ,ROW4_MED=row4중간

                           ,ESG_HI_ARI =esg_hiari
                           ,ESG_LOW_KO = esg_lowko
                           ,ESG_LOW_ARI =esg_lowari
                           ,SIN_KO =sin_kodex
                           ,SIN_ARI =sin_ari)
                           #------------------------------------------------------------------------------




if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5555)

