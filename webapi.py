import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import yfinance as yt
import streamlit as st
from streamlit_option_menu import option_menu

#MAIN MODEL

main_model = load_model('closemodel.h5')
main_model_label = pickle.load(open('labelenodermain.l', 'rb'))
main_model_scale1 = pickle.load(open('scaler1.s', 'rb'))
main_model_scale2 = pickle.load(open('scaler2.s', 'rb'))


#NIFTY20 MODELS AND SCALER

asianpaint_mod = pickle.load(open('asianpaint.sas', 'rb'))
asianpaint_scale = pickle.load(open('asianpaint_scale.l', 'rb'))

bajajfinsv_mod = pickle.load(open('bajajfinsv.sas', 'rb')) 
bajajfinsv_scale = pickle.load(open('bajajfinsv_scale.l', 'rb'))

bharatiartl_mod = pickle.load(open('bhartiartl.sas', 'rb'))
bharatiartl_scale = pickle.load(open('bhartiartl_scale.l', 'rb'))

dmart_mod = pickle.load(open('dmart.sas', 'rb'))
dmart_scale = pickle.load(open('dmart_scale.l', 'rb'))

hcltech_mod = pickle.load(open('hcltech.sas', 'rb'))
hcltech_scale = pickle.load(open('hcltech_scale.l', 'rb'))

hindunilvr_mod = pickle.load(open('hindunilvr.sas', 'rb'))
hindunilvr_scale = pickle.load(open('hindunilvr_scale.l', 'rb'))

infy_mod = pickle.load(open('infy.sas', 'rb'))
infy_scale = pickle.load(open('infy_scale.l', 'rb'))

itc_mod = pickle.load(open('itc.sas', 'rb'))
itc_scale = pickle.load(open('itc_scale.l', 'rb'))

lt_mod = pickle.load(open('lt.sas', 'rb'))
lt_scale = pickle.load(open('lt_scale.l', 'rb'))

maruti_mod = pickle.load(open('maruti.sas', 'rb'))
maruti_scale = pickle.load(open('maruti_scale.l', 'rb'))

reliance_mod = pickle.load(open('reliance.sas', 'rb'))
reliance_scale = pickle.load(open('reliance_scale.l', 'rb'))

tcs_mod = pickle.load(open('tcs.sas', 'rb'))
tcs_scale = pickle.load(open('tcs_scale.l', 'rb'))

ultracemco_mod = pickle.load(open('ultracemco.sas', 'rb'))
ultracemco_scale = pickle.load(open('ultracemco_scale.l', 'rb'))

wipro_mod = pickle.load(open('wipro.sas', 'rb'))
wipro_scale = pickle.load(open('wipro_scale.l', 'rb'))

#NIFTY BANK MODELS AND SCALER

aubank_mod = pickle.load(open('aubank.sas', 'rb'))
aubank_scale = pickle.load(open('aubank_scale.l', 'rb'))

axisbank_mod = pickle.load(open('axisbank.sas', 'rb'))
axisbank_scale = pickle.load(open('axisbank_scale.l', 'rb'))

bandhanbnk_mod = pickle.load(open('bandhanbnk.sas', 'rb'))
bandhanbnk_scale = pickle.load(open('bandhanbnk_scale.l', 'rb'))

bankbaroda_mod = pickle.load(open('bankbaroda.sas', 'rb'))
bankbaroda_scale = pickle.load(open('bankbaroda_scale.l', 'rb'))

federalbnnk_mod = pickle.load(open('federalbnk.sas', 'rb'))
federalbnk_scale = pickle.load(open('federalbnk_scale.l', 'rb'))

hdfcbank_mod = pickle.load(open('hdfcbank.sas', 'rb'))
hdfcbank_scale = pickle.load(open('hdfcbank_scale.l', 'rb'))

icicibank_mod = pickle.load(open('icicibank.sas', 'rb'))
icicibank_scale = pickle.load(open('icicibank_scale.l', 'rb'))

idfc_mod = pickle.load(open('idfc.sas', 'rb'))
idfc_scale = pickle.load(open('idfc_scale.l', 'rb'))

indusindbk_mod = pickle.load(open('indusindbk.sas', 'rb'))
indusindbk_scale = pickle.load(open('indusindbk_scale.l', 'rb'))

kotakbank_mod = pickle.load(open('kotakbank.sas', 'rb'))
kotakbank_scale = pickle.load(open('kotakbank_scale.l', 'rb'))

pnb_mod = pickle.load(open('pnb.sas', 'rb'))
pnb_scale = pickle.load(open('pnb_scale.l', 'rb'))

sbin_mod = pickle.load(open('sbin.sas', 'rb'))
sbin_scale = pickle.load(open('sbin_scale.l', 'rb'))



#PHARMA SECTOR MODELS AND SCALER

alkem_mod = pickle.load(open('alkem_mode.sas', 'rb'))
alkem_scale = pickle.load(open('alkem_scale.l', 'rb'))

auropharma_mod = pickle.load(open('auropharma.sas', 'rb'))
auropharma_scale = pickle.load(open('auropharma_scale.l', 'rb'))

biocon_mod = pickle.load(open('biocon.sas', 'rb'))
biocon_scale = pickle.load(open('biocon_scale.l', 'rb'))

cipla_mod = pickle.load(open('cipla.sas', 'rb'))
cipla_scale = pickle.load(open('cipla_scale.l', 'rb'))

drreddy_mod = pickle.load(open('drreddy.sas', 'rb'))
drreddy_scale = pickle.load(open('drreddy_scale.l', 'rb'))

glenmark_mod = pickle.load(open('glenmark.sas', 'rb'))
glenark_scale = pickle.load(open('glenmark_scale.l', 'rb'))

ipcalab_mod = pickle.load(open('ipcalab.sas', 'rb'))
ipcalab_scale = pickle.load(open('ipcalab_scale.l', 'rb'))

jbchepharm_mod = pickle.load(open('jbchepharm.sas', 'rb'))
jbchepharm_scale = pickle.load(open('jbchepharm_scale.l', 'rb'))

lupin_mod = pickle.load(open('lupin.sas', 'rb'))
lupin_scale = pickle.load(open('lupin_scale.l', 'rb'))

sunpharma_mod = pickle.load(open('sunpharma.sas', 'rb'))
sunpharma_scale = pickle.load(open('sunpharma_scale.l', 'rb'))

suven_mod = pickle.load(open('suven.sas', 'rb'))
suven_scale = pickle.load(open('suven_scale.l', 'rb'))

syngene_mod = pickle.load(open('syngene.sas', 'rb'))
syngene_scale = pickle.load(open('syngene_scale.l', 'rb'))

zyduslife_mod = pickle.load(open('zyduslife.sas', 'rb'))
zyduslife_scale = pickle.load(open('zyduslife_scale.l', 'rb'))

st.sidebar.image('logo.png')
with st.sidebar:
    
    selected = option_menu("Dashboard", ['HOME',
                                         'PREDICT',
                                         'NIFTY 20',
                                         'NIFTY BANK',
                                         'NIFTY PHARMA',
                                         'ABOUT',
                                         'USAGE',
                                         'CONTACT & SUPPORT',
                                         ], icons = ['house-fill',
                                                     'currency-rupee',
                                                     'buildings',
                                                     'bank',
                                                     'clipboard2-pulse-fill',
                                                     'file-person',
                                                     'mouse2-fill',
                                                     'headset'],default_index = 0)   

if (selected == 'HOME'):
    
    st.title("See where the stock market is headed before it gets there.")
    st.subheader("Your window into the future of the stock market.")
    st.image("graph3.jpg")
    st.header("Your strong companion for your Investment Endeavour")
    st.header("Our AI-powered API provides a range of features to enhance your investment strategies:")
    st.markdown('''> Real-time Price Predictions: Stay ahead of the market with real-time predictions
                for over 10,000 stocks.''')
    st.markdown('''> Historical Data Analysis: Gain insights from comprehensive historical data to 
                identify trends and make informed investment decisions.''')
    st.markdown('''Our AI models are trained on a vast collection of high-quality data from reputable 
                sources, ensuring the accuracy and reliability of our predictions. We continuously monitor
                and refine our models to maintain their effectiveness in an ever-changing market.''')
    
    
if (selected == 'PREDICT'):

    
    st.title("PREDICT")
    st.header("Predicts the Close price of about 1725+ companies on The National stock exchange")
    st.subheader("FEATURES")
    st.markdown("> Can give close to accurate values")
    st.markdown("> More reliable")
    st.markdown("> Trained on larger data and multiple features")
    st.subheader("Give the company stock symbol")
    
    
    stock_search = st.text_input("stock symbol")
    symbol = stock_search.upper()

    df = yt.download(f"{symbol}.NS")
    df = df.tail(1)
    df = df.drop(['Close'], axis = 1)
    st.dataframe(data = df, width = 2050)
    df = df[['Open','High','Low','Adj Close','Volume']]
    name = f"{symbol}.NS"
    rep = df.shape[0]

    comp = pd.DataFrame({"Company name": np.concatenate([np.repeat(name,rep)])})
    comp = np.asarray(comp['Company name'])
    df['Company Name'] = comp.tolist()
    
    pred = ''
    
    if st.button("PREDICT"):
        
        lbl = df["Company Name"].values
        label = main_model_label.transform(lbl)

        df = df.drop(["Company Name"], axis = 1)
        df["Company_name"] = label.tolist()
        df = df.values
        arr = np.asarray(df)
        tr = main_model_scale2.transform(arr)
        
        pred = main_model.predict(tr)
        pred =  main_model_scale1.inverse_transform(pred)
        pred = pred[0]

    st.success(pred)
    
    
    
if (selected == 'NIFTY 20'):
    
    st.title("NIFTY 20")
    st.header("Predicts the close prices for Top 20 companies of NSE by market cap")
    st.subheader("About the models")
    st.markdown("> Trained on single comapany data set")
    st.markdown("> Trained on a single feature, Only on close prices")
    st.markdown("> Takes the past 60 days close prices to predict the next day close price")
    st.markdown("> Less accurate prediction when compared to the Main model")
    st.markdown("> Useful in confirming trends")
    st.subheader("Click the Stock symbols to get your prediction")
    
    asianpaint = ''
    if st.button("ASIANPAINT"):

        scaler = asianpaint_scale
        mod = asianpaint_mod

        stock_search = 'ASIANPAINT'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        asianpaint = pred_price
        print(asianpaint)
    st.success(asianpaint)
    
    bajajfinsv = ''
    if st.button("BAJAJFINSV"):

        scaler = bajajfinsv_scale
        mod = bajajfinsv_mod

        stock_search = 'BAJAJFINSV'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        bajajfinsv = pred_price
        print(bajajfinsv)
    st.success(bajajfinsv)
    
    
    bharatiartl = ''
    if st.button("BHARATIARTL"):

        scaler = bharatiartl_scale
        mod = bharatiartl_mod

        stock_search = 'BHARATIARTL'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        bharatiartl = pred_price
        print(bharatiartl)
    st.success(bharatiartl)
    
    
    dmart = ''
    if st.button("DMART"):

        scaler = dmart_scale
        mod = dmart_mod

        stock_search = 'DMART'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        dmart = pred_price
        print(dmart)
    st.success(dmart)
    
    
    hcltech = ''
    if st.button("HCLTECH"):

        scaler = hcltech_scale
        mod = hcltech_mod

        stock_search = 'HCLTECH'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        hcltech = pred_price
        print(hcltech)
    st.success(hcltech)
    
    
    hindunilvr = ''
    if st.button("HINDUNILVR"):

        scaler = hindunilvr_scale
        mod = hindunilvr_mod

        stock_search = 'HINDUNILVR'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        hindunilvr = pred_price
        print(hindunilvr)
    st.success(hindunilvr)
    
    
    infy = ''
    if st.button("INFY"):

        scaler = infy_scale
        mod = infy_mod

        stock_search = 'INFY'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        infy = pred_price
        print(infy)
    st.success(infy)
    
    
    itc = ''
    if st.button("ITC"):

        scaler = bajajfinsv_scale
        mod = bajajfinsv_mod

        stock_search = 'ITC'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        itc = pred_price
        print(itc)
    st.success(itc)
    
    
    lt = ''
    if st.button("LT"):

        scaler = lt_scale
        mod = lt_mod

        stock_search = 'LT'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        lt = pred_price
        print(lt)
    st.success(lt)
    
    maruti = ''
    if st.button("MARUTI"):

        scaler = maruti_scale
        mod = maruti_mod

        stock_search = 'MARUTI'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        maruti = pred_price
        print(maruti)
    st.success(maruti)
    
    
    reliance = ''
    if st.button("RELIANCE"):

        scaler = reliance_scale
        mod = reliance_mod

        stock_search = 'RELIANCE'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        reliance = pred_price
        print(reliance)
    st.success(reliance)
    
    
    tcs = ''
    if st.button("TCS"):

        scaler = tcs_scale
        mod = tcs_mod

        stock_search = 'TCS'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        tcs = pred_price
        print(tcs)
    st.success(tcs)
    
    
    ultracemco = ''
    if st.button("ULTRACEMCO"):

        scaler = ultracemco_scale
        mod = ultracemco_mod

        stock_search = 'ULTRACEMCO'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        ultracemco = pred_price
        print(ultracemco)
    st.success(ultracemco)
    
    
    wipro = ''
    if st.button("WIPRO"):

        scaler = wipro_scale
        mod = wipro_mod

        stock_search = 'WIPRO'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        wipro = pred_price
        print(wipro)
    st.success(wipro)
    
       
    
if (selected == 'NIFTY BANK'):
    
    st.title("NIFTY BANK")
    st.header("Predicts the close prices for Top capitalised banking stocks")
    st.subheader("About the models")
    st.markdown("> Trained on single comapany data set")
    st.markdown("> Trained on a single feature, Only on close prices")
    st.markdown("> Takes the past 60 days close prices to predict the next day close price")
    st.markdown("> Less accurate prediction when compared to the Main model")
    st.markdown("> Useful in confirming trends")
    st.subheader("Click the Stock symbols to get your prediction")
    
    aubank = ''
    if st.button("AUBANK"):

        scaler = aubank_scale
        mod = aubank_mod

        stock_search = 'AUBANK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        aubank = pred_price
        print(aubank)
    st.success(aubank)
    
    axisbank = ''
    if st.button("AXISBANK"):

        scaler = axisbank_scale
        mod = axisbank_mod

        stock_search = 'AXISBANK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        axisbank = pred_price
        print(axisbank)
    st.success(axisbank)
    
    bandhanbnk = ''
    if st.button("BANDHANBNK"):

        scaler = bandhanbnk_scale
        mod = bandhanbnk_mod

        stock_search = 'BANDHANBNK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        bandhanbnk = pred_price
        print(bandhanbnk)
    st.success(bandhanbnk)
    
    
    bankbaroda = ''
    if st.button("BANKBARODA"):

        scaler = bankbaroda_scale
        mod = bankbaroda_mod

        stock_search = 'BANKBARODA'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        bankbaroda = pred_price
        print(bankbaroda)
    st.success(bankbaroda)
    
    
    federalbnk = ''
    if st.button("FEDERALBNK"):

        scaler = federalbnk_scale
        mod = federalbnnk_mod

        stock_search = 'FEDERALBNK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        federalbnk = pred_price
        print(federalbnk)
    st.success(federalbnk)
    
    
    hdfcbank = ''
    if st.button("HDFCBANK"):

        scaler = hdfcbank_scale
        mod = hdfcbank_mod

        stock_search = 'HDFCBANK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        hdfcbank = pred_price
        print(hdfcbank)
    st.success(hdfcbank)
    
    
    icicibank = ''
    if st.button("ICICIBANK"):

        scaler = icicibank_scale
        mod = icicibank_mod

        stock_search = 'ICICIBANK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        icicibank = pred_price
        print(icicibank)
    st.success(icicibank)
    
    
    idfc = ''
    if st.button("IDFC"):

        scaler = idfc_scale
        mod = idfc_mod

        stock_search = 'IDFC'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        idfc = pred_price
        print(idfc)
    st.success(idfc)
    
    
    indusindbk = ''
    if st.button("INDUSINDBK"):

        scaler = indusindbk_scale
        mod = indusindbk_mod

        stock_search = 'INDUSINDBK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        indusindbk = pred_price
        print(indusindbk)
    st.success(indusindbk)
    
    
    kotakbank = ''
    if st.button("KOTAKBANK"):

        scaler = kotakbank_scale
        mod = kotakbank_mod

        stock_search = 'KOTAKBANK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        kotakbank = pred_price
        print(kotakbank)
    st.success(kotakbank)
    
    
    pnb = ''
    if st.button("PNB"):

        scaler = pnb_scale
        mod = pnb_mod

        stock_search = 'PNB'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        pnb = pred_price
        print(pnb)
    st.success(pnb)
    
    
    sbin = ''
    if st.button("SBIN"):

        scaler = sbin_scale
        mod = sbin_mod

        stock_search = 'SBIN'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        sbin = pred_price
        print(sbin)
    st.success(sbin)
    
    

if (selected == 'NIFTY PHARMA'):
    
    st.title("NIFTY PHARMA")
    st.header("Predicts the close prices for Top capitalised pharma stocks")
    st.subheader("About the models")
    st.markdown("> Trained on single comapany data set")
    st.markdown("> Trained on a single feature, Only on close prices")
    st.markdown("> Takes the past 60 days close prices to predict the next day close price")
    st.markdown("> Less accurate prediction when compared to the Main model")
    st.markdown("> Useful in confirming trends")
    st.subheader("Click the Stock symbols to get your prediction")
    
    
    alkem = ''
    if st.button("ALKEM"):

        scaler = alkem_scale
        mod = alkem_mod

        stock_search = 'ALKEM'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        alkem = pred_price
        print(alkem)
    st.success(alkem)
    
    
    auropharma = ''
    if st.button("AUROPHARMA"):

        scaler = auropharma_scale
        mod = auropharma_mod

        stock_search = 'AUROPHARMA'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        auropharma = pred_price
        print(auropharma)
    st.success(auropharma)
    
    
    biocon = ''
    if st.button("BIOCON"):

        scaler = biocon_scale
        mod = biocon_mod

        stock_search = 'BIOCON'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        biocon = pred_price
        print(biocon)
    st.success(biocon)
    
    
    cipla = ''
    if st.button("CIPLA"):

        scaler = cipla_scale
        mod = cipla_mod

        stock_search = 'CIPLA'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        cipla = pred_price
        print(cipla)
    st.success(cipla)
    
    
    drreddy = ''
    if st.button("DRREDDY"):

        scaler = drreddy_scale
        mod = drreddy_mod

        stock_search = 'DRREDDY'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        drreddy = pred_price
        print(drreddy)
    st.success(drreddy)
    
    
    glenmark = ''
    if st.button("GLENMARK"):

        scaler = glenark_scale
        mod = glenmark_mod

        stock_search = 'GLENMARK'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        glenmark = pred_price
        print(glenmark)
    st.success(glenmark)
    
    
    ipcalab = ''
    if st.button("IPCALAB"):

        scaler = ipcalab_scale
        mod = ipcalab_mod

        stock_search = 'IPCALAB'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        ipcalab = pred_price
        print(ipcalab)
    st.success(ipcalab)
    
    
    jbchepharm = ''
    if st.button("JBCHEPHARM"):

        scaler = jbchepharm_scale
        mod = jbchepharm_mod

        stock_search = 'JBCHEPHARM'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        jbchepharm = pred_price
        print(jbchepharm)
    st.success(jbchepharm)
    
    
    lupin = ''
    if st.button("LUPIN"):

        scaler = lupin_scale
        mod = lupin_mod

        stock_search = 'LUPIN'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        lupin = pred_price
        print(lupin)
    st.success(lupin)
    
    
    sunpharma = ''
    if st.button("SUNPHARMA"):

        scaler = sunpharma_scale
        mod = sunpharma_mod

        stock_search = 'SUNPHARMA'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        sunpharma = pred_price
        print(sunpharma)
    st.success(sunpharma)
    
    
    suven = ''
    if st.button("SUVEN"):

        scaler = suven_scale
        mod = suven_mod

        stock_search = 'SUVEN'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        suven = pred_price
        print(suven)
    st.success(suven)
    
    
    syngene = ''
    if st.button("SYNGENE"):

        scaler = syngene_scale
        mod = syngene_mod

        stock_search = 'SYNGENE'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        syngene = pred_price
        print(syngene)
    st.success(syngene)
    
    zyduslife = ''
    if st.button("ZYDUSLIFE"):

        scaler = zyduslife_scale
        mod = zyduslife_mod

        stock_search = 'ZYDUSLIFE'
        symbol = stock_search.upper()
        new_df = yt.download(f"{symbol}.NS")
        new_df = new_df.filter(['Close'])

        last_60_days = new_df[-60:].values

        last_scaled = scaler.transform(last_60_days)

        Xtesta = []
        Xtesta.append(last_scaled)
        Xtesta = np.array(Xtesta)
        Xtesta = np.reshape(Xtesta, (Xtesta.shape[0], Xtesta.shape[1], 1))

        pred_price = mod.predict(Xtesta)
        pred_price = scaler.inverse_transform(pred_price)
        zyduslife = pred_price
        print(zyduslife)
    st.success(zyduslife)
    
    
    
if (selected == 'ABOUT'):
    
    st.title("About the API")
    st.image("pngwing.com.png")
    st.header("Empowering Investors with AI-Powered Stock Market Predictions")
    st.markdown('''Unleash the power of cutting-edge artificial intelligence to gain 
                valuable insights into the ever-evolving stock market landscape. 
                Our innovative web application harnesses the strength of deep learning 
                algorithms to provide comprehensive stock market predictions, 
                empowering investors to make informed decisions and navigate 
                the financial world with greater confidence.''')
    
    st.header("A Free Solution for Informed Investing")
    st.markdown('''We believe that access to accurate and timely financial information
                should be democratized. That's why we've made our stock market prediction
                platform absolutely free to use. Individuals of all backgrounds, from 
                seasoned investors to novices just starting their financial journey, 
                can leverage the power of AI to make informed investment decisions.''')
    
    st.header("A Testament to Student Innovation")
    st.markdown('''This groundbreaking web application is a testament to the exceptional
                talent and innovation of Amrita students.''')
    
    st.header("Your Journey to Informed Investing Begins Here")
    st.markdown('''Take the first step towards a more informed and rewarding investment 
                experience. Start using our free stock market prediction platform today 
                and harness the power of AI to make smarter investment decisions.''')
    
    
if (selected == 'USAGE'):
    
    st.title("Usage")
    st.header("Harnessing AI as a Guiding Companion in Your Investment Journey")
    st.markdown('''While artificial intelligence (AI) has revolutionized various industries,
                it's crucial to recognize that it's not a panacea for investment decisions.
                Instead of relying solely on AI predictions, consider it as a valuable 
                companion to your comprehensive investment analysis.''')
    
    st.subheader("1. Diversify Your Information Sources:")
    st.markdown('''AI algorithms are trained on vast amounts of data, but they may overlook 
                nuances or subtle patterns that experienced investors can detect. Incorporate 
                traditional technical analysis, fundamental analysis, and market sentiment 
                indicators to gain a holistic perspective.''')
    
    st.subheader("2. Validate AI Predictions with Independent Research:")
    st.markdown('''AI predictions can provide valuable insights, but it's essential to validate 
                them with your own research. Assess the company's financial health, industry trends,
                and competitive landscape before making investment decisions.''')
    
    st.subheader("3. Employ AI as a Risk Management Tool:")
    st.markdown('''AI can identify potential risks and market volatility, enabling you to make 
                informed decisions to mitigate risks and protect your portfolio.''')
    
    st.subheader("4. Use AI to Enhance Your Trading Strategies:")
    st.markdown('''AI can help refine your trading strategies by identifying optimal entry and 
                exit points, analyzing historical trends, and recognizing patterns that may be 
                difficult for humans to detect.''')
    
    st.subheader("4. Use AI to Enhance Your Trading Strategies:")
    st.markdown('''AI algorithms are constantly evolving, so it's crucial to monitor their performance
                and adjust your reliance accordingly. Stay informed about advancements in AI and evaluate
                its effectiveness in your investment decisions.''')
    
    st.markdown('''Remember, AI is a powerful tool, but it's not a substitute for sound 
                investment judgment. Use AI as a valuable companion to your comprehensive 
                investment analysis, and always exercise due diligence before making 
                investment decisions.''')
    
    
if (selected == 'CONTACT & SUPPORT'):
    
    st.title("Contact & Support")
    st.subheader('''We understand that using our AI API can raise questions or require assistance.
                 Our dedicated support team is here to help you navigate the API's features, 
                 resolve any issues, and ensure a seamless user experience.''')
    
    st.markdown('''We strive to make our support resources easily accessible and transparent. 
                Here are the ways you can reach us:''')
    st.markdown('<a href="mailto:akshayramesh543@gmail.com">Mail to us ..</a>', unsafe_allow_html = True)
    st.subheader("Your valuble feedback helps us improve more....")
    st.markdown('<a href="mailto:akshayramesh543@gmail.com">Write to us ..</a>', unsafe_allow_html = True)
    
    