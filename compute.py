##
# imports
##
import datetime, urllib, re, math, json, requests,io
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize
import scipy.signal
from sklearn.linear_model import LinearRegression
from IPython.display import display, HTML
from ipywidgets import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# try:
from packaging import version
# except ImportError:
#     !pip install packaging

if( version.parse(pd.__version__) < version.parse("0.23.4")):
    print("update pandas")
    #pip install pandas --upgrade #--ignore-installed
    print(version.parse(pd.__version__))
##
# constants
##
dataTCS= {}
    
#** pandas
# change max number of rows to show
pd.set_option('display.max_rows', 300)

#** plotters formats

import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from cycler import cycler
#* dpi size
plt.rcParams['figure.dpi'] = 200
myFmt = mdates.DateFormatter('%m/%d')
myLocator = mticker.MultipleLocator(7)

default_cycler = (cycler(marker=['.','*','+','x','s']) *
                  cycler(color=['b','g','m','y','c','r']) *
                  cycler(linestyle=['--']) *
                  cycler(lw=[1]))

plt.rc('axes', prop_cycle=default_cycler)
import asyncio

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def cancel(self):
        self._task.cancel()

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
        return debounced
    return decorator


def getList(level,parent=""):
    if(level=="country"):
        idx    = pd.IndexSlice[:,:,:,:,"cases",level]
        selIdx= "country";
        #f=lambda i: pd.IndexSlice[:,:,:,i,     "cases",level]
    elif(level=="state"):
        #must have country with state
        if(parent!=""):
            idx    = pd.IndexSlice[:,:,:,parent,"cases",level]
        else:
            idx    = pd.IndexSlice[:,:,:,:,     "cases",level]
        selIdx= "state";
        #f=lambda i: pd.IndexSlice[:,:,i,:,     "cases",level]
    elif(level=="county"):
        #must have country with state
        if(parent!=""):
            idx    = pd.IndexSlice[:,:,parent,:,"cases",level]
        else:
            idx    = pd.IndexSlice[:,:,:,:,     "cases",level]            
        selIdx= "county";
        #f=lambda i: pd.IndexSlice[:,i,:,:,     "cases",level]
    g = dataTCS.loc[idx,:][dataTCS.loc[idx,"date"] == (datetime.date.today()-datetime.timedelta(days=2)) ]
    g.sort_values(by="value",inplace=True,ascending=False);      
    g.reset_index(inplace=True)
    return dict([ ["%s: %d"%(row[selIdx],row["value"]) , row[selIdx]] for idx, row in g.iterrows()])
    #return dict([ ["%s: %d"%(c[selIdx],dataTCS.loc[f(c[selIdx]),"value"].values[-1]),c[selIdx]] for c  in g.index.values])

#** panda data frame styles
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
    
    
def plotter(minCount=5000,
            minCountKpi="cases",
            equalize=False,
            equalizeTrg="Italy",
            equalizeCount=100, 
            plotKpi="deaths",
            level="country", 
            country="" ,
            dteOffset=-1
           ):
    output=[]
    idx = pd.IndexSlice
    
    if(level=="country"):
        g = dataTCS.loc[idx[:,:,:,equalizeTrg,minCountKpi,level],:]
    else:
        g = dataTCS.loc[idx[:,:,equalizeTrg,:,minCountKpi,level],:]    
    
    if(equalize):        
        if(np.max(g["value"]) < equalizeCount):
            print("reduce equalize count")
            return
        gdts = g[g["value"]>equalizeCount]["date"].values[0]
        gdte = g["date"].values[dteOffset]
    else:
        gdts = g["date"].values[0]
        gdte = g["date"].values[-1]
    N=(gdte-gdts).days
    
    if(level=="country"):
        idxScreen = lambda ctr :pd.IndexSlice[:,:,:,ctr,minCountKpi,level];
        idxPlot   = lambda ctr :pd.IndexSlice[:,:,:,ctr,plotKpi    ,level];
        idxer     = pd.IndexSlice            [:,:,:,:,  minCountKpi,level]
    else:
        #ctr becomes a loop variable for state
        idxScreen = lambda ctr :pd.IndexSlice[:,:,ctr,:,minCountKpi,level];
        idxPlot   = lambda ctr :pd.IndexSlice[:,:,ctr,:,plotKpi    ,level];
        idxer     = pd.IndexSlice            [:,:,:,:,  minCountKpi,level]
    
    for iii, row in dataTCS.loc[idxer,:][
        (dataTCS.loc[idxer,:]["date"]  == gdte     ) &
        (dataTCS.loc[idxer,:]["value"] >  minCount ) 
        #& (dataTCS.loc[idxer2,:]["value"] >5000 ) 
        ].iterrows():
        ctr = iii[3] if level=="country" else iii[2]
        g = dataTCS.loc[idxScreen(ctr),:]
        if(equalize):
            #check if sufficent samples exist to equlize
            if False == (g["value"][dteOffset]>equalizeCount).any():continue
                
            dts = g[g["value"]>equalizeCount]["date"].values[0]
            data= g[g["date"]>=dts]["value"]
            M   = min(N,sp.size(data))
            ts  = [(gdts + datetime.timedelta(i)) for i in range(0,sp.size(data))] 
        else:
            ts  = g["date"]
            data= g["value"]
            M   = sp.size(g)
               
        output.append([ts[0:M],data[0:M].values,ctr])    
    
    return [output,gdts,gdte]

def table_gen( country="United States", level="state", numRows=12):
    import seaborn as sns
    # gradient across two colols, second is one color
    cm = sns.diverging_palette(150,10, n=12,l=55,center='light',as_cmap=True)
    cm = sns.light_palette('red',as_cmap=True)


   
    dt1 = datetime.datetime.today().date()-datetime.timedelta(days=1)
    dt2 = datetime.datetime.today().date()-datetime.timedelta(days=2)

    midx = pd.MultiIndex.from_tuples([
        (dt1,"cases"),(dt1,"deaths"),(dt2,"cases"),(dt2,"deaths"),("delta","cases"),("delta","death"),("rate","cases"),("rate","death")
        ])

    data=list()
    ii=0
    for dt in [dt1, dt2]:    
        for dtype in  ["cases","deaths"]:        
            if(level=="country"):
                idx = pd.IndexSlice[:,:,:,:,      [dtype],level]
            elif(level=="state"):
                if(country=='.'):
                    idx = pd.IndexSlice[:,:,:,:,[dtype],level]
                else:
                    idx = pd.IndexSlice[:,:,:,country,[dtype],level]
            elif(level=="county"):
                #does not work!
                if(country=='.'):
                    idx = pd.IndexSlice[:,:,:,:,[dtype],level]
                else:
                    idx = pd.IndexSlice[:,:,:,country,[dtype],level]   
                    
            gg = dataTCS.loc[idx,:][dataTCS.loc[idx,"date"]==dt]["value"].to_frame().reset_index().set_index([level]).rename(
                columns={"value":'%d'%ii})
            data.append(gg)
            ii +=1

    dfFinal = pd.concat([data[0]['0'],data[1]['1'],
                         data[2]['2'],data[3]['3']], axis=1,sort=False)    
    dfFinal["casesD"] = dfFinal['0'] - dfFinal['2']
    dfFinal["deathD"] = dfFinal['1'] - dfFinal['3']
    dfFinal = dfFinal.replace(0,1e-4)
    dfFinal["rateC"]  = (dfFinal['0'] -dfFinal['2'] ) / dfFinal['2'] 
    dfFinal["rateD"]  = (dfFinal['1'] -dfFinal['3'] ) / dfFinal['3']
    dfFinal.columns=midx
    dfFinal
    display(dfFinal.sort_values(by=("delta","cases"),ascending=False).head(numRows)
        .style
        .background_gradient(cmap=cm, subset=['delta',"rate"])
        .format({('rate',"cases"): "{:.1%}",('rate',"death"): "{:.1%}"})
        )
    
    
# place a model on all countries with high counts
# cumulative model
def cumModel(t,A,B,C):
    return A/(np.cosh( B * (t-C))**2)        
# new case model
def newcModel(t,A,B,C):        
    return A/B * ( np.tanh( B * (t-C)) + 1 )
# logistic model
def logisModel(t,K,Q,B,v,M):
    # assume A is 0 (asymptote ), keep C as 1 for normal case
    #return A + (K-A) / ( (C + Q * np.exp(-B * (t-M)))**(1./v) )
    return K / ( (1+Q * np.exp(-B * (t-M)))**(1./v) )


    



def projector(
    stype = 'cases',
    level = "state",
    target= "California",
    minCount = 100,
    dteOffset=0,
    smoothLength=5,
    smoothOrder =2,
    logyPlot    =True
):
    """
    Project future growth
    type    - cases, deaths
    states  -
    """
    if(  level == "county"):  idx = pd.IndexSlice[:,    target,:,:, stype,level]
    elif(level == "state"):   idx = pd.IndexSlice[:,:,  target,:,   stype,level]
    elif(level == "country"): idx = pd.IndexSlice[:,:,:,target,     stype,level]
    if(dteOffset==0):
        y = dataTCS.loc[idx,"value"]
    else:
        y = dataTCS.loc[idx,"value"][0:dteOffset]
    if(sp.size(y)<5):
        print("not enough data")
        return
    y=y.astype("int32")
    ys  = sp.signal.savgol_filter(y, smoothLength, smoothOrder, mode='nearest')
    v=y.diff().values[1:]
    y=y[1:]
    A=max(v)
    C=np.argmax(v)

    # y smoothed

    ys=ys.astype("int32")
    vs=np.diff(ys)[1:]
    ys=ys[1:]


    t = np.arange(0,sp.size(y))
    f = lambda t,B: newcModel(t, A, B, C)
    paramsB, params_covariance = optimize.curve_fit(f, t, y, maxfev  = int(1e4))
    B=paramsB[0]

    # did not work well,lowered value of B
    f = lambda t,A,B: newcModel(t, A, B, C)
    paramsAB, params_covariance = optimize.curve_fit(f, t, y, p0=[A*1,.2],bounds=([A*0,-np.inf],[np.inf,np.inf]))
    Ae=paramsAB[0]
    Be=paramsAB[1]    

    # trying use function wiht more arguments, using original C
    fLogistic = lambda t,K,B,Q,v,M: logisModel(t,K=K,Q=Q,B=B,v=v,M=np.clip(M,C-21,C+21))
    params, params_covariance   = optimize.curve_fit(fLogistic,t,y,   maxfev  = int(1e4))  
    params2, params_covariance2 = optimize.curve_fit(fLogistic,t,ys,  maxfev  = int(1e4))
    print(params)

    fig = make_subplots(rows=2, cols=3, 
                    shared_xaxes=True, 
                    shared_yaxes=True, 
                    vertical_spacing=0.00,
                    horizontal_spacing = 0.0)
    
    
    t = np.arange(0,sp.size(y)+300)

    yh =newcModel(t,A,B,C)
    vh = cumModel(t,A,B,C)
    #plotty(ax00, ax10, y.values,v, yh,vh, label="tanh B=%.2f"%B,logyPlot=logyPlot)
    fig.add_trace(go.Scatter(y=yh,line = dict(color='green', width=1, dash='dash')), row=1, col=1,)
    fig.add_trace(go.Scatter(y=y ,line = dict(color='blue',  width=1, dash='dot' )), row=1, col=1,)
    fig.add_trace(go.Scatter(y=vh,line = dict(color='green', width=1, dash='dash')), row=2, col=1,)
    fig.add_trace(go.Scatter(y=v ,line = dict(color='blue',  width=1, dash='dot' )), row=2, col=1,)


    yh = fLogistic(t,params[0],params[1],params[2],params[3],params[4])    
    yh1=yh
    vh = np.diff(yh)
    vh1=vh
    fig.add_trace(go.Scatter(y=yh1,line = dict(color='green', width=1, dash='dash')), row=1, col=2)
    fig.add_trace(go.Scatter(y=ys ,line = dict(color='blue',  width=1, dash='dot' )), row=1, col=2)
    fig.add_trace(go.Scatter(y=vh1,line = dict(color='green', width=1, dash='dash')), row=2, col=2)
    fig.add_trace(go.Scatter(y=v  ,line = dict(color='blue',  width=1, dash='dot' )), row=2, col=2)

    Np = vh.argmax()    
    NtoP=Np-sp.size(v);

    #ax11.annotate("", xy=(sp.size(v), vh[-1]), xytext=(Np, vh[-1]),
    #             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    #ax11.text(sp.size(vs), int(vs[-1]*.6),"%d"%(NtoP),  {'color': 'red', 'fontsize': 10})
    
    
    yh = fLogistic(t,params2[0],params2[1],params2[2],params2[3],params2[4])    
    yh2=yh
    vh = np.diff(yh)      
    vh2=vh
    
    fig.add_trace(go.Scatter(y=yh2,line = dict(color='green', width=1, dash='dash')), row=1, col=3,)
    fig.add_trace(go.Scatter(y=ys ,line = dict(color='blue',  width=1, dash='dot' )), row=1, col=3,)
    fig.add_trace(go.Scatter(y=vh2,line = dict(color='green', width=1, dash='dash')), row=2, col=3,)
    fig.add_trace(go.Scatter(y=vs ,line = dict(color='blue',  width=1, dash='dot' )), row=2, col=3,)
    
    
    Np = vh.argmax()    
    NtoP=Np-sp.size(vs);

    #ax12.annotate("", xy=(sp.size(vs), vh[-1]), xytext=(Np, vh[-1]),
    #             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    #ax12.text(sp.size(vs), int(vs[-1]*.6),"%d"%(NtoP),  {'color': 'red', 'fontsize': 10})
    
    C=max(C, np.argmax(vh))
    try:
        M = C-min(np.argwhere(yh1>minCount)[0][0],np.argwhere(yh2>minCount)[0][0])
    except:
        print(np.argwhere(yh1>minCount))
        print(np.argwhere(yh2>minCount))
        return
    
    if(M <=1):
        print("min count to close to peak")
        return    
    
    if(logyPlot) :
        fig.update_layout( yaxis_type="log")
        fax_y = lambda x:np.log10(x)
    else:
        fig.update_layout( yaxis_type="linear") 
        fax_y = lambda x:x
    
    #ax00.set_ylim([1+minCount * .5,int(max(np.max(yh1),np.max(yh2))* 1.5)])        
    #ax10.set_ylim([1,np.max([np.max(vh1),np.max(vh2), np.max(v)])* 1.5])   
    

    fig.update_xaxes(range=[C-M,C+M],row=2,col=1)
    fig.update_xaxes(range=[C-M,C+M],row=2,col=2)
    fig.update_xaxes(range=[C-M,C+M],row=2,col=3)


    ss = dataTCS.loc[idx,"name"][0]#[i  for i in idx if i != '-1']
    print("{desc}: K={K:,} B={B:.2f} Q={Q:.2f} v={v:.2f} Ndays to peak=({N})".format(
        desc=' '.join(ss),
        K=int(params[0]),B=params[1],Q=params[2],v=params[3],N=NtoP
        ))       
    fig.show()
 


def getTimedGrowthRate(df,pidx,Nlength=3, debug=False,plot=False): 
    yo = df.loc[pidx,:].sum().values
    y  = sp.signal.savgol_filter(yo, 5, 2, mode='nearest')
    if(plot):
        plt.figure()
        plt.plot(yo,label="original")
        plt.plot(y, label="smoothed")
        plt.legend()
    
     # daily new case
    v = np.diff(y)#.values[1:]



#     # find the start date 
    idxs = np.argwhere(y>100)[2][0]    
    dts  = df.loc[pidx,:].columns[idxs]
    
#     #find the inflection point (max of v)
    idxi = sp.size(v)-1 #np.argmax(v)
    dti  = df.loc[pidx,:].columns[idxi]
    
    if(debug):print("%d %d %d" %(idxs, idxi,sp.size(y)))
    
    vGR = np.zeros(idxi-idxs-(Nlength-1))       
    i0= 0
    
    
    for i in range(idxs,idxi-(Nlength-1)):
        lv = sp.log10(y[i:i+Nlength])
        #lv = y[i:i+Nlength]
        if(debug):print(lv)
        m=(lv[-1]-lv[0])/Nlength
        if(debug):print("%d %.2f %.2f"%(i,m, np.exp(m)))
        vGR[i0]=m             
        i0+=1
    if(plot):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax2.plot(vGR,'b+')
        ax1.semilogy(y[idxs:idxi],'r')
        ax1.semilogy(v[idxs:idxi],'g')
    return vGR

# country ="Germany"
# pidx = pd.IndexSlice[country,"NA",:,:]
# getTimedGrowthRate(dataJHU,pidx,plot=True,debug=False,Nlength=5);
# dataJHU.loc[pidx].sum()
