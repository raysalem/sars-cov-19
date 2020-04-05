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

def getList(level,parent=""):
    if(level=="country"):
        idx    = pd.IndexSlice[:,:,:,:,"cases",level]
    elif(level=="state"):
        #must have country with state
        if(parent!=""):
            idx    = pd.IndexSlice[:,:,:,parent,"cases",level]
        else:
            idx    = pd.IndexSlice[:,:,:,:,     "cases",level]

    g = dataTCS.loc[idx,:][dataTCS.loc[idx,"date"] == datetime.date(2020,4,1) ]
    if(level=="country"):
        return np.sort([c[3] for c  in g.index.values])
    else:
        return np.sort([c[2] for c  in g.index.values])

#** panda data frame styles
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
    
    
def plotter(minCount=5000,minCountKpi="cases",equalize=True,equalizeTrg="ITA",equalizeCount=100, plotKpi="deaths",logyPlot=True,
           level="country", country="", reference=False, xlim=-1, ylim=-1):
    idx = pd.IndexSlice
        
    fig,ax = plt.subplots(1,1)
    if(equalize):
        if(level=="country"):
            g = dataTCS.loc[idx[:,:,:,equalizeTrg,minCountKpi,level],:]
        else:
            g = dataTCS.loc[idx[:,:,equalizeTrg,:,minCountKpi,level],:]
        if(max(g["value"]) < equalizeCount):
            print("reduce equalize count")
            return
        gdts = g[g["value"]>equalizeCount]["date"].values[0]
        gdte = g["date"].values[-1]
        N = (gdte-gdts).days#sp.size(g[g["value"]>equalizeCount]["date"].values)
        print("Equalize to %s : date:%s to %s" % (equalizeTrg,gdts,gdte))
    if(reference):
        ref =  [8., 13., 21., 34.]
        for rate in np.array(ref)/100.:            
            d = g[g["date"]>=gdts]["date"]
            #y0= g[g["date"]>=gdts]["value"].values[0]
            y0=equalizeCount
            M= sp.size(d.values)
            ax.plot(d,y0*np.exp(rate * np.arange(0,M)),'k--')#,label="ref%.1f"%(rate*100))
        print(ref)
    if(level=="country"):
        idxScreen = lambda ctr :idx[:,:,:,ctr,minCountKpi,level];
        idxPlot   = lambda ctr :idx[:,:,:,ctr,plotKpi    ,level];
        countries = dataTCS.index.levels[3]
    else:
        #ctr becomes a loop variable for state
        idxScreen = lambda ctr :idx[:,:,ctr,:,minCountKpi,level];
        idxPlot   = lambda ctr :idx[:,:,ctr,:,plotKpi    ,level];
        # using states here
        countries = np.sort(np.unique(dataTCS.loc[idx[:,:,:,country,"cases",level],:].index.values))[1:]
    
    # temp commented out    
    for ctr in countries:  
        try:
            if dataTCS.loc[idxScreen(ctr),"value"].values[-1]< minCount: continue     
        except:
            continue
        try:
            g = dataTCS.loc[idxPlot(ctr),["date","value"]]        
        except:
            continue
        if(equalize):
            #check if sufficent samples exist to equlize
            if False == (g["value"]>equalizeCount).any():continue
            dts = g[g["value"]>equalizeCount]["date"].values[0]
            print("%s --> %d" %(ctr,(dts-gdts).days))
            data= g[g["date"]>=dts]
            M = min(N,sp.size(data["value"]))
            ts = [(gdts + datetime.timedelta(i)) for i in range(0,sp.size(data["value"]))] 
        else:
            ts=g.index
            data=g["value"]
            M=sp.size(g)
        ax.plot(ts[0:M],data["value"][0:M].values,label=dataTCS.loc[idxPlot(ctr),:].name.values[0][0:5])    
    
    ax.xaxis.set_major_locator(myLocator)
    ax.xaxis.set_major_formatter(myFmt)
    ax.grid()
    mm = max(dataTCS.loc[idx[:,:,:,:,plotKpi],"value"])
    if(ylim==-1):
        ax.set_ylim(equalizeCount * .75, mm * 1.25)
    else:
        ax.set_ylim(equalizeCount * .75, ylim)
    if(xlim==-1):
        ax.set_xlim(gdts,gdte)
    else:
        ax.set_xlim(gdts,gdts+datetime.timedelta(xlim))
    ax.legend()
    fig.autofmt_xdate()    
    plt.title("%s vs time"%plotKpi);

    if(logyPlot) : plt.yscale('log')
        
# country screen
minCount     = 10000
minCountKpi  = "cases" # 'active', 'cases', 'deaths', 'growthFactor', 'recovered', 'tested'
# Equalize (kpi same as plot)
equalize     = True
equalizeTrg  = "ITA"
equalizeCount= 500

# plot 
plotKpi      = "cases"
logyPlot     = True
#plotter(minCount,minCountKpi,equalize,equalizeTrg,equalizeCount, plotKpi,logyPlot)         



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

def plotty(ax1, ax2, y,v,yh, vh, C,minCount,label, label2=''):
    M = np.argwhere(y>minCount)[0][0]
    ax1.semilogy(yh)
    ax1.semilogy(y)
    ax1.grid();
    #print(np.max(np.max(yh),np.max(y)))
    ax1.set_xlim([C-M,C+M])
    ax1.set_ylim([minCount * .75,int(max(np.max(yh),np.max(y))* 1.25)])
    ax1.set_title(label)
    
    ax2.plot(vh)    
    ax2.plot(v)    
    ax2.grid()    
    ax2.set_xlim([C-M,C+M])
    ax2.set_ylim([0,int(max(np.max(vh),np.max(v))* 1.25)])
    ax2.set_title(label2)



def projector(
    stype = 'cases',
    level = "state",
    target= "California",
    minCount = 0
):
    """
    Project future growth
    type    - cases, deaths
    states  -
    """
    if(  level == "county"):  idx = pd.IndexSlice[:,    target,:,:, stype,level]
    elif(level == "state"):   idx = pd.IndexSlice[:,:,  target,:,   stype,level]
    elif(level == "country"): idx = pd.IndexSlice[:,:,:,target,     stype,level]
    
    y = dataTCS.loc[idx,"value"]
    if(sp.size(y)<5):
        print("not enough data")
        return
    y=y.astype("int32")
    ys  = sp.signal.savgol_filter(y, 5, 2, mode='nearest')
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
    paramsB, params_covariance = optimize.curve_fit(f, t, y, p0=[.2])
    B=paramsB[0]

    # did not work well,lowered value of B
    f = lambda t,A,B: newcModel(t, A, B, C)
    paramsAB, params_covariance = optimize.curve_fit(f, t, y, p0=[A*1,.2],bounds=([A*0,-np.inf],[np.inf,np.inf]))
    Ae=paramsAB[0]
    Be=paramsAB[1]    
    #print("%14s - %f:%f | %5d:%5d "%(ctr,B,Be, A,Ae))

    # trying use function wiht more arguments
    fLogistic = lambda t,K,B,Q,v: logisModel(t,K=K,Q=Q,B=B,v=v,M=C)
    params, params_covariance = optimize.curve_fit(fLogistic,t,y,    bounds=([0,.02,0,0],[1e7,0.5,3,1]))  
    params2, params_covariance2 = optimize.curve_fit(fLogistic,t,ys, bounds=([0,.02,0,0],[1e7,0.5,3,1]))  

    #M = 18# 2.5 weeks
    #M = np.argwhere(yh>minCount)[0][0]
    fig  = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(3, 3)
    ax00 = fig.add_subplot(spec[0, 0])
    ax10 = fig.add_subplot(spec[1, 0])
    ax01 = fig.add_subplot(spec[0, 1])
    ax11 = fig.add_subplot(spec[1, 1])
    ax02 = fig.add_subplot(spec[0, 2])
    ax12 = fig.add_subplot(spec[1, 2])
    #ax2  = fig.add_subplot(spec[:,2])

    t = np.arange(0,sp.size(y)+60)

    yh =newcModel(t,A,B,C)
    vh = cumModel(t,A,B,C)
    plotty(ax00, ax10, y.values,v, yh,vh, C,minCount,"tanh","w/B=%.2f" % B)


    yh = fLogistic(t,params[0],params[1],params[2],params[3])    
    vh = np.diff(yh)
    plotty(ax01, ax11, y.values,v, yh,vh, C,minCount,"logistic")


    yh = fLogistic(t,params2[0],params2[1],params2[2],params2[3])    
    vh = np.diff(yh)
    plotty(ax02, ax12, ys,vs,yh,vh, C,minCount,"logistic In:Smooth")
    Np = vh.argmax()    
    NtoP=Np-sp.size(vs);
    ax11.annotate("", xy=(sp.size(vs), vs[-1]), xytext=(Np, vs[-1]),
                 arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    ax11.text(sp.size(vs), int(vs[-1]*.6),"%d"%(NtoP),  {'color': 'red', 'fontsize': 10})


    ss = dataTCS.loc[idx,"name"][0]#[i  for i in idx if i != '-1']
    print("{desc}: K={K:,} B={B:.2f} Q={Q:.2f} v={v:.2f} Ndays to peak=({N})".format(
        desc=' '.join(ss),
        K=int(params[0]),B=params[1],Q=params[2],v=params[3],N=NtoP
        ))       

 