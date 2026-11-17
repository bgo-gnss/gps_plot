#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
"""


def plotTime(sta, start=None, end=None, save=None, ylim=None, special=None,figDir="",events={}, fix=False,Dir=None,frfile=False,dfile=False):
    """
    plot multigas time series 
    """

    import os

    import matplotlib as mpl
    
    import timesmatplt as tplt
    import gpsUtils as utils
    from gtimes.timefunc import TimefromYearf, currYearfDate, currDatetime

    # The backend of matplotlib
    if save == "eps":
        mpl.use('ps')
    else:
        mpl.use('WebAgg')
    # Needs to be imported after the backend is defined
    import matplotlib.pyplot as plt

    if special and special != "test":
        end = currDatetime()

        if special == "90d":
            start = currDatetime(days=-91, refday=end)
        if special == "year":
            start = currDatetime(days=-366, refday=end)
        if special=="fixedstart":
            pass
        if special == "full":
            start = None 


    data = getmultigasdata(sta,frfile=frfile,start=start, end=end )
    
    firstpoint = data.index[0]
    lastpoint = data.index[-1]
    #this is where I decide to plot the unfiltered or filtered data
    rh_mask = mask_highrh(data)

    refTitle="Multigas"

    aquisitions = data.aquisition.unique()

    warnp = 0 # at which point we want to label the newest data green 
    # title string list [Title, Subtitle]
    Title = tplt.makelatexTitle( sta, lastpoint, ref=refTitle , warnp=warnp)

    

    # the extend of the plot
    if start and end:
        space = (end - start)/40
    elif start and not end:
        space = (lastpoint - start)/40
    elif not start and end:
        space = (end - firstpoint)/40
    else:
        space = (lastpoint - firstpoint)/40

    if start: # start of the plot
        start = start - space
    else: 
        start = firstpoint - space

    
    if end: # end of the plot
        end = end + space
    else: 
        end = lastpoint + space


    gaslabels=['SO$_2$ [ppm]','H$_2$S [ppm]','CO$_2$ [\%]']
    bcolors = ('Blue', 'Blue','Blue' )
    wcolors = ('red')
    ycolors = ('yellow','yellow','yellow')
    gcolors = ('0.75','0.75','0.75')
    grcolors = ('Green','Green','Green')
    gastypes=['so2','h2s','co2']
    warnings=[0.5,1.5,5]
    ylims = [(-1.0,2.0),(-1.0,5.0),(-0.01,8)]
    offsets = [0.0, 0.0, 0.05]

    fig = stdGaspltFrame(Ylabel=gaslabels, Title=Title)
    maxes = fig.axes

    ### ------


    for axs, bcolor, ycolor, gcolor, grcolor, wcolor, gastype, warning, ylim, offset in zip(maxes, bcolors, ycolors, gcolors, grcolors, wcolors, gastypes, warnings, ylims, offsets):

        axs.set_xlim([start,end])


        if len(ylim) == 1:
            ymin = gdata[gastype].min()
            ymax = gdata[gastype].max()
            axs.set_ylim([ymin-ylim[0],ymax+ylim[0]])

        if len(ylim) == 2:
            axs.set_ylim(ylim[0],ylim[1])

        #axs.get_yaxis().get_major_formatter().set_useOffset(False) 
        #print( axs.get_yaxis().get_major_formatter().get_useOffset() )

        # --- Plotting the data
        #data = data.loc[rh_mask]
        
        # plot all unprocessed data in grey
        aq_grouped = data.groupby('aquisition')[gastype]
        aq_grouped.apply(plot_aq,axs,gcolor)

        if gastype == "co2":
            shift = data[data['co2'] < 0.05].co2.describe().values[5]
            #print shift
        else:
            shift = aq_grouped.transform(rmoffset,offset)
            #print shift.describe()
        
        # plot shifted data in green
        data[gastype] = aq_grouped.apply(rmbaseline) - shift
        #print shift.describe()
        aq_grouped.apply(plot_aq,axs,grcolor)


        # plot shifted data with low humidity in blue THE BEST DATA
        aq_grouped = data.loc[rh_mask].groupby('aquisition')[gastype]
        aq_grouped.apply(plot_aq,axs,bcolor)

        axs.axhline(y=warning, linewidth=2, color = 'red')


    fig = addTimeofPlot(fig,dstr="%Y%m%d")
    fig = tsTickLabels(fig, period=(end-start))

    taxs = fig.axes[-1].twinx()
    taxs.set_xlim([start,end])
    taxs.patch.set_visible(False)
    taxs.set_ylim(0,100)

    for aq in aquisitions:
        pdata = data[data.aquisition == aq]
        x = pdata.index
        rh = pdata['rh']
        # plot humidity in orange
        taxs.plot(x,rh,linewidth=2, color = 'orange')

    taxs.axhline(y=93, linewidth=2, color = 'red')

    ## ---------------------------------------------
    if events:
        tplt.addEvent(events,fig)

    #tplt.inpLogo(fig)
 

    if save:
        filend="-%s-%s" % ("summit",refTitle.lower())

        if special:
            if special=="fixedstart":
                filend += "_since-%s" % ( firstpoint.strftime("%Y%m%d"), )  
            else:
                filend += "-%s" % special

        else:
           filend += "_%s-%s" % ( firstpoint.strftime("%Y%m%d") , lastpoint.strftime("%Y%m%d") )

        
        figFile = sta + filend
        fileName = os.path.join(figDir,figFile)


        #filename="{0:s}-{0:s}".format('hekla','multigas')rmbaseline
        tplt.saveFig(fileName,"eps",fig)
        utils.conv2png(fileName+"."+save,logo=None)
        #tplt.saveFig("hekla-multigas-unfiltered","eps",fig)
    else: 
        plt.show()


# functions on groups

def rmbaseline(y):
    """
    Remove baseline from data
    """
    import peakutils
    
    tmp = y - y.min()
    

    return tmp - peakutils.baseline(tmp.values,2) + y.min() 


def rmoffset(y,offset):
    """
    Remove offset
    """

    return y.min()


def plot_aq(pdata,axs,color):
    """
    plotting individual aquisitions
    """
    #x = pdata.index
    #y = pdata[gastype]

    axs.plot(pdata,linewidth=3,color = color)
    


def addTimeofPlot(fig,dstr="%Y%m%d"):
    """
    Adding vertical line signifiying now
    """

    from gtimes.timefunc import toDatetimel, currDatetime

    timeofPlot = currDatetime()
    print timeofPlot

    maxes = fig.axes
    for axs in maxes:
        axs.axvline(x=timeofPlot, color = 'black', linestyle="--",  zorder=2) 

    return fig


def mask_first(x):

    """
    """
    import numpy as np

    result = np.ones_like(x)
    result[0] = 0 
    
    return result

def init_mask(x,maskcond):
    """
    """
    import numpy as np

    if x[0] > maskcond:
        result = np.zeros_like(x)
    else:
        result = np.ones_like(x)

    return result


def mask_highrh(data,humit=93):
    """
    """

    aq_grouped = data.groupby('aquisition')
    ### this is checking if the first genuine sample (not sample 0) has humidity greater than 93, 
    ### throw out the whole aquisition even if it dries out during the aquisition
    rh_mask = aq_grouped.rh.transform(init_mask,humit).astype(bool)

    ### this is throwing out all other measurements, where humidity is greater than 93
    rh_mask[data.rh > humit] = False


    return rh_mask

#-------------------------

def filtermultigas(data):
    """
    """
    return data



def getmultigasdata(station, start=None, end=None, frfile=False, wrfile=True, fname=None):
    """
    """


    import requests
    import pandas as pd
    import datetime as dt
    strf="%Y-%m-%d %H:%M:%S"

    #pd.options.mode.chained_assignment = None

    if fname == None:
        fname = "{0:s}-multigas.dat".format(station)
    
    # ----------------- To get all the data needs to be fixed in the service -------------
    if start == None:
        start = dt.datetime.strptime("2012-12-01 00:00:00", "%Y-%m-%d %H:%M:%S") 
    if end == None:
         end = dt.datetime.now()
    
    dspan='?date_from={0:s}&date_to={1:s}'.format( start.strftime(strf), end.strftime(strf) )
    #print dspan
    # ------------------


    if frfile == True:
        data = pd.read_pickle(fname)

    else:
        url_rest = 'http://dev-api01.vedur.is:11223/aot/gas/'
        station_marker = station
    
        #request = requests.get(url_rest+station_marker+'?date_from=2012-12-01 00:00:00&date_to=2017-12-31 00:00:00')
        request = requests.get(url_rest+station_marker+dspan)
        data = pd.DataFrame.from_records(request.json(),index='timestamp')
        data.index = pd.to_datetime(data.index)
        
        if wrfile == True:
            data.to_pickle(fname)
    
    #aq_grouped = data.groupby(['aquisition'])
    #mask = aq_grouped['aquisition'].transform(mask_first).astype(bool)
    mask = data.groupby(['aquisition'])['aquisition'].transform(mask_first).astype(bool)
    data = data.loc[mask]
    data.loc[:,'co2'] *= 0.0001 # data['co2'].multiply(0.0001) converting ppm to %

    return data[start:end]

    
def stdGaspltFrame(Ylabel=None, Title=None):
    """
    Frame for plotting standard GPS time series. takes in Ylabel and Title
    and constructs empty figure with no data.
    input:
        Ylabel, 
        Title,

    output:
        fig, Figure object.

    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    #import matplotlib.image as image

    from gtimes.timefunc import currTime
    
    # constructing a figure with three axes and adding a title
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13,10))
    fig.subplots_adjust(hspace=0.15)


    if type(Title) is list:
        plt.suptitle(Title[0], y=0.965,x=0.51)
        axes[0].set_title(Title[1],y=1.01)
    elif type(Title) is str:
        axes[0].set_title(Title)
    else:
        pass


    if Ylabel == None: #
        Ylabel = ("North [mm]","East [mm]","Up [mm]")
        #Ylabel = ("Nordur [mm]","Austur [mm]","Upp [mm]")


    # --- apply custom stuff to the whole fig ---
    plt.minorticks_on
    plt.gcf().autofmt_xdate(rotation=0, ha='left')
    mpl.rcParams['legend.handlelength'] = 0    
    mpl.rcParams['text.usetex'] = True
    #mpl.rcParams['text.latex.unicode'] = True
    mpl.rc('text.latex', preamble='\usepackage{color}')
    #mpl.rc('text.latex', preamble='\usepackage[pdftex]{graphicx}')
    #mpl.rc('text.latex', preamble='\usepackage[icelandic]{babel}')
    #mpl.rc('text.latex', preamble='\usepackage[T1]{fontenc}')
    #mpl.rc('font', family='Times New Roman')
    
    for i in range(3):
        axes[i].set_ylabel(Ylabel[i], fontsize='x-large', labelpad=0)
        axes[i].axhline(0,color='black') # zero line


    for ax in axes[-1:-4:-1]:
        #tickmarks lables and other and other axis stuff on each axes 
        # needs improvement
    
        # --- X axis ---
        xax = ax.get_xaxis()
    
        xax.set_tick_params(which='minor', direction='inout', length=4, top='off')
        xax.set_tick_params(which='major', direction='inout', length=10, top='off')
        
        if ax is axes[0]: # special case of top axes
            xax.set_tick_params(which='major', direction='inout', length=10, top='on')
            xax.set_tick_params(which='minor', direction='inout', length=4, top='on')
        else:
            ax.spines['top'].set_visible(False)
            
    
    return fig



def spltMultigasFrame(Ylabel=None, Title=None):
    """
        Ylabel, 
        Title,

    output:
        fig, Figure object.

    """
    #import matplotlib.image as image

    from gtimes.timefunc import currTime
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    # constructing a figure with three axes and adding a title
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,7))

    
    if Title:
        ax.set_title(Title)
    else:
        pass

    if Ylabel == None: #
        Ylabels = ("Y1","Y2","Y3")


    # --- apply custom stuff to the whole fig ---
    plt.minorticks_on
    plt.gcf().autofmt_xdate(rotation=0, ha='left')
    mpl.rcParams['legend.handlelength'] = 0    
    mpl.rcParams['text.usetex'] = True
    mpl.rc('text.latex', preamble='\usepackage{color}')
    
    ax.set_ylabel(Ylabel, fontsize='x-large', labelpad=0)
    ax.axhline(0,color='black') # zero line


    #tickmarks lables and other and other axis stuff on each axes 
    # needs improvementimport matplotlib.pyplot as plt
    
    # --- X axis ---
    xax = ax.get_xaxis()
    
    xax.set_tick_params(which='minor', direction='inout', length=4, top='on')
    xax.set_tick_params(which='major', direction='inout', length=10, top='on')
        
            
    
    return fig

def tsSingTickLabels(fig,period=None):
    """
    """
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from datetime import timedelta
    
    import timesmatplt.timesmatplt as tplt
    #minorLoc, minorFmt, majorLoc,majorFmt = tsLabels(period)

    axes = fig.axes
    
    # major labels in separate layer
    axes[0].get_xaxis().set_tick_params(which='major', pad=15) 


    for ax in axes:
        #tickmarks lables and other and other axis stuff on each axes 
        # needs improvement
        ax.grid(True)
        ax.grid(False, which='major',axis='y')
        if period < timedelta(2600):
            ax.grid(True, which='minor',axis='x',)
            ax.grid(False, which='major',axis='x',)
    
        # --- X axis ---
        xax = ax.get_xaxis()

        xax = tplt.tslabels(xax,period=period,locator='minor',formater='minor')
        xax = tplt.tslabels(xax,period=period,locator='major',formater=None)
        xax = tplt.tslabels(xax,period=period,locator=None,formater='major')

        xax.set_tick_params(which='minor', direction='inout', length=4)
        xax.set_tick_params(which='major', direction='inout', length=15)

        for tick in xax.get_major_ticks():
            tick.label1.set_horizontalalignment('left')

    
        # --- Y axis ---
        yax = ax.get_yaxis()
        yax.set_minor_locator(mpl.ticker.AutoMinorLocator())
        yax.set_tick_params(which='minor', direction='in', length=4)
        yax.set_tick_params(which='major', direction='in', length=10)


    return fig

def tsTickLabels(fig,period=None):
    """
    """
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from datetime import timedelta
    
    import timesmatplt as tplt
    #minorLoc, minorFmt, majorLoc,majorFmt = tsLabels(period)

    axes = fig.axes
    
    # major labels in separate layer
    axes[-1].get_xaxis().set_tick_params(which='major', pad=15) 
    axes[0].get_xaxis().set_tick_params(which='major', labelbottom='off') 
    axes[1].get_xaxis().set_tick_params(which='major', labelbottom='off') 


    for ax in axes[-1:-4:-1]:
        #tickmarks lables and other and other axis stuff on each axes 
        # needs improvement
        ax.grid(True)
        #ax.grid(True,linestyle='solid',axis='x')
        if period < timedelta(2600):
            ax.grid(True, which='minor',axis='x',)
            ax.grid(False, which='major',axis='x',)
    
        # --- X axis ---
        xax = ax.get_xaxis()

        xax = tplt.tslabels(xax,period=period,locator='minor',formater='minor')

        xax = tplt.tslabels(xax,period=period,locator='major',formater=None)
        if ax is axes[2]:
            xax = tplt.tslabels(xax,period=period,locator=None,formater='major')

        xax.set_tick_params(which='minor', direction='inout', length=4)
        xax.set_tick_params(which='major', direction='inout', length=15)

        for tick in xax.get_major_ticks():
            tick.label1.set_horizontalalignment('left')

        #for label in xax.get_ticklabels('major')[::]:
            #label.set_visible(False)
            #label.set_text("test")
         #   print "text: %s" % label.get_text()
            #xax.label.set_horizontalalignment('center')
    
        # --- Y axis ---
        yax = ax.get_yaxis()
        yax.set_minor_locator(mpl.ticker.AutoMinorLocator())
        yax.set_tick_params(which='minor', direction='in', length=4)
        yax.set_tick_params(which='major', direction='in', length=10)


    return fig

def plotSingMultigas(data,aquisitions=None, gastypes=None, colors=None, warnings=None, ylims=None, start=None, end=None):
    """
    """
    import matplotlib.pyplot as plt


    fig = spltMultigasFrame()

    ax = fig.axes[0] 
    maxes = [ax, ax.twinx(), ax.twinx()]
    fig.subplots_adjust(right=0.75)

    maxes[0].set_xlim([start,end])
    fig = tsSingTickLabels(fig, period=(end-start))

    maxes[-1].spines['right'].set_position(('axes', 1.08))
    maxes[-1].set_frame_on(True)
    maxes[-1].patch.set_visible(False)

    for axs, bcolor, wcolor, gastype, warning, ylim in zip(maxes, bcolors, wcolors, gastypes, warnings, ylims): 
        if len(ylim) == 1:
            ymin = gdata[gastype].min() 
            ymax = gdata[gastype].max()
            axs.set_ylim([ymin-ylim[0],ymax+ylim[0]])
        elif len(ylim) ==2:
            axs.set_ylim([ylim[0],ylim[1]])


        for aq in aquisitions:
            pdata=data[data.aquisition == aq ][1:]
            x = pdata.index
            y = pdata[gastype]
            # --- Plotting the data
    
            axs.plot(x,y,linewidth=2,color = bcolor)
            axs.axhline(y=warning, linewidth=2, color = wcolor)
        
        axs.set_ylabel('%s' % gastype, color = 'black')
        axs.tick_params(axis='y', color = 'black')

    return fig


