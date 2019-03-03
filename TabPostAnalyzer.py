#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 
# 
# xiaohu yifan, 2017 
#
# Inputs:  () 
#           step 0 : plot original files
#           step 1 : TV denoising to define included cells and alcohol time
#           step 2 : with or without modified alcohol time, smooth data then save output
#                   
# Output:
#
#           None : all the important datas are saved into the processed folder

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse
import os
import seaborn as sns
import math
import sys
import csv
from tvregdiff import TVRegDiff

class Df_timegrid:
    """put df and time_grid as class value then saved to pickle"""

    def __init__(self, df, relative_time_grid):
        self.df = df;
        self.time_grid_relative=relative_time_grid;
        #print df
    
    def checkLength(self):
        if self.df is not None and self.time_grid_relative is not None:
            print "length of df is: ", len(self.df);
            print "length of relative_time_grid is: ",len(self.time_grid_relative);
            

def generate_ts_image(data_ts, data_tidy):
    sorted_ts = data_ts.loc[data_tidy.sort_values(by='mortality').index,:]
    image = np.zeros((len(sorted_ts),len(sorted_ts.columns),3), dtype = np.uint8)
    for i in range(len(sorted_ts)):
        image[i,:,:] = _PITrace2RGB(sorted_ts.iloc[i,:])
    return image

def generate_pooled_timeseries(filenames):
    dfs = dict()
    all_times = []
    time_index = dict()
    for fn in filenames:
        dfs[fn] = pd.read_csv(open(fn, 'Ur'),sep='\t',index_col=False)
        times = dfs[fn].columns[4:]
        all_times.extend(list(times))
        time_index[fn] = times
    time_grid = [parse(t) for t in np.sort(all_times)]
    relative_time_grid = np.array([float((t-time_grid[0]).total_seconds())/3600 for t in time_grid])
    #all_gaps = [relative_time_grid[i+1]-relative_time_grid[i] for i in range(len(relative_time_grid)-1)]

    from sklearn.cluster import KMeans
    kmc = KMeans(n_clusters = np.max([len(times) for times in time_index.values()])).fit( np.reshape(relative_time_grid,(len(relative_time_grid),1)) )
    centroids_times = np.sort(np.ravel(kmc.cluster_centers_))

    index_tuples = []
    for fn in dfs.keys():
        index_tuples.extend([(os.path.basename(fn),ind) for ind in dfs[fn].loc[:,'index']])

    centroids_df = pd.DataFrame(index = pd.MultiIndex.from_tuples(index_tuples, names=['fn', 'index']), columns=list(centroids_times)) 
    for fn in filenames:
        rtimes = np.array([ float((parse(t)-time_grid[0]).total_seconds())/3600 for t in time_index[fn]])
        data = dfs[fn].loc[:, time_index[fn]]
        for i in range(len(data)) :
            centroids_df.loc[(os.path.basename(fn),dfs[fn].loc[data.index[i],'index']),:] = np.interp(centroids_times, rtimes, data.iloc[i,:]).astype(float)

    return centroids_df
    
def generate_tidy_data(filenames, t_end):

    #tidy_c = ['fn','t1','index','x','y','mortality','status','PI_traint1','PI_trait2','PI_trait3']
    tidy_c = ['fn','t1','index','x','y','mortality','status','peak','valley','valley_t','peak_t','threshold_t-']
    df = pd.DataFrame(columns=tidy_c) 
    all_times = []

    for fn in filenames:

        dfi = pd.read_csv(open(fn, 'Ur'),sep='\t',index_col=False)
        times = dfi.columns[4:]
        #rtimes = np.array([float((parse(t)-parse(times[0])).total_seconds())/3600 for t in times])
        dfi.loc[:,'fn'] = os.path.basename(fn)
        dfi.loc[:,'t1'] = times[0]
        dfi.loc[:,'peak'] = dfi.loc[:,times].apply(lambda x:x.max(),axis=1)
        dfi.loc[:,'peak_t'] = dfi.loc[:,times].apply(lambda x:x.argmax(),axis=1)
        dfi.loc[:,'valley'] = dfi.loc[:,times].apply(lambda x:x[:x.argmax()].min(),axis=1)
        dfi.loc[:,'valley_t'] = dfi.loc[:,times].apply(lambda x:x[:x.argmax()].argmin(),axis=1)
        dfi.loc[:,'threshold_t-'] = dfi.apply(lambda x:times[times.get_loc(x['mortality'])-1],axis=1)
        df = df.append(dfi.loc[:,tidy_c])
        all_times.extend(list(times))

    time_grid = [parse(t) for t in np.sort(np.unique(all_times))]
    relative_time_grid = np.array([float((t-time_grid[0]).total_seconds())/3600 for t in time_grid])
    df.loc[:,'mortality'] = df.loc[:,'mortality'].map(lambda t: float((parse(t)-time_grid[0]).total_seconds())/3600)
    df.loc[:,'peak_t'] = df.loc[:,'peak_t'].map(lambda t: float((parse(t)-time_grid[0]).total_seconds())/3600)
    df.loc[:,'valley_t'] = df.loc[:,'valley_t'].map(lambda t: float((parse(t)-time_grid[0]).total_seconds())/3600)
    df.loc[:,'threshold_t-'] = df.loc[:,'threshold_t-'].map(lambda t: float((parse(t)-time_grid[0]).total_seconds())/3600)
    df.loc[:,'t1'] = df.loc[:,'t1'].map(lambda t: float((parse(t)-time_grid[0]).total_seconds())/3600)
    df['status'] = np.ones(len(df))
    df.loc[df['mortality']>t_end,'status'] = 2
    df['index'] = df['index'].astype(int)
    df = df.set_index(['fn','index'])

    return df, relative_time_grid, time_grid[0]
    
def _PITrace2RGB(trace):
    rgb_line = np.zeros((len(trace),3), dtype=np.uint8)
    profile = np.array(trace-200)
    profile[profile<=0] = 0.0000000000001
    rgb_line[:,0] = np.uint8((profile/np.float(profile.max()))*255)
    return rgb_line


class PIProfileAnalyzer:
    """post analyze .tab file
    """    
    
    MIN_PEAK_INCREASE = 3
    TVRegDiff_para = (200, 0.06)
    BACKGROUND_THRESHOLD = 330;
    TVRegDiff_paras = [#(200, 0.015),
                   #(200, 0.02),
                   (200, 0.03),
                   #(200, 0.04),(200, 0.045),(200, 0.05),
                   (200, 0.06),
                   #(200, 0.07),(200, 0.08),(200, 0.09),
                   (200, 0.12),(200, 0.18),
                   #(200, 0.2),(200, 0.24),
                    (200, 0.3)]
    
    chosen_para = (200, 0.06)
    MIN_PEAK_DECLINE = 3
    tidy_c = ['fn','t1','index','x','y','mortality','status','peak','valley','valley_t','peak_t','threshold_t-']
    background_lower_bnd = 200
    background_upper_bnd = 330
    ALCOHOL_DELTA_TOLERANCE = 2
    
    def __init__(self, pth, filterFile="", debug=False, redefineInCells=True):
        
        if len(os.path.split(pth)[1]) == 0:
            self.path = os.path.split(os.path.split(pth)[0])[0]
            self.saveName = os.path.split(os.path.split(pth)[0])[1]
        else:
            self.path = os.path.split(pth)[0]
            self.saveName = os.path.split(pth)[1]
        
        self.filenames = []
        self.dfs = dict()
        self.time_index = dict()
        self.all_times = []
        self.all_illums = dict()
        
        self.filterFile = filterFile;
        self.debug = debug
        self.redefineInCells = redefineInCells
        
        sns.set_context('paper')
        sns.set_style('white');
        
        self.getFilenames()
        
    def getFilenames(self):
    
        import glob
        existing_fns = glob.glob(os.path.join(self.path, self.saveName,'*.tab'))
        # get all the files in the path corresponding the filterFile.txt
        if self.filterFile != "":
            with open(self.filterFile, 'rU') as rf:
                for line_fn in rf:
                    line_fn = line_fn.rstrip()
                    if len(line_fn) > 0:
                        if os.path.join(self.path, line_fn) in existing_fns:
                            self.filenames.append(os.path.join(self.path, line_fn))
                            #print line_fn
        else:
            self.filenames = glob.glob(os.path.join(self.path, self.saveName,'*.tab'))
            
        print "prefix name of pickles and pngs will be : ", self.saveName
        
    def readFiles(self):
        
        import glob
        existing_fns = glob.glob(os.path.join(self.path, self.saveName,'*.tab'))
        
        count = 0;
        fcount = 0;
        for fn in self.filenames:

            df = pd.read_csv(open(fn, 'Ur'),sep='\t',index_col=False)
            if self.debug:
                print "run file :(",fcount,")", os.path.basename(fn)
            fcount += 1
            count += len(df)
            self.dfs[os.path.basename(fn)] = df
            self.time_index[os.path.basename(fn)] = [ parse(t) for t in self.dfs[os.path.basename(fn)].columns[4:]]
            self.all_times.extend(list(self.dfs[os.path.basename(fn)].columns[4:]))
            
        print "Total cell number : ", count
        print "filtered file/all file: ", fcount, '/', len(existing_fns)
   
    def getTimeGrid(self):
        self.time_grid = [parse(t) for t in np.sort(np.unique(self.all_times))]
        self.t0 = self.time_grid[0]
     
    def checkTimeJump(self):
        for (fn, tindex) in self.time_index.items():
            plt.plot(np.arange(len(tindex)),[(t-self.time_grid[0]).total_seconds()/3600 for t in tindex],label=fn)
            plt.legend(loc=4,fontsize=10)
        
    def viewOriginalPlots(self, height = -1):
        f, axes = plt.subplots(len(self.dfs.keys()), 1)
        for (i,fn) in zip(range(len(self.dfs.keys())),self.dfs.keys()):
            fl_ts = self.dfs[fn].iloc[:,4:]
            fl_image = generate_ts_image(fl_ts, self.dfs[fn])
            axes[i].imshow(fl_image, aspect=40/float(len(fl_ts)), interpolation='None')
            axes[i].grid(False)
            axes[i].set_title(fn, fontsize=12)
        
        if height < 0:
            f.set_size_inches(8,(len(self.dfs)-1)*0.3+0.008*np.sum([len(df) for df in self.dfs.values()]))
        else:
            f.set_size_inches(8, height)
        plt.tight_layout()
        f.savefig(os.path.join(self.path, self.saveName+' colomap_by_files.png'))
        return f, axes
    
    def viewCellTraces(self,fn,start,end,pn=5):
        if end >= len(self.dfs[fn]):
            end = len(self.dfs[fn])
        plt_n = (end-start+1)/pn + 2; 
        f, axes = plt.subplots(plt_n, 1)
        fl_ts = self.dfs[fn].iloc[:,4:]
        sorted_ts = fl_ts.loc[self.dfs[fn].sort_values(by='mortality').index,:]
        if end >= len(sorted_ts):
            end = len(sorted_ts)-2
        fl_image = generate_ts_image(fl_ts, self.dfs[fn])
        axes[0].imshow(fl_image,  aspect=20/float(len(fl_ts)), interpolation='None')
        for j in range(1,len(axes)):
            for i in range(pn*(j-1)+start,pn*(j-1)+start+pn if pn*(j-1)+start+pn <= end else end+1):
                axes[j].plot(np.array(sorted_ts.iloc[i,:]),label='cell '+str(i))
                axes[j].set_xlim([0,np.shape(fl_image)[1]])
                axes[j].legend(loc=2)

        f.set_size_inches(10,3*(1+len(axes)))
        plt.tight_layout()
        
        
    def denoise(self):
        self.all_filtered_Ls = dict()
        for fn in self.dfs.keys():
            fn_ts = np.array(self.dfs[fn].iloc[:,4:]).T.astype(float)
            fn_ts[fn_ts<1]=np.nan
            L=np.concatenate(([1],np.exp(np.cumsum(np.nanmean(np.diff(np.log(fn_ts),axis=0), axis=1)))))
            logLprime = TVRegDiff(np.log(L), *self.TVRegDiff_para, plotflag=0, diagflag=0)
            logLscale = np.sum(np.cumsum(logLprime[1:-1])*(np.log(L[1:])-np.log(L[0])))/np.sum(np.cumsum(logLprime[1:-1])*np.cumsum(logLprime[1:-1]))
            filtered_L = L[0]*np.concatenate(([1],np.exp(np.cumsum(logLprime[1:-1])*logLscale)))
            illum = L/filtered_L
            self.all_filtered_Ls [(fn,self.TVRegDiff_para)] = filtered_L
            self.all_illums [(fn,self.TVRegDiff_para)] = illum
            
    def tightPlot(self, every=10, prec=1):
        f, axes = plt.subplots(len(self.filenames)*2, 1)
        for (i,fn) in zip(range(len(self.filenames)),self.dfs.keys()):
            fl_ts = self.dfs[fn].iloc[:,4:]
            fl_image = generate_ts_image(fl_ts, self.dfs[fn])
            filtered_L = self.all_filtered_Ls [(fn,self.TVRegDiff_para)]
            illum = self.all_illums [(fn,self.TVRegDiff_para)]
            
            x = range(len(illum));
            t= self.time_index[fn];
            v = [ ('{:.'+str(prec)+'f}').format((l - self.t0).total_seconds()/3600) for l in t]
            labels = [];
            for index, val in enumerate(v):
                if index%every==0:
                    labels.append(val)
                else:
                    labels.append("");
            plt.sca(axes[i*2+1])
            plt.xticks(x, labels, color='red', fontsize=10)
            
            axes[i*2+1].plot(x,illum,label=str(self.TVRegDiff_para))
            axes[i*2].imshow(fl_image, aspect=40/float(len(fl_ts)), interpolation='None')
            axes[i*2].set_title(fn)
            axes[i*2+1].set_title(fn)
            axes[i*2+1].set_xlim([0,fl_image.shape[1]])
            #axes[i*2+1].set_ylim([0.8,1.2])
            axes[i*2+1].legend(loc=2)
            axes[i*2+1].plot([0,fl_image.shape[1]],[1,1],'k--',linewidth=0.5)
        f.set_size_inches(10,10*len(self.filenames))
        plt.tight_layout()
        
    def tightPlotFiles(self):

        plt.figure(100)
        for fn in self.dfs.keys():
            plt.plot([ (t-self.t0).total_seconds()/3600 for t in self.time_index[fn]],self.all_illums[(fn,self.TVRegDiff_para)],label=fn)
        plt.legend()
        plt.gcf().set_size_inches(8,6)
        plt.tight_layout()
        
    def getTidyData(self):
        #print "run getTidyData()"
        self.all_fl_ts = generate_pooled_timeseries(self.filenames)
        self.old_tidy, ts, self.t0 = generate_tidy_data(self.filenames, (self.time_grid[-1]-self.t0).total_seconds()/3600) 
    
    def denoisePlotAll(self,onlyPlot=False,asp=0):
        f, (ax1, ax2) = plt.subplots(2, 1)
        if not onlyPlot:
            all_ts = np.array(self.all_fl_ts).T.astype(float)
            all_ts[all_ts<self.background_lower_bnd]=self.background_lower_bnd
            L=np.concatenate(([1],np.exp(np.cumsum(np.nanmean( np.diff(np.log(all_ts),axis=0), axis=1)))))
            logLprimeTemp = TVRegDiff(np.log(L), *self.TVRegDiff_para, plotflag=0, diagflag=0)
                #print "Fun(denoisePlotAll) : logLprime ", logLprimeTemp
            logLscale = np.sum(np.cumsum(logLprimeTemp[1:-1])*(np.log(L[1:])-np.log(L[0])))/np.sum(np.cumsum(logLprimeTemp[1:-1])*np.cumsum(logLprimeTemp[1:-1]))
            filtered_L = L[0]*np.concatenate(([1],np.exp(np.cumsum(logLprimeTemp[1:-1])*logLscale)))
            self.illum = L/filtered_L
            self.all_fl_image = generate_ts_image(self.all_fl_ts, self.old_tidy)
        
        if asp == 0:
            asp = float(self.all_fl_image.shape[1])/float(self.all_fl_image.shape[0])/float(3)
        ax1.imshow(self.all_fl_image, aspect=asp, interpolation='None')
        ax2.plot(self.illum)
        ax1.set_xlim([0,len(self.illum)])
        ax2.set_xlim([0,len(self.illum)])
        f.set_size_inches(12,12)
        plt.tight_layout()
        self.figAll = f;
        self.ax1_profile = ax1;
        self.ax2_illumMean = ax2;
        
    
    def defineAlcoholTime(self,rankIndex = 0,aIndex = -1,more=10):
        """
            Alcohol time is determined as the time before max 'illumination' change, evaluated for every 3-frame windows 
        """
        
        illum_ts = pd.Series(data=self.illum, index=self.all_fl_ts.columns)
        self.rate_of_change = sorted( [ (
                                    (illum_ts.iloc[i+self.ALCOHOL_DELTA_TOLERANCE]-illum_ts.iloc[i])/
                                    (illum_ts.index[i+self.ALCOHOL_DELTA_TOLERANCE]-illum_ts.index[i])
                                   ,i) 
                                for i in range(len(illum_ts)-2) ], 
                                reverse=True, key=lambda r:abs(r[0]))
        
        self.rank = rankIndex;
        if aIndex < 0:
            self.alcoholIndex = self.rate_of_change[self.rank][1]
        else:
            self.alcoholIndex = aIndex;
            if aIndex >= len(self.illum):
                self.alcoholIndex = len(self.illum)-2;
        
        self.ax2_illumMean.plot((self.alcoholIndex,self.alcoholIndex),(np.min(self.illum),self.illum[self.alcoholIndex]),'r');
   
        """Other possible candidates are printed, also seen in the plot 2 cells above"""
        print "Other possible candidates are printed, also seen in the plot 2 cells above"
        for i in range(more):
            print "[",i,"] ",(self.rate_of_change[i][1],
                self.all_fl_ts.columns[self.rate_of_change[i][1]], 
                self.all_fl_ts.columns[self.rate_of_change[i][1]-1], 
                self.rate_of_change[i][0])
            
        
    
    def setAlcoholIndeces(self):
        from datetime import timedelta
        self.alcohol_t = self.t0+ timedelta(hours=self.all_fl_ts.columns[self.alcoholIndex-1])

        self.alcohol_indeces =  {fn:([i for (i,t) in zip(range(len(self.time_index[fn])),self.time_index[fn]) 
                             if t>self.alcohol_t ]+[len(self.time_index[fn])])[0] for fn in self.dfs.keys()}
    
    @staticmethod
    def traceHasZeroSeq(pi_trace, seq_n = 3):
        for i in range(len(pi_trace)-seq_n+1):
            if np.sum(pi_trace[i:i+seq_n]) == 0:
                return True
        return False

    @staticmethod
    def traceHasMortalityEvent(pi_trace, background_threshold=330, initial_peak = MIN_PEAK_INCREASE):
        return (pi_trace.max() > background_threshold) and (pi_trace.argmax()>initial_peak-1)  

    @staticmethod
    def traceFixableZeros(pi_trace, background_threshold=background_upper_bnd, seq_n=3):
        """
        Above, any cells with sequences of 0 are excluded. 
        Now, some of them might just be missing background signal. 
        Here we define any streches of 0 as fixable if their neighboring non-zero 
        values are background level.
        """
        left_boundaries = []
        right_boundaries = []
        for i in range(len(pi_trace)-seq_n):
            if pi_trace[i] < 1 and all(pi_trace[i+1:i+seq_n+1]>=1):
                right_boundaries.append((i,
                                         np.mean(pi_trace[i+1:i+seq_n+1]),
                                         all(pi_trace[i+1:i+seq_n+1]<background_threshold)
                                        ))
        for i in range(seq_n,len(pi_trace)):
            if pi_trace[i] < 1 and all(pi_trace[i-seq_n:i]>=1):
                left_boundaries.append((i,
                                        np.mean(pi_trace[i-seq_n:i]),
                                       all(pi_trace[i-seq_n:i]<background_threshold)
                                       ))
        zero_boundaries = []
        if abs(len(left_boundaries)-len(right_boundaries)) > 1:
            print pi_trace
            raise Exception('Too many left or right zero boundaries')
        if len(left_boundaries) - len(right_boundaries) == 1:
            if left_boundaries[-1][2] is True:
                zero_boundaries.append((left_boundaries[-1][0],len(pi_trace),left_boundaries[-1][1]))
            for i in range( len(right_boundaries) ):
                if left_boundaries[-1][2] and right_boundaries[-1][2]:
                    zero_boundaries.append((left_boundaries[i][0],
                                            right_boundaries[i][0]+1,
                                            0.5*(left_boundaries[i][1]+right_boundaries[i][1])
                                           ))
        if len(right_boundaries) - len(left_boundaries) == 1:
            if right_boundaries[-1][2] is True:
                zero_boundaries.append((0,right_boundaries[0][0]+1,right_boundaries[0][1]))
            for i in range( len(left_boundaries) ):
                if left_boundaries[-1][2] and right_boundaries[-1][2]:
                    zero_boundaries.append((left_boundaries[i][0],
                                            right_boundaries[1+i][0]+1,
                                            0.5*(left_boundaries[i][1]+right_boundaries[i+1][1])
                                           ))
        if len(right_boundaries) == len(left_boundaries):
            for i in range( len(right_boundaries) ):
                if left_boundaries[-1][2] and right_boundaries[-1][2]:
                    zero_boundaries.append((left_boundaries[i][0],
                                            right_boundaries[i][0]+1,
                                            0.5*(left_boundaries[i][1]+right_boundaries[i][1])
                                           ))
        return zero_boundaries
    
    def defineIncludedCells(self):
        """Next, we need to determine which cells to include"""
        self.cells_included = dict()
        for fn in self.dfs.keys():
            self.cells_included[fn] = [
                                    (not self.traceHasZeroSeq(self.dfs[fn].iloc[i,4:4+self.alcohol_indeces[fn]])) and 
                                    self.traceHasMortalityEvent(self.dfs[fn].iloc[i,4:4+self.alcohol_indeces[fn]]) 
                                  for i in range(len(self.dfs[fn])) ]
        for (fn, ar) in self.cells_included.items():
            print fn, np.sum(ar), len(ar)
    
    def redefineIncludedCells(self):
        """ Optional...
        If there too many cells exluded due to 0 values
        """
        '''Now fix the fixable 0s to the average of boundary values. Recalcuate the included cells'''
        count = 0
        for fn in self.dfs.keys():
            for i in range(len(self.dfs[fn])):
                if self.traceHasZeroSeq(self.dfs[fn].iloc[i,4:]):
                    zero_boundaries = self.traceFixableZeros(self.dfs[fn].iloc[i,4:])
                    if len(zero_boundaries) > 0:
                        count += 1
                        for (li,ri,val) in zero_boundaries:
                            ovals = np.array(self.dfs[fn].iloc[i,4+li:4+ri])
                            ovals[ovals<1] = val
                            self.dfs[fn].iloc[i,4+li:4+ri] = ovals

        print count
        self.cells_included = dict()
        for fn in self.dfs.keys():
            self.cells_included[fn] = [
                                    (not self.traceHasZeroSeq(self.dfs[fn].iloc[i,4:4+self.alcohol_indeces[fn]])) and 
                                    self.traceHasMortalityEvent(self.dfs[fn].iloc[i,4:4+self.alcohol_indeces[fn]]) 
                                  for i in range(len(self.dfs[fn])) ]
            
        for (fn, ar) in self.cells_included.items():
            print fn, np.sum(ar), len(ar)
            if np.sum(ar) == 0:
                print "delete file ", fn
                del self.cells_included[fn]
                del self.dfs[fn]
           
            
    def smoothData(self):
        corrected_filtered_Ls = dict()
        self.corrected_illums = dict()
        for TVRegDiff_para in self.TVRegDiff_paras:
            #print TVRegDiff_para
            for fn in self.dfs.keys():
                #print 'Working on '+fn
                fn_ts = np.array(self.dfs[fn].iloc[self.cells_included[fn],4:4+self.alcohol_indeces[fn]]).T.astype(float)
                fn_ts[fn_ts<1]=np.nan
                L=np.concatenate(([1],np.exp(np.cumsum(np.nanmean( np.diff(np.log(fn_ts),axis=0), axis=1)))))
                logLprime = TVRegDiff(np.log(L), *TVRegDiff_para, plotflag=0, diagflag=0)
                logLscale = np.sum(np.cumsum(logLprime[1:-1])*(np.log(L[1:])-np.log(L[0])))/np.sum(np.cumsum(logLprime[1:-1])*np.cumsum(logLprime[1:-1]))
                filtered_L = L[0]*np.concatenate(([1],np.exp(np.cumsum(logLprime[1:-1])*logLscale)))
                illum = L/filtered_L
                corrected_filtered_Ls [(fn,TVRegDiff_para)] = filtered_L
                self.corrected_illums [(fn,TVRegDiff_para)] = illum

   
    
    def determineLifespan(self):
        self.df = pd.DataFrame(columns=self.tidy_c) 
        self.timeseries = dict()
        for fn in self.dfs.keys():
            dfi = pd.DataFrame(index=self.dfs[fn].index[self.cells_included[fn]],columns=self.tidy_c)
            times = self.dfs[fn].columns[4:4+self.alcohol_indeces[fn]]
            dfi.loc[:,'fn'] = fn
            dfi.loc[:,'t1'] = times[0]
            dfi.loc[:,'cindex'] = self.dfs[fn].loc[self.cells_included[fn],'index']
            dfi.loc[:,'index'] = self.dfs[fn].index[self.cells_included[fn]]
            dfi.loc[:,['x','y']] = self.dfs[fn].loc[self.cells_included[fn],['x','y']]
            ts = self.dfs[fn].iloc[self.cells_included[fn],4:4+self.alcohol_indeces[fn]].astype(float,copy=True)
            ts.loc[:,:] = np.array(ts.loc[:,:])/np.array(self.corrected_illums[(fn,self.chosen_para)]).astype(float)
            for ind in ts.index:
                dfi.loc[ind,'peak'] = ts.loc[ ind, times[self.MIN_PEAK_INCREASE]: ].max()
                dfi.loc[ind,'peak_t'] = ts.loc[ ind, times[self.MIN_PEAK_INCREASE]: ].argmax()
                dfi.loc[ind,'valley'] = ts.loc[ ind, :dfi.loc[ind,'peak_t'] ].min()
                dfi.loc[ind,'valley_t'] = ts.loc[ ind, :dfi.loc[ind,'peak_t'] ].argmin()
                dfi.loc[ind,'status'] = (2,1)[np.array(ts.loc[ ind, times[self.MIN_PEAK_INCREASE]: ]).argmax()+self.MIN_PEAK_INCREASE<
                                              self.alcohol_indeces[fn]-self.MIN_PEAK_DECLINE]
                threshold = (dfi.loc[ind,'peak'] + dfi.loc[ind,'valley'])*0.5
                valley_ti = np.array(ts.loc[ ind, :dfi.loc[ind,'peak_t'] ]).argmin()
                if dfi.loc[ind,'valley_t'] != dfi.loc[ind,'peak_t']:
                    above_tis = np.arange(len(times))[np.array(ts.loc[ind,:]-threshold >= 0)]
                    dfi.loc[ind,'mortality'] = times[above_tis[above_tis>valley_ti][0]]
                    dfi.loc[ind,'threshold_t-'] = times[above_tis[above_tis>valley_ti][0]-1]
                else:                        
                    dfi.loc[ind,'mortality'] = times[0]
                    dfi.loc[ind,'status'] = 3
                    dfi.loc[ind,'threshold_t-'] = np.sort(self.all_times)[0]

            self.df = self.df.append(dfi.loc[:,self.tidy_c])
            self.timeseries[fn] = ts

        self.relative_time_grid = np.array([float((t-self.t0).total_seconds())/3600 for t in self.time_grid])
        self.df.loc[:,'mortality'] = self.df.loc[:,'mortality'].map(lambda t: float((parse(t)-self.t0).total_seconds())/3600)
        self.df.loc[:,'peak_t'] = self.df.loc[:,'peak_t'].map(lambda t: float((parse(t)-self.t0).total_seconds())/3600)
        self.df.loc[:,'valley_t'] = self.df.loc[:,'valley_t'].map(lambda t: float((parse(t)-self.t0).total_seconds())/3600)
        self.df.loc[:,'threshold_t-'] = self.df.loc[:,'threshold_t-'].map(lambda t: float((parse(t)-self.t0).total_seconds())/3600)
        self.df.loc[:,'t1'] = self.df.loc[:,'t1'].map(lambda t: float((parse(t)-self.t0).total_seconds())/3600)
        self.df['index'] = self.df['index'].astype(int)
        self.df['status'] = self.df['status'].astype(int)
        self.df = self.df.set_index(['fn','index'],drop=False)   
    
    
    def outputPlot(self):
        import re
        f, axes = plt.subplots(len(self.filenames), 1)
        for (i,fn) in zip(range(len(self.filenames)),sorted(self.dfs.keys())):
            fc_ts = self.timeseries[fn]
            fc_image = generate_ts_image(fc_ts, self.df.loc[fn])
            axes[i].imshow(fc_image, aspect=40/float(fc_ts.shape[0]), interpolation='None')
            axes[i].set_title(fn)
            axes[i].set_xlabel('Frame #')
            axes[i].set_ylabel('Cell #')
        f.set_size_inches(10,4.5*len(self.filenames))
        plt.tight_layout()
        f.savefig(os.path.join(self.path, self.saveName+' colomap_by_files.png'))

    def defineCentroidTimes(self):
        from sklearn.cluster import KMeans
        kmc = KMeans(n_clusters = np.max([len(times) for times in self.time_index.values()])).fit( np.reshape(self.relative_time_grid,(len(self.relative_time_grid),1)) )
        centroids_times = np.sort(np.ravel(kmc.cluster_centers_))
        centroids_times = centroids_times[centroids_times < (self.alcohol_t-self.t0).total_seconds()/3600]
        self.centroids_times = centroids_times;

        self.index_tuples = [ tpl for tpl in self.df.index.get_values()]
        centroids_df = pd.DataFrame(index = pd.MultiIndex.from_tuples(self.index_tuples, 
                                            names=['fn', 'index']), columns=list(self.centroids_times)) 
        for fn in self.timeseries.keys():
            tsdata = self.timeseries[fn]
            times = np.array([ float((parse(t)-self.t0).total_seconds())/3600 for t in tsdata.columns])
            for ind in tsdata.index:
                centroids_df.loc[(fn,ind),:] = np.interp(self.centroids_times, times, tsdata.loc[ind,:]).astype(float)
        self.centroids_df = centroids_df;
        
    def plotAllFiles(self,asp=0.02):
        sns.set_context('poster')
        sns.set_style('white')
        ts_full_image = generate_ts_image(self.centroids_df, self.df)
        plt.figure(310)
        plt.imshow(ts_full_image, aspect=asp, interpolation='None')
        plt.title(self.saveName)
        plt.xlabel('Frame #')
        plt.ylabel('Cell #')
        #plt.gcf().set_size_inches(10,30)
        plt.tight_layout()
        plt.gcf().savefig(os.path.join(self.path,self.saveName+' colomap_all_files.png'))
        
    
    def saveDataToPickles(self):
        import pickle
        #import Df_timegrid
        df_timegridrelative = Df_timegrid(self.df, self.relative_time_grid);
        f = open(os.path.join(self.path, self.saveName+' df_relative_timegrid.pickle'), 'wb')
        pickle.dump(df_timegridrelative, f)
        self.df.to_pickle(os.path.join(self.path, self.saveName+' mortality_dataframe.pickle'))
        self.centroids_df.to_pickle(os.path.join(self.path, self.saveName+' timeseries_dataframe.pickle'))
        analysis_metadata = {
            'time_grid' : self.time_grid,
            'time_indeces' : self.time_index,
            'background_threshold' : self.BACKGROUND_THRESHOLD,
            'cells_included': self.cells_included,
            'alcohol_t': self.alcohol_t,
            'alcohol_indeces': self.alcohol_indeces,
            'corrected_illums': self.corrected_illums,
            'all_illums': self.all_illums,
            'parameters': {'MIN_PEAK_INCREASE':self.MIN_PEAK_INCREASE, 
                           'MIN_PEAK_DECLINE': self.MIN_PEAK_DECLINE,
                           'TVRegDiff_para':self.chosen_para}
        }
        pickle.dump(analysis_metadata, open(os.path.join(self.path, self.saveName+' analysis_metadata.pickle'),'w'))
    
    def process(self,doPIcolormap=True, doReview=True,doOutput=True):
        if doPIcolormap:
            self.readFiles();
            self.getTimeGrid();
            if self.debug:
                sns.set_style('darkgrid');
                self.checkTimeJump();
                sns.set_style('white');
            self.viewOriginalPlots();
            print """
            Step 0: Before we process any data, preview the PI profiles in 
            each tab files. No need for any parameters, could always run directly.
            self.checkTimeJump() could be used to check time indeces of each files, 
            but is only triggered if self.debug is True.
            
            Use self.viewCellTraces(fileName,start,end,pn=5) to check cell traces
            """
            
        if doReview:
            self.denoise();
            self.tightPlot();
            if self.debug:
                self.tightPlotFiles();
            self.getTidyData();
            self.denoisePlotAll();
            self.defineAlcoholTime();
            print """
            Step1: Use TVReg to calculate illumination levels, which are used
            to estimate alcohol Time and then which cell to included in the end.
            This step requires a set of data visualisation and report, so that
            the user can verify the automatically chosen alcohol time is correct,
            and the correct cells are included.
            
            The first visualisations are the PI profiles for each file and 
            their illumination levels. The second visualisation is all illuminations
            plotted into the same figure. The third visualisation is the combined
            PI profile and illumination.
            
            We also report the number of included or excluded cells in each file.
            
            The user SHOULD make sure alcohol time is correct, or manually set
            it according to suggestions.
            """

        if doOutput:
            self.setAlcoholIndeces();    
            self.defineIncludedCells();
            if self.redefineInCells:
                self.redefineIncludedCells()
            print """
            Step2: After the user have reviewed and set the alcohol time,
            re-calculate cells to include. Then again use included data to
            re-estimate illumination, estimate lifespan, and output data
            in the form of pickles.
            """
            self.smoothData()
            self.determineLifespan()
            self.outputPlot()
            self.defineCentroidTimes()
            self.plotAllFiles(0.2)
            self.saveDataToPickles()


class TabPostAnalyzer(PIProfileAnalyzer):
    """post analyze .tab file
    """
    
    def __init__(self, pth, filterFile="", group=1, debug=False, redefineInCells=True):
        
        if len(os.path.split(pth)[1]) == 0:
            self.path = os.path.split(os.path.split(pth)[0])[0]
            self.saveName = os.path.split(os.path.split(pth)[0])[1]
        else:
            self.path = os.path.split(pth)[0]
            self.saveName = os.path.split(pth)[1]
        
        self.filenames = []
        self.dfs = dict()
        self.time_index = dict()
        self.all_times = []
        self.all_illums = dict()
        
        self.filterFile = filterFile;
        self.partial = group
        self.debug = debug
        self.redefineInCells = redefineInCells
        
        sns.set_context('poster')
        sns.set_style('white');
        
        self.getFilenames()
        
    def getFilenames(self):
    
        import glob
        existing_fns = glob.glob(os.path.join(self.path, self.saveName,'*.tab'))
        # get all the files in the path corresponding the filterFile.txt
        if self.filterFile != "":
            with open(self.filterFile, 'rU') as rf:
                previous = ''
                count = self.partial
                for line_fn in rf:
                    line_fn = line_fn.rstrip()
                    if len(line_fn) > 0:
                        if os.path.join(self.path, line_fn) in existing_fns and os.path.split(line_fn)[0]==self.saveName:
                            if previous == '':
                                count -= 1
                            if count == 0:
                                self.filenames.append(os.path.join(self.path, line_fn))
                            #print line_fn
                    previous = line_fn
        else:
            self.filenames = glob.glob(os.path.join(self.path, self.saveName,'*.tab'))
            
        print "prefix name of pickles and pngs will be : ", self.saveName+'-'+str(self.partial)
        
    def readFiles(self):
        
        import glob
        existing_fns = glob.glob(os.path.join(self.path, self.saveName,'*.tab'))
        
        count = 0;
        fcount = 0;
        for fn in self.filenames:

            df = pd.read_csv(open(fn, 'Ur'),sep='\t',index_col=False)
            if self.debug:
                print "run file :(",fcount,")", os.path.basename(fn)
            fcount += 1
            count += len(df)
            self.dfs[os.path.basename(fn)] = df
            self.time_index[os.path.basename(fn)] = [ parse(t) for t in self.dfs[os.path.basename(fn)].columns[4:]]
            self.all_times.extend(list(self.dfs[os.path.basename(fn)].columns[4:]))
            
        print "Total cell number : ", count
        print "filtered file/all file: ", fcount, '/', len(existing_fns)
        
        self.saveName = self.saveName+'-'+str(self.partial)
        
    def viewOriginalPlots(self, height = -1):
        f, axes = plt.subplots(len(self.dfs.keys()), 1)
        for (i,fn) in zip(range(len(self.dfs.keys())),self.dfs.keys()):
            fl_ts = self.dfs[fn].iloc[:,4:]
            fl_image = generate_ts_image(fl_ts, self.dfs[fn])
            axes[i].imshow(fl_image, aspect=40/float(len(fl_ts)), interpolation='None')
            axes[i].grid(False)
            axes[i].set_title(fn, fontsize=12)
        
        if height < 0:
            f.set_size_inches(12,len(self.dfs)*0.3+0.02*np.sum([len(df) for df in self.dfs.values()]))
        else:
            f.set_size_inches(12, height)
        plt.tight_layout()
        f.savefig(os.path.join(self.path, self.saveName+'_colomap_by_files.png'))
        return f, axes
        
    
    def outputPlot(self):
        import re
        f, axes = plt.subplots(len(self.filenames), 1)
        for (i,fn) in zip(range(len(self.filenames)),sorted(self.dfs.keys())):
            fc_ts = self.timeseries[fn]
            fc_image = generate_ts_image(fc_ts, self.df.loc[fn])
            axes[i].imshow(fc_image, aspect=40/float(fc_ts.shape[0]), interpolation='None')
            axes[i].set_title(fn)
            axes[i].set_xlabel('Frame #')
            axes[i].set_ylabel('Cell #')
        f.set_size_inches(10,4.5*len(self.filenames))
        plt.tight_layout()
        f.savefig(os.path.join(self.path, self.saveName+'_colomap_by_files.png'))
        
    def plotAllFiles(self,asp=0.02):
        sns.set_context('poster')
        sns.set_style('white')
        ts_full_image = generate_ts_image(self.centroids_df, self.df)
        plt.figure(310)
        plt.imshow(ts_full_image, aspect=asp, interpolation='None')
        plt.title(self.saveName)
        plt.xlabel('Frame #')
        plt.ylabel('Cell #')
        #plt.gcf().set_size_inches(10,30)
        plt.tight_layout()
        plt.gcf().savefig(os.path.join(self.path,self.saveName+'_colomap_all_files.png'))
        
    
    def saveDataToPickles(self):
        import pickle
        #import Df_timegrid
        df_timegridrelative = Df_timegrid(self.df, self.relative_time_grid);
        f = open(os.path.join(self.path, self.saveName+'_df_relative_timegrid.pickle'), 'wb')
        pickle.dump(df_timegridrelative, f)
        self.df.to_pickle(os.path.join(self.path, self.saveName+'_mortality_dataframe.pickle'))
        self.centroids_df.to_pickle(os.path.join(self.path, self.saveName+'_timeseries_dataframe.pickle'))
        analysis_metadata = {
            'time_grid' : self.time_grid,
            'time_indeces' : self.time_index,
            'background_threshold' : self.BACKGROUND_THRESHOLD,
            'cells_included': self.cells_included,
            'alcohol_t': self.alcohol_t,
            'alcohol_indeces': self.alcohol_indeces,
            'corrected_illums': self.corrected_illums,
            'all_illums': self.all_illums,
            'parameters': {'MIN_PEAK_INCREASE':self.MIN_PEAK_INCREASE, 
                           'MIN_PEAK_DECLINE': self.MIN_PEAK_DECLINE,
                           'TVRegDiff_para':self.chosen_para}
        }
        pickle.dump(analysis_metadata, open(os.path.join(self.path, self.saveName+'_analysis_metadata.pickle'),'w'))
