import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse
import os
import rpy2
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
pth = os.path.realpath(__file__)
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects import pandas2ri, Formula, globalenv
#import pandas.rpy.common as com
pandas2ri.activate()
rsurvival = rpacks.importr('survival')
rbshazard = rpacks.importr('bshazard')
rflexsurv = rpacks.importr('flexsurv')

from scipy.stats import norm, chi2
from scipy.optimize import newton

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

def _PITrace2RGB(trace):
    
    rgb_line = np.zeros((len(trace),3), dtype=np.uint8)
    
    profile = np.array(trace-200)
    profile[profile<0] = 0
    rgb_line[:,0] = np.uint8((profile/np.float(profile.max()))*255)
    
    return rgb_line
        
def generate_ts_image(data_ts, data_tidy):
    
    sorted_ts = data_ts.loc[data_tidy.sort_values(by='mortality').index,:]

    image = np.zeros((len(sorted_ts),len(sorted_ts.columns),3), dtype = np.uint8)
    for i in range(len(sorted_ts)):
        image[i,:,:] = _PITrace2RGB(sorted_ts.iloc[i,:])
        
    return image
    
# def locationplot(df, **kwds):
    
def KaplanMeier(data, relative_time_grid):
    
    mortalities = pd.Series(data.loc[data['status']==1,'mortality'].value_counts(), index=relative_time_grid).fillna(0)
    censored = pd.Series(data.loc[data['status']==2,'mortality'].value_counts(), index=relative_time_grid).fillna(0)
    counts = mortalities + censored
    n = len(counts)
    # At Risk individuals at t_i are those who are at risk just before t_i, thus including the mortality & censoring events at t_i
    atRisk = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=0), counts), index=relative_time_grid)
    # Individuals that are at risk just after t_i, thus not include the mortality & censoring events at t_i
    atRiskNext = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=1), counts), index=relative_time_grid)
    # Survived individuals are those who did not die just after t_i, thus including censored individuals at t_i and atRiskNext individuals
    survived = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=1), counts) + np.array(censored), index=relative_time_grid)
    
    censorRatio = (survived / atRiskNext).loc[atRiskNext!=0]
    censorAdjust = pd.Series(data=np.exp(np.dot(np.tril(np.ones((len(censorRatio),len(censorRatio))),k=0) ,np.log(censorRatio))),index=censorRatio.index)
    ## Aalen O. et al (2008) formula 3.26, 3.32 & my own formula
    KaplanMeier = survived.loc[censorAdjust.index[1:]] / float(sum(counts)) * censorAdjust.values[:-1]
    ## Aalen O. et al (2008) formula 3.28, 3.14, 3.15
    def deltaNAV (d, Y):
        return np.sum([ np.power(Y-l,-2) for l in range(int(d))])
    KM_var = np.power(KaplanMeier, 2) * np.dot(np.tril(np.ones((len(KaplanMeier),len(KaplanMeier))),k=0), np.array([deltaNAV(mortalities.loc[ind],atRisk.loc[ind]) for ind in KaplanMeier.index]))
    ## Aalen O. et al (2008) formula 3.29, could also be 3.30
    KM_lower = KaplanMeier + norm.ppf(ALPHA/2)*np.sqrt(KM_var)
    KM_upper = KaplanMeier + norm.ppf(1-ALPHA/2)*np.sqrt(KM_var)
    ## Aalen O. et al (2008) formula 3.30
    KM_lower = np.power(KaplanMeier, np.exp( norm.ppf(ALPHA/2)*np.sqrt(KM_var)/(KaplanMeier*np.log(KaplanMeier)) ) )
    KM_upper = np.power(KaplanMeier, np.exp( norm.ppf(1-ALPHA/2)*np.sqrt(KM_var)/(KaplanMeier*np.log(KaplanMeier)) ) )
    
    return pd.DataFrame(data={"survivorship":KaplanMeier,"upper_ci":KM_upper,"lower_ci":KM_lower,"variance":KM_var }, index=KaplanMeier.index)
    

#INFINITESIMAL_COUNT = 0.001
ALPHA = 0.05

def NelsonAalen(data, relative_time_grid):

    mortalities = pd.Series(data.loc[data['status']==1,'mortality'].value_counts(), index=relative_time_grid).fillna(0)
    censored = pd.Series(data.loc[data['status']==2,'mortality'].value_counts(), index=relative_time_grid).fillna(0)
    counts = mortalities + censored
    n = len(counts)
    # At Risk individuals at t_i are those who are at risk just before t_i, thus including the mortality & censoring events at t_i
    atRisk = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=0), counts), index=relative_time_grid)
    # Individuals that are at risk just after t_i, thus not include the mortality & censoring events at t_i
    # atRiskNext = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=1), counts), index=relative_time_grid)
    # Survived individuals are those who did not die just after t_i, thus including censored individuals at t_i and atRiskNext individuals
    #survived = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=1), counts) + np.array(censored), index=relative_time_grid)
    
    ## atRisk[atRisk==0]=INFINITESIMAL_COUNT
    ## Aalen O. et al (2008) formula 3.4 & 3.13, should be 3.12
    ## NelsonAalen = np.dot(np.tril(np.ones((n,n)),k=0),  mortalities/atRisk)
    ## Aalen O. et al (2008) formula 3.12
    def deltaNA (d, Y):
        return np.sum([ np.power(Y-l,-1) for l in range(int(d))])
    NelsonAalen = np.dot(np.tril(np.ones((n,n)),k=0),  [deltaNA(mortalities.loc[ind],atRisk.loc[ind]) for ind in atRisk.index] )
    ## Aalen O. et al (2008) formula 3.5 & 3.16, should be 3.15
    ##NA_var = np.dot( np.tril(np.ones((n,n)),k=0), (mortalities*(atRisk-mortalities))/(atRisk*atRisk*atRisk) )
    ## Aalen O. et al (2008) formula 3.5 & 3.16, should be 3.15
    def deltaNAV (d, Y):
        return np.sum([ np.power(Y-l,-2) for l in range(int(d))])
    NA_var = np.dot(np.tril(np.ones((n,n)),k=0),  [deltaNAV(mortalities.loc[ind],atRisk.loc[ind]) for ind in atRisk.index] )
    # Aalen O. et al (2008) formula 3.7
    NA_lower = NelsonAalen * np.exp( norm.ppf(ALPHA/2)*np.sqrt(NA_var)/NelsonAalen )
    NA_upper = NelsonAalen * np.exp( norm.ppf(1-ALPHA/2)*np.sqrt(NA_var)/NelsonAalen )
    ## Aalen O. et al (2008) formula 3.6
    ## NA_lower = NelsonAalen + norm.ppf(ALPHA/2)*np.sqrt(NA_var)
    ## NA_upper = NelsonAalen + norm.ppf(1-ALPHA/2)*np.sqrt(NA_var)
    
    return pd.DataFrame(data={"cumulative_hazard":NelsonAalen,"upper_ci":NA_upper,"lower_ci":NA_lower,"variance":NA_var }, index=relative_time_grid)
    
def BinningHazardNAEstimates(data, relative_time_grid):
    
    from sklearn.cluster import KMeans
    kmc = KMeans(n_clusters = int(np.floor(np.max(relative_time_grid)))).fit( np.reshape(relative_time_grid,(len(relative_time_grid),1)) )
    gap_mids = [ ((relative_time_grid[i]+relative_time_grid[i+1])/2,
                  relative_time_grid[i],relative_time_grid[i+1],      
                  i ) 
                for i in range(len(relative_time_grid)-1) if kmc.labels_[i] != kmc.labels_[i+1] ]

    mortalities = pd.Series(data.loc[data['status']==1,'mortality'].value_counts(), index=relative_time_grid).fillna(0)
    censored = pd.Series(data.loc[data['status']==2,'mortality'].value_counts(), index=relative_time_grid).fillna(0)
    counts = mortalities + censored
    n = len(counts)
    # At Risk individuals at t_i are those who are at risk just before t_i, thus including the mortality & censoring events at t_i
    atRisk = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=0), counts), index=relative_time_grid)
    
    NA = NelsonAalen(data, relative_time_grid)
    
    NA_limit = 0
    points = []
    for i in range(1,len(gap_mids)):
        (t,tm,tp,ind) = gap_mids[i]
        if NA.loc[tm,'lower_ci']>NA_limit:
            NA_limit = NA.loc[tm,'upper_ci']
            points.append(
            ( t,ind,tm,tp,
                 NA.loc[tm,'cumulative_hazard'],
                 NA.loc[tm,'lower_ci'],
                 NA.loc[tm,'upper_ci'],
                )
            )
       
    from statsmodels.stats.proportion import proportion_confint
    txy = []
    for i in range(1,len(points)):
        current_point = points[i]
        previous_point = points[i-1]
        d = np.sum(mortalities.loc[previous_point[3]:current_point[2]])
        Y = atRisk.loc[previous_point[3]]
        txy.append(
                (previous_point[2],current_point[3],current_point[0]-previous_point[0],
                 0.5*(previous_point[4]+current_point[4]),
                 0.5*(previous_point[5]+current_point[5]),
                 0.5*(previous_point[6]+current_point[6]),
                 (current_point[4]-previous_point[4])/(current_point[0]-previous_point[0]),
                 Y,d,
                 proportion_confint(d,Y,method='jeffrey',alpha=0.05)[0],
                 proportion_confint(d,Y,method='jeffrey',alpha=0.05)[1]
                )
                ) 
        
    pltdata = pd.DataFrame(data=txy,columns = ['tL','tR','dt','NA','lciNA','uciNA','dNA','Y','d','lcid','ucid'])
    pltdata['hr_eb_l'] = (pltdata['d']/pltdata['Y'] - pltdata['lcid'])/pltdata['dt']
    pltdata['hr_eb_u'] = (pltdata['ucid'] - pltdata['d']/pltdata['Y'])/pltdata['dt']
    pltdata['na_eb_l'] = [pltdata.loc[i,'NA']-pltdata.loc[i,'lciNA'] for i in pltdata.index]
    pltdata['na_eb_r'] = [pltdata.loc[i,'uciNA']-pltdata.loc[i,'NA'] for i in pltdata.index]
    
    return pltdata
    

def Breslow(data, relative_time_grid):
    
    NsAal = NelsonAalen(data, relative_time_grid)

    return pd.DataFrame(data = {"survivorship":np.exp(-NsAal["cumulative_hazard"]),
                                "upper_ci":np.exp(-NsAal["lower_ci"]),
                                "lower_ci":np.exp(-NsAal["upper_ci"])}, index=NsAal.index)
    
def BSHazardR(data):
    
    rdf = pandas2ri.py2ri(data[['mortality','status']].applymap(float))
    rbsfit = rbshazard.bshazard('Surv(mortality,status==1)~1',data=rdf)
    bsfit = { rbsfit.names[i]:pandas2ri.ri2py(rbsfit[i]) for i in range(len(rbsfit)) if rbsfit[i] is not rpy2.rinterface.NULL }
    
    return bsfit
    
from collections import namedtuple
KSmResidues = namedtuple('KSmResidues', ('KM_residues', 'maximum_cdf'))
KStestResult = namedtuple('KstestResult', ('statistic', 'pvalue','critical_stat'))

def KSm_gof(data, relative_time_grid, sf, args=(), alternative='two-sided'):
    """
    Calculate the modified one-sided Kolmogorov-Smirnov statistic, adjusted for
    right-censored time-to-death data, as described in Fleming et al (1980).

    This calcuate the residues for a non-parametric test of the distribution 
    G(x) of an observed time-to-death random variable against a given 
    distribution F(x). The actual test is performed in the function KSm_test.
    Under the null hypothesis the two distributions are identical, G(x)=F(x).
    This function returns the residues whose extreme values are used to perform 
    the KS test. The KS test is only valid for continuous distributions.
    
    Reference
    ----------
    Thomas R. Fleming, Judith R. O'Fallon, Peter C. O'Brien and David P. 
    Harrington. Biometrics Vol. 36, No. 4 (Dec., 1980), pp. 607-625

    Parameters
    ----------
    data : pandas DataFrame
        Contains the time-to-death observations. Should contain at columns
        named 'mortality' indicating time to event, and named 'status' 
        indicating event type: death (status==1) or censored (status==2)
    relative_time_grid : array
        All observation timepoints. Should be a superset of data['mortality']
    sf : callable
        The survival function of the null distribution. sf = 1 - cdf
    args : tuple, sequence, optional
        Distribution parameters.

    Returns
    -------
    KM_residues : pandas Series
        KS residues, whose extremes are the KS statistic
    maximum_cdf :  float
        Estimated c.d.f. at the maximal observation time. between (0,1), needed
        for calculating the asymptotic distribution of the KS statistic.
    """
    
    def deltaNA (d, Y):
        return np.sum([ np.power(Y-l,-1) for l in range(int(d))])
    
    mortalities = pd.Series(data.loc[data['status']==1,'mortality'].value_counts(), index=relative_time_grid).fillna(0)
    censored = pd.Series(data.loc[data['status']==2,'mortality'].value_counts(), index=relative_time_grid).fillna(0)
    counts = mortalities + censored
    n = len(counts)
    atRisk = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=0), counts), index=relative_time_grid)
    survived = pd.Series(data=np.dot(np.triu(np.ones((n,n)),k=1), counts) + np.array(censored), index=relative_time_grid)
    
    sfvals = sf(relative_time_grid, *args)
    
    def deltaNA (d, Y):
        return np.sum([ np.power(Y-l,-1) for l in range(int(d))])
    beta_NelsonAalen = np.dot(np.tril(np.ones((n,n)),k=0),  [deltaNA(mortalities.loc[ind],atRisk.loc[ind]) for ind in atRisk.index] )
    alpha_CumCensor = np.dot(np.tril(np.ones((n,n)),k=0),  [deltaNA(censored.loc[ind],survived.loc[ind]) for ind in survived.index] )
    A_CumHazardCenAdjusted = np.dot(np.tril(np.ones((n,n)),k=0),
                              [0]+[np.exp(-0.5*alpha_CumCensor[i-1])*np.log(sfvals[i-1]/sfvals[i]) for i in range(1,len(atRisk))] )
    B_NelsonAalenCenAdjusted = np.dot(np.tril(np.ones((n,n)),k=0),
                              [0]+[np.exp(-0.5*alpha_CumCensor[i-1])*deltaNA(mortalities.iloc[i],atRisk.iloc[i]) for i in range(1,len(atRisk))] )
    Yminus = np.sqrt(n) * 0.5 * (sfvals[1:]+np.exp(-beta_NelsonAalen[:-1])) * (A_CumHazardCenAdjusted[1:] - B_NelsonAalenCenAdjusted[:-1])
    Yplus = np.sqrt(n) * 0.5 * (sfvals+np.exp(-beta_NelsonAalen)) * (A_CumHazardCenAdjusted - B_NelsonAalenCenAdjusted)
    Yconcat = pd.concat([pd.Series(data=Yplus,index=atRisk.index), pd.Series(data=Yminus,index=atRisk.index[1:])]).sort_index()
    R_MaxCDF = 1 - 0.5*(np.exp(-beta_NelsonAalen[-1])+sfvals[-1])
    
    return KSmResidues(Yconcat, R_MaxCDF)
        
def KSm_2samples(data1, relative_time_grid1, data2, relative_time_grid2):
    """
    Calculate the modified two-sided Kolmogorov-Smirnov statistic, adjusted for
    right-censored time-to-death data, as described in Fleming et al (1980).

    This calcuate the residues for a non-parametric test for the null
    hypothesis that 2 independent samples are drawn from the same continuous 
    distribution. The actual test is performed in the function KSm_test.
    
    Reference
    ----------
    Thomas R. Fleming, Judith R. O'Fallon, Peter C. O'Brien and David P. 
    Harrington. Biometrics Vol. 36, No. 4 (Dec., 1980), pp. 607-625

    Parameters
    ----------
    data1, data2 : pandas DataFrame
        Contains the time-to-death observations. Should contain at columns
        named 'mortality' indicating time to event, and named 'status' 
        indicating event type: death (status==1) or censored (status==2)
    relative_time_grid1, relative_time_grid2 : array
        All observation timepoints. Should be a superset of data['mortality']

    Returns
    -------
    KM_residues : pandas Series
        KS residues, whose extremes are the KS statistic
    maximum_cdf :  float
        Estimated averaged c.d.f. at the maximal time. between (0,1), needed
        for calculating the asymptotic distribution of the KS statistic.
    """
    
    def deltaNA (d, Y):
        return np.sum([ np.power(Y-l,-1) for l in range(int(d))])

    mortalities1 = pd.Series(data1.loc[data1['status']==1,'mortality'].value_counts(), index=relative_time_grid1).fillna(0)
    censored1 = pd.Series(data1.loc[data1['status']==2,'mortality'].value_counts(), index=relative_time_grid1).fillna(0)
    counts1 = mortalities1 + censored1
    l1 = len(counts1)
    atRisk1 = pd.Series(data=np.dot(np.triu(np.ones((l1,l1)),k=0), counts1), index=relative_time_grid1)
    survived1 = pd.Series(data=np.dot(np.triu(np.ones((l1,l1)),k=1), counts1) + np.array(censored1), index=relative_time_grid1)
    
    mortalities2 = pd.Series(data2.loc[data2['status']==1,'mortality'].value_counts(), index=relative_time_grid2).fillna(0)
    censored2 = pd.Series(data2.loc[data2['status']==2,'mortality'].value_counts(), index=relative_time_grid2).fillna(0)
    counts2 = mortalities2 + censored2
    l2 = len(counts2)
    atRisk2 = pd.Series(data=np.dot(np.triu(np.ones((l2,l2)),k=0), counts2), index=relative_time_grid2)
    survived2 = pd.Series(data=np.dot(np.triu(np.ones((l2,l2)),k=1), counts2) + np.array(censored2), index=relative_time_grid2)
    
    #Merge the 2 time_grids
    unique_time_in_grid2 = [t for t in relative_time_grid2 if t not in relative_time_grid1]
    unique_time_in_grid1 = [t for t in relative_time_grid1 if t not in relative_time_grid2]
    
    relative_time_grid = np.unique(np.sort(relative_time_grid1+relative_time_grid2))
    mortalities1 = pd.concat([mortalities1,  pd.Series(data=float(0),index=unique_time_in_grid2)]).sort_index()
    mortalities2 = pd.concat([mortalities2,  pd.Series(data=float(0),index=unique_time_in_grid1)]).sort_index()
    
    def lastValueInTimeSeries(ts, t):
        if np.searchsorted(np.array(ts.index), t) != 0:
            return ts.iloc[np.searchsorted(np.array(ts.index), t)-1]
        else:
            return ts.iloc[0]
    
    atRisk1 = pd.concat([atRisk1,  pd.Series(data=[lastValueInTimeSeries(atRisk1, t) for t in unique_time_in_grid2],index=unique_time_in_grid2)]).sort_index()
    atRisk2 = pd.concat([atRisk2,  pd.Series(data=[lastValueInTimeSeries(atRisk2, t) for t in unique_time_in_grid1],index=unique_time_in_grid1)]).sort_index()
    n1 = np.sum(counts1)
    n2 = np.sum(counts2)
    n = n1 + n2
    relative_time_grid = np.unique(np.sort(list(atRisk1.index[atRisk1>0])+list(atRisk2.index[atRisk2>0])))
    l = len(relative_time_grid)
    
    beta1_NelsonAalen = np.dot(np.tril(np.ones((l,l)),k=0),  [deltaNA(mortalities1.loc[ind],atRisk1.loc[ind]) for ind in relative_time_grid] )
    beta2_NelsonAalen = np.dot(np.tril(np.ones((l,l)),k=0),  [deltaNA(mortalities2.loc[ind],atRisk2.loc[ind]) for ind in relative_time_grid] )
    alpha1minus_CumCensor = pd.Series(data=[ deltaNA(n1-atRisk1.loc[relative_time_grid[j]],n1) - beta1_NelsonAalen[j-1] for j in range(1,l)], index=relative_time_grid[1:])
    alpha2minus_CumCensor = pd.Series(data=[ deltaNA(n2-atRisk2.loc[relative_time_grid[j]],n2) - beta2_NelsonAalen[j-1] for j in range(1,l)], index=relative_time_grid[1:])
    
    eta_SqrtAvgN = pd.Series(data=[1/np.sqrt( 1/(n1*np.exp(-alpha1minus_CumCensor.loc[ind])) + 1/(n2*np.exp(-alpha2minus_CumCensor.loc[ind])) )
                                             for ind in relative_time_grid[1:]], index=relative_time_grid[1:])
    U_NelsonAalenDiffAdjusted = np.dot(np.tril(np.ones((l,l)),k=0), 
            [0]+[ eta_SqrtAvgN.loc[ind]*(deltaNA(mortalities1.loc[ind],atRisk1.loc[ind])-deltaNA(mortalities2.loc[ind],atRisk2.loc[ind]))
             for ind in relative_time_grid[1:]] )
    
    Y_n1n2 = pd.Series(data=0.5 * (np.exp(-beta1_NelsonAalen) + np.exp(-beta2_NelsonAalen)) * U_NelsonAalenDiffAdjusted, index=relative_time_grid )
    R_MaxCDF = 1 - 0.5 * (np.exp(-beta1_NelsonAalen[-1]) + np.exp(-beta2_NelsonAalen[-1]))
    
    return KSmResidues(Y_n1n2, R_MaxCDF)


def _criticalPartialBrownianBridge(alpha, maxcdf=0.999, alternative='two-sided'):
    
    R_MaxCDF = maxcdf
    R_func = np.sqrt(R_MaxCDF-R_MaxCDF*R_MaxCDF)
    
    if alternative=='two-sided':
        alpha = alpha/2
        
    return newton(lambda x:1-alpha-norm.cdf(x/R_func) + norm.cdf((2*R_MaxCDF-1)*x/R_func)*np.exp(-2*x*x), 1.5)

    
def KSm_test(residues, maxcdf, alternative='two-sided', alpha=0.05):
    """
    Performs the Kolmogorov-Smirnov 1-or-2-sided tests modified for right-
    censored time-to-death data, based on residues calculated from functions
    KSm_2samples or KSm_gof.

    The test statistic is based on a time-transformed Brownian bridge, more
    precisely, the sumprema of a Brownian bridge restricted to (0, maxcdf).
    Asymptotic distribution is described in Fleming et al (1980) and Schey 
    (1977). The two-sided version uses a conservative approximation described
    in Schey (1977), p ~ 2p(A,R), when the one-sided probability < 0.4.
    The exact two-sided probability could be calculated, but not yet
    implemented, see Koziol and Byar (1975).
    
    Reference
    ----------
    [1] Thomas R. Fleming, Judith R. O'Fallon, Peter C. O'Brien and David
    P. Harrington. Biometrics Vol. 36, No. 4 (Dec., 1980), pp. 607-625.
    [2] Schey, H. M. (1977). Communications in Statistics A6, 1361-1365.
    [3] Koziol, J. R. and Byar, D. P. (1975). Technometrics 17, 507-510.
    
    Parameters
    ----------
    residues : pandas Series
        First element of the output of functions KSm_2samples and KSm_gof
    maxcdf : float
        Second element of the output of functions KSm_2samples and KSm_gof
    alternative : {'two-sided', 'one-sided'}, optional
        Defines the alternative hypothesis (see explanation above).
        Default is 'two-sided'.
    alpha : float, optional
        Test size used to determine critical level for the statitic.
        Default is 0.05
    
    Returns
    -------
    statistic : float
        KS statistic
    pvalue : float or str
        p-value. When two-sided, could be  '>= 0.8'
    critical_stat : float
        Critical threshold for the statistic, determined by default or input 
        alpha
    """
     
    Ystats = residues
    R_MaxCDF = maxcdf
    R_func = np.sqrt(R_MaxCDF-R_MaxCDF*R_MaxCDF)
        
    if alternative=='two-sided':
        
        A_MaxAbsY = np.max(np.abs(Ystats))
        prob_BrownianBridge = 1-norm.cdf(A_MaxAbsY/R_func) + norm.cdf((2*R_MaxCDF-1)*A_MaxAbsY/R_func)*np.exp(-2*A_MaxAbsY*A_MaxAbsY)
        alpha = alpha/2
        
        if prob_BrownianBridge < 0.4:
            if alpha > 0.4:
                raise Exception('alpha >0.4 could not be used with Schey (1977) two-sided approximation')
            else:
                return KStestResult(A_MaxAbsY, prob_BrownianBridge*2, 
                    newton(lambda x:1-alpha-norm.cdf(x/R_func) + norm.cdf((2*R_MaxCDF-1)*x/R_func)*np.exp(-2*x*x), 1.5) )
        else:
            return KStestResult(A_MaxAbsY, '>= 0.8',
                newton(lambda x:1-alpha-norm.cdf(x/R_func) + norm.cdf((2*R_MaxCDF-1)*x/R_func)*np.exp(-2*x*x), 1.5) )
            
    if alternative=='one-sided':
        
        V_MaxY = np.max([0]+list(Ystats.values))
        prob_BrownianBridge = 1-norm.cdf(V_MaxY/R_func) + norm.cdf((2*R_MaxCDF-1)*V_MaxY/R_func)*np.exp(-2*V_MaxY*V_MaxY)
        return KStestResult(V_MaxY, prob_BrownianBridge, 
            newton(lambda x:1-alpha-norm.cdf(x/R_func) + norm.cdf((2*R_MaxCDF-1)*x/R_func)*np.exp(-2*x*x), 1.5) )
    else:
        raise Exception('Unrecognised alternative option:'+str(alternative))


def GGMfit_flxsrv_fit(data):
    
    flexsurvreg = rflexsurv.flexsurvreg
    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    #custom_env =  ro.r['new.env']()
    ro.r.source(os.path.join(os.path.split(pth)[0], 'Rmodels', 'GammaGompertzMakeham.R'), echo=False,verbose=False);
    #ro.r.attach(custom_env)
    gammagompertzmakeham = globalenv['gammagompertzmakeham']
    hgammagompertzmakeham = globalenv['hgammagompertzmakeham']
    Hgammagompertzmakeham = globalenv['Hgammagompertzmakeham']
    pgammagompertzmakeham = globalenv['pgammagompertzmakeham']
    
    rdf = pandas2ri.py2ri(data[['mortality','status']].applymap(float))
    flxsrv_fit = flexsurvreg(Formula('Surv(mortality,status==1)~1'),data=rdf, dist = gammagompertzmakeham,cl=(1-ALPHA))
    return flxsrv_fit
    
def GGfit_flxsrv_fit(data):
    
    flexsurvreg = rflexsurv.flexsurvreg
    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    #custom_env =  ro.r['new.env']()
    ro.r.source(os.path.join(os.path.split(pth)[0], 'Rmodels', 'GammaGompertz.R'), echo=False,verbose=False);
    #ro.r.attach(custom_env)
    gammagompertzmakeham = globalenv['gammagompertz']
    hgammagompertzmakeham = globalenv['hgammagompertz']
    Hgammagompertzmakeham = globalenv['Hgammagompertz']
    pgammagompertzmakeham = globalenv['pgammagompertz']
    
    rdf = pandas2ri.py2ri(data[['mortality','status']].applymap(float))
    flxsrv_fit = flexsurvreg(Formula('Surv(mortality,status==1)~1'),data=rdf, dist = gammagompertzmakeham,cl=(1-ALPHA))
    return flxsrv_fit

def GGMfit(data):
    
    flexsurvreg = rflexsurv.flexsurvreg
    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    #custom_env =  ro.r['new.env']()
    ro.r.source(os.path.join(os.path.split(pth)[0], 'Rmodels', 'GammaGompertzMakeham.R'), echo=False,verbose=False);
    #ro.r.attach(custom_env)
    gammagompertzmakeham = globalenv['gammagompertzmakeham']
    hgammagompertzmakeham = globalenv['hgammagompertzmakeham']
    Hgammagompertzmakeham = globalenv['Hgammagompertzmakeham']
    pgammagompertzmakeham = globalenv['pgammagompertzmakeham']
    
    rdf = pandas2ri.py2ri(data[['mortality','status']].applymap(float))
    flxsrv_fit = flexsurvreg(Formula('Surv(mortality,status==1)~1'),data=rdf, dist = gammagompertzmakeham,cl=(1-ALPHA))
    model_para = pandas2ri.ri2py(ro.r['as.data.frame'](flxsrv_fit.rx2('res')))
    
    def ML_survivorship(times= np.arange(0,np.max(data['mortality']),0.1) ):
        
        ptime = np.array(times) 
        rtime = ro.Vector(ptime)
        
        return np.array(pgammagompertzmakeham(rtime,
             model_para['est']['rate'],
             model_para['est']['beta'],
             model_para['est']['s'],
             model_para['est']['lambda'],lower_tail=False))
    
    def ML_cumulative_hazard(times= np.arange(0,np.max(data['mortality']),0.1) ):
        
        ptime = np.array(times) 
        rtime = ro.Vector(ptime)
        
        return np.array(Hgammagompertzmakeham(rtime,
             model_para['est']['rate'],
             model_para['est']['beta'],
             model_para['est']['s'],
             model_para['est']['lambda']))
    
    def ML_hazard(times= np.arange(0,np.max(data['mortality']),0.1) ):
        
        ptime = np.array(times) 
        rtime = ro.Vector(ptime)
        
        return np.array(hgammagompertzmakeham(rtime,
             model_para['est']['rate'],
             model_para['est']['beta'],
             model_para['est']['s'],
             model_para['est']['lambda']))
    
    return {'AIC': flxsrv_fit.rx2('AIC')[0], 
            'logLik': flxsrv_fit.rx2('loglik')[0],
            'events': flxsrv_fit.rx2('events')[0],
            'tRisk': flxsrv_fit.rx2('trisk')[0], 
            'model_paras': model_para,
            'ML_survivorship': ML_survivorship,
            'ML_cumulative_hazard': ML_cumulative_hazard,
            'ML_hazard': ML_hazard}
            
def GGfit(data):
    
    flexsurvreg = rflexsurv.flexsurvreg
    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    #custom_env =  ro.r['new.env']()
    ro.r.source(os.path.join(os.path.split(pth)[0], 'Rmodels', 'GammaGompertz.R'), echo=False,verbose=False);
    #ro.r.attach(custom_env)
    gammagompertz = globalenv['gammagompertz']
    hgammagompertz = globalenv['hgammagompertz']
    Hgammagompertz = globalenv['Hgammagompertz']
    pgammagompertz = globalenv['pgammagompertz']
    
    rdf = pandas2ri.py2ri(data[['mortality','status']].applymap(float))
    flxsrv_fit = flexsurvreg(Formula('Surv(mortality,status==1)~1'),data=rdf, dist = gammagompertz,cl=(1-ALPHA))
    model_para = pandas2ri.ri2py(ro.r['as.data.frame'](flxsrv_fit.rx2('res')))
    
    def ML_survivorship(times= np.arange(0,np.max(data['mortality']),0.1) ):
        
        ptime = np.array(times) 
        rtime = ro.Vector(ptime)
        
        return np.array(pgammagompertz(rtime,
             model_para['est']['beta'],
             model_para['est']['s'],
             model_para['est']['rate'],lower_tail=False))
    
    def ML_cumulative_hazard(times= np.arange(0,np.max(data['mortality']),0.1) ):
        
        ptime = np.array(times) 
        rtime = ro.Vector(ptime)
        
        return np.array(Hgammagompertz(rtime,
             model_para['est']['beta'],
             model_para['est']['s'],
             model_para['est']['rate']))
    
    def ML_hazard(times= np.arange(0,np.max(data['mortality']),0.1) ):
        
        ptime = np.array(times) 
        rtime = ro.Vector(ptime)
        
        return np.array(hgammagompertz(rtime,
             model_para['est']['beta'],
             model_para['est']['s'],
             model_para['est']['rate']))
    
    return {'AIC': flxsrv_fit.rx2('AIC')[0], 
            'logLik': flxsrv_fit.rx2('loglik')[0],
            'events': flxsrv_fit.rx2('events')[0],
            'tRisk': flxsrv_fit.rx2('trisk')[0], 
            'model_paras': model_para,
            'ML_survivorship': ML_survivorship,
            'ML_cumulative_hazard': ML_cumulative_hazard,
            'ML_hazard': ML_hazard}

def loglik_test(fsf_null, fsf_hy):
    
    D = -2*(fsf_null.rx2('loglik')[0] - fsf_hy.rx2('loglik')[0])
    df = fsf_hy.rx2('npars')[0]- fsf_null.rx2('npars')[0]
    return  D, df, chi2.sf(D,df)

    
def GGM_test_2samples(data_test, data_ref, covariate_name='dataset', covariate_create=True):
    
    flexsurvreg = rflexsurv.flexsurvreg
    ro.r.source(os.path.join(os.path.split(pth)[0], 'Rmodels', 'GammaGompertzMakeham.R'), echo=False,verbose=False);
    gammagompertzmakeham = globalenv['gammagompertzmakeham']
    hgammagompertzmakeham = globalenv['hgammagompertzmakeham']
    Hgammagompertzmakeham = globalenv['Hgammagompertzmakeham']
    pgammagompertzmakeham = globalenv['pgammagompertzmakeham']
    
    data1 = data_test.copy()
    data0 = data_ref.copy()
    if covariate_create:
        data0[covariate_name] = '0-ref'
        data1[covariate_name] = '1-strain'
    dfall = pd.concat([data0, data1])
    rdfall = pandas2ri.py2ri(dfall[['mortality','status',covariate_name]])
    
    formula_surv_str = 'Surv(mortality,status==1)'
    formulae = {'f0': '1',
                'f000': 'lambda('+covariate_name+')',
                'f001': 's('+covariate_name+')+lambda('+covariate_name+')',
                'f010': 'beta('+covariate_name+')+lambda('+covariate_name+')',
                'f011': 'beta('+covariate_name+')+s('+covariate_name+')+lambda('+covariate_name+')',
                'f100': covariate_name+'+lambda('+covariate_name+')',
                'f101': covariate_name+'+s('+covariate_name+')+lambda('+covariate_name+')',
                'f110': covariate_name+'+beta('+covariate_name+')+lambda('+covariate_name+')',
                'f111': covariate_name+'+beta('+covariate_name+')+s('+covariate_name+')+lambda('+covariate_name+')'}
    
    fs_ggm_fits = dict()                
    for key,formula_str in formulae.items():
        fs_ggm_fits[key] = flexsurvreg(Formula(formula_surv_str+'~'+formula_str), data=rdfall, dist = gammagompertzmakeham)
    
    fs_ggm_stats = dict()
    for key in formulae.keys():
        fs_ggm_stats[key]= { 'AIC':fs_ggm_fits[key].rx2('AIC')[0],
                            'npars':fs_ggm_fits[key].rx2('npars')[0],
                            'logLik': fs_ggm_fits[key].rx2('loglik')[0]
                            }
    
    formulae_sortedByAIC = sorted(fs_ggm_fits.keys(),key=lambda k: fs_ggm_fits[k].rx2('AIC')[0])
    bestHyptByAIC = formulae_sortedByAIC[0]
    hyptTested = pd.DataFrame(columns=['parameter','H0','H1','p-value','N(exp)','N(total)','AIC(H0)-AIC(H1)','npars(H1)-npars(H0)'])
    parNames = ['rate','beta','s']
    if bestHyptByAIC is not 'f0':
        for dgt in range(1,4):
            par_b = int(bestHyptByAIC[dgt])
            alterHypt = bestHyptByAIC[:dgt]+str(abs(par_b-1))+bestHyptByAIC[(dgt+1):]
            if par_b == 1:
                h0key = alterHypt
                h1key = bestHyptByAIC
            elif par_b == 0:
                h0key = bestHyptByAIC
                h1key = alterHypt
            (D_loglik, diff_npars, pv) = loglik_test(fs_ggm_fits[h0key],fs_ggm_fits[h1key])
            hyptTested.loc[len(hyptTested)] = { 'parameter': parNames[dgt-1],
                                'H0':formulae[h0key],
                                'H1':formulae[h1key],
                                'p-value':pv,
                                'AIC(H0)-AIC(H1)':fs_ggm_stats[h0key]['AIC']-fs_ggm_stats[h1key]['AIC'],
                                'npars(H1)-npars(H0)':diff_npars,
                                'N(exp)':len(data1),
                                'N(total)': len(dfall)}
    elif bestHyptByAIC == 'f0':
        alterHypt = 'f000'
        (D_loglik, df, pv) = loglik_test(fs_ggm_fits[bestHyptByAIC],fs_ggm_fits[alterHypt])
        hyptTested.loc[len(hyptTested)] = { 'parameter': 'lambda',
                            'H0':formulae[bestHyptByAIC],
                            'H1':formulae[alterHypt],
                            'p-value':pv,
                            'AIC(H0)-AIC(H1)':fs_ggm_stats[bestHyptByAIC]['AIC']-fs_ggm_stats[alterHypt]['AIC'],
                            'npars(H1)-npars(H0)':diff_npars,
                            'N(exp)':len(data1),
                            'N(total)':len(dfall)}
                            
    return {'bestHypothesisByAIC':formulae[bestHyptByAIC],
            'HypothesesTested':hyptTested, 
            'bestHypothesisFitPara':pandas2ri.ri2py(ro.r['as.data.frame'](fs_ggm_fits[bestHyptByAIC].rx2('res')))}


if __name__ == '__main__':

    import seaborn as sns
    sns.set_context("paper")
    sns.set_style("darkgrid")