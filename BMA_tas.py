from ILAMB.Confrontation    import Confrontation
from ILAMB.Variable         import Variable
from ILAMB.ModelResult      import ModelResult
#from mpl_toolkits.basemap   import Basemap
#from netCDF4                import Dataset
#from mpi4py                 import MPI
from sklearn                import linear_model
from sklearn.neighbors      import KernelDensity
from scipy.stats            import norm
from copy                   import deepcopy
import ILAMB.ilamblib as il
import numpy as np
import matplotlib.pyplot as plt
#import logging
import os

def ReadModelResult(model_root, models=[], model_year=[], filter="", regex=""):
    M = []
    if len(model_year) != 2: model_year = None
    max_model_name_len = 0
    print("\nSearching for model results in %s\n" % model_root)
    for subdir, dirs, files in os.walk(model_root):
        for mname in dirs:
            if len(models) > 0 and mname not in models: continue
            M.append(ModelResult(os.path.join(subdir,mname), modelname = mname, filter=filter, regex=regex, model_year = model_year))
            max_model_name_len = max(max_model_name_len,len(mname))
        break
    M = sorted(M,key=lambda m: m.name.upper())
    return M


def ReadBenchmark(filname, variable_name, alternate_vars="", study_limits=[]):
    obs = Variable(filename       = filname,
                variable_name  = variable_name,
                alternate_vars = alternate_vars,
                t0 = None if len(study_limits) != 2 else study_limits[0],
                tf = None if len(study_limits) != 2 else study_limits[1])
    if obs.time is None: raise il.NotTemporalVariable()
    return obs


def PrepareData(obs, M, variable, alternate_vars=""):
    MOD = []
    for m in M:
        mod = m.extractTimeSeries(variable,
                                    alt_vars     = alternate_vars,
                                    initial_time = obs.time_bnds[ 0,0],
                                    final_time   = obs.time_bnds[-1,1],
                                    lats         = None if obs.spatial else obs.lat,
                                    lons         = None if obs.spatial else obs.lon)
        obs, mod = il.MakeComparable(obs, mod,
                                        clif_ref = True,
                                        #extents   = self.extents,
                                        #logstring = "[%s][%s]" % (self.longname,m.name)
                                        )
        MOD.append(mod)
        
        ###############################################################################################
        ################## Maybe apply interpolation to all models? ####################################
        obs = obs.interpolate(lat=MOD[0].lat, lon=MOD[0].lon, itype="bilinear")
    return obs, MOD

def RunBMA(obs, MOD):
    n_mod = len(MOD)
    obs_data = obs.data
    obs_mask = obs_data.mask
    
    #Extract time
    length_train = np.shape(obs_data)[0]

    #Prepare data
    obs_data = obs_data.compressed().reshape((length_train, -1)) #approximately 360 * 18000
    dataset = [m.data[~obs_mask].reshape((length_train, -1)) for m in MOD]

    #bias correction
    def BiasCorrection(obs_data, dateset):
        dataset_new = []
        y_train = obs_data.reshape(-1, 1)
        for i in range(n_mod):
            x_train = dataset[i].reshape(-1, 1)
            reg = linear_model.LinearRegression()
            reg.fit(x_train, y_train)
            dataset_new.append(reg.predict(x_train).reshape((length_train, -1)))
        return dataset_new

    dataset = BiasCorrection(obs_data, dataset)
    
    #Stopping condition
    epsilon = 1e-3 
    B = 5000 

    def CalculateLikelihood(obs, dataset, w, sigma):
        l = np.log( ( (w.reshape(n_mod, 1, 1) * norm.pdf(np.tile(obs, (n_mod, 1, 1)), dataset, sigma)).sum(axis=0) ) + 1e-300 ).sum(axis=(0, 1))
        return l

    #Initialization
    w = np.zeros((n_mod, B))
    w[:, 0] = 1/n_mod 

    sigma = np.zeros(B)
    sigma[0] = np.sqrt(np.average((np.tile(obs_data, (n_mod, 1, 1)) - dataset)**2))

    l_0 = CalculateLikelihood(obs_data, dataset, w[:, 0], sigma[0])

    #Iteration
    for b in range(1, B):

        #E-step
        z = w[:, (b-1)].reshape(n_mod, 1, 1) * (norm.pdf( np.tile(obs_data, (n_mod, 1, 1)), dataset, sigma[b-1] ) + 1e-30)
        denominator = z.sum(axis=0) 
        z = z / denominator
        
        #M-step
        w[:, b] = np.average(z, axis=(1, 2))
        sigma[b] = np.average( (z * ( np.tile(obs_data, (n_mod, 1, 1)) - dataset)**2).sum(axis=0) )
        sigma[b] = np.sqrt(sigma[b])

        #stopping-condition
        np.seterr(over="ignore", under="ignore")
        l = CalculateLikelihood(obs_data, dataset, w[:, b], sigma[b])
        if np.absolute(l - l_0) > epsilon:
            l_0 = l
        else:
            break
    
    #v_BMA = deepcopy(MOD[0])
    #v_BMA.data = np.average(np.array([m.data for m in MOD]), axis = 0, weights = w)
    #v_BMA.data = np.ma.masked_array(v_BMA.data, mask=MOD[0].data.mask)
    #MOD.append(v_BMA)

    w = w[:, 0:b]
    sigma = sigma[0:b]

    data_BMA = np.empty_like(dataset[0])
    data_BMA = np.average(np.array(dataset), axis=0, weights=w[:, -1])
    dataset.append(data_BMA)

    np.savetxt("./tas_result/Weights_tas.txt", w)
    np.savetxt("./tas_result/Sigma_tas.txt", sigma)
    return w, sigma, obs_data, dataset

def PlotWeights(w):
    n_mod = w.shape[0]
    time = np.arange(w.shape[1])
    plt.figure(figsize=(12, 8))
    for i in range(n_mod):
        plt.plot(time, w[i], label="Model {i}".format(i=str(i+1)))
    plt.legend(loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("Weights")
    plt.savefig("./tas_result/Weights_tas.png", format="png", dpi=600)
    plt.close()
    return

def PlotModels(obs_data, dataset_BMA):

    #Calculate spatial average
    obs_mean = np.average(obs_data, axis=1)
    SpatialMean = np.average(dataset_BMA, axis=2)
    time = np.arange(obs_mean.shape[0])
    plt.figure(figsize=(12, 8))
    plt.plot(time, obs_mean, color="black", linestyle="--", label="Obs", lw = 0.5)
    for i in range(SpatialMean.shape[0]):
        plt.plot(time, SpatialMean[i], label="Model {i}".format(i=str(i+1)), lw = 0.5)
    plt.legend(loc="best")
    plt.xlabel("Time(Months)")
    plt.ylabel("Tas(K)")
    plt.savefig("./tas_result/SpatialMean_tas.png", format="png", dpi=600)
    plt.close()
    return

def CalculateCC(obs_data, dataset_BMA):
    #Calculate correlation coefficient
    #n_MOD = len(dataset_BMA)
    obs_data_row = obs_data.reshape(obs_data.size)
    #dataset_BMA_row = np.apply_over_axes(func1d=np.reshape, axis=0, arr=dataset_BMA, newshape=obs_data.size)
    dataset_BMA_row = []
    for i in range(len(dataset_BMA)):
        dataset_BMA_row.append(dataset_BMA[i].reshape(obs_data.size))
    dataset_BMA_row = np.array(dataset_BMA_row)
    corr = np.corrcoef(np.vstack((obs_data_row, dataset_BMA_row)))[0, 1:]
    #header = tuple( "Model {i}".format( i = str(k) ) for k in range(1, n_MOD))
    #header = header + tuple("BMA")
    #header = ", ".join(header)
    np.savetxt("./tas_result/Corr_tas.txt", corr, delimiter=",")
    return corr

def CalculateRMSE(obs_data, dataset_BMA):
    #def RMSE(mod):
    #    return np.sqrt(np.average((obs_data - mod)**2))
    #RMSE = np.apply_along_axis(RMSE, 0, dataset_BMA)
    RMSE = []
    for i in range(len(dataset_BMA)):
        RMSE.append( np.sqrt(np.average((obs_data - dataset_BMA[i])**2)) )
    RMSE = np.array(RMSE)
    np.savetxt("./tas_result/RMSE_tas.txt", RMSE, delimiter=",")
    return RMSE

def PlotDensity(obs_data, dataset_BMA, Bw):
    for bw in Bw:
        #bw = 1.5
        obs_mean = np.average(obs_data, axis=1)[:, np.newaxis]
        SpatialMean = np.average(dataset_BMA, axis=2)
        #X_plot = np.linspace(0, np.amax(np.vstack((obs_mean[:, 0], SpatialMean)))*2, 1000)[:, np.newaxis]
        X_plot = np.linspace(270, 297, 1000)[:, np.newaxis]
        plt.figure(figsize=(12, 8))
        for i in range(SpatialMean.shape[0]):
            if i == SpatialMean.shape[0] - 1:
                kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(SpatialMean[i].reshape((-1, 1)))
                log_dens = kde.score_samples(X_plot)
                plt.plot(X_plot[:, 0], np.exp(log_dens), label="Model BMA")
            else :
                kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(SpatialMean[i].reshape((-1, 1)))
                log_dens = kde.score_samples(X_plot)
                plt.plot(X_plot[:, 0], np.exp(log_dens), label="Model {i}".format(i=str(i+1)))
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(obs_mean)
        log_dens = kde.score_samples(X_plot)
        plt.plot(X_plot[:, 0], np.exp(log_dens), label="Obs")
        plt.legend(loc="best")
        plt.xlabel("Tas(K)")
        plt.ylabel("PDF")
        plt.savefig("./tas_result/Density_tas_bw={bw}.png".format(bw=bw), format="png", dpi=600)            
        plt.close()


if __name__ == "__main__":

    model_root = "/home/haoyang/ILAMB/ConfBMA/MODELS/"
    M = ReadModelResult(model_root)
    
    obs_file = "/home/haoyang/ILAMB/ConfBMA/DATA/tas/tas_CRU_1976_2005.nc"
    variable = "tas"
    obs = ReadBenchmark(obs_file, variable)

    obs, MOD = PrepareData(obs, M, variable)

    w, sigma, obs_data, dataset_BMA = RunBMA(obs, MOD)

    PlotWeights(w)
    CalculateCC(obs_data, dataset_BMA)
    CalculateRMSE(obs_data, dataset_BMA)
    PlotDensity(obs_data, dataset_BMA)
    #PlotModels(obs_data, dataset_BMA)

    



# class Model_BMA:
#     def __init__(self, **keywords):

        

