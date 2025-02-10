from utils import yemi_sorting_loader as ysl
from utils import local_paths

import importlib
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw

from utils import plottingHelper as ph
from utils import taskInfoFunctions as ti
from numpy.fft import fft, ifft
from scipy.stats import linregress
from utils import spikeTrainAnalysisHelpers as sta
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import pdb
import pprint
import sys
import json
import patchworklib as pw
from utils import dataPathDict as dpd
import pynapple as nap
from utils import pythonDebuggingHelpers as dh
import shelve
import pickle
import spikeinterface as si
from utils import pdfTextSet
from utils import spikeTrainAnalysisHelpers as sta
from utils import spikeInterfaceHelpers as sih

processed_data_dir=local_paths.get_processed_data_dir()

def process_channels(sorter_names, sessID,channels,scene_feature_tsd,trial_num_tsd,sceneSet_num_tsd,ep,numTrialsPerScene):
    #sortingResultsBaseDir='/home/ttj8/pythonCode'
    #sortingResultsBaseDir='/home/ttj8/project/pythonCodeProject'
    sortingResultsBaseDir='/home/ttj8/scratch60'
    minRateForUnit=1.5
    minRateForUnit=1
    #saveDir='/gpfs/milgram/pi/turk-browne/ttj8/pythonCodeProject/unitRastersMiyashita'
    saveDir='/gpfs/milgram/pi/turk-browne/ttj8/pythonCodeProject/rastersForYemiSortedUnits'
    #saveDataDir='/gpfs/milgram/pi/turk-browne/ttj8/pythonCodeProject/dataForYemiSortedUnits'
    #saveDataDir='/home/ttj8/scratch60/perTrialPklData'
    saveDataDir='/home/ttj8/project/pythonCodeProject/perTrialPklDataBackup/perTrialPklData'
    condNumToColor=ti.get_cond_num_to_color_list(sessID)
    
    for channel in channels:
            
        for sii,sorter_name in enumerate(sorter_names):
           
            fs=30000.0
            unitIDPerTimeStamp=ysl.load_unit_id_spike_train(sessID, channel)
            spike_times_np=np.array(unitIDPerTimeStamp['Spike Time'])
            unit_ids_np=np.array(unitIDPerTimeStamp['Unit ID'])
            #pdb.set_trace()
            #currChTimeStamps=nap.Tsd(t= spike_times_np, d=unit_ids_np )
            
            sorting = se.NumpySorting.from_times_labels(np.int_(spike_times_np*fs),unit_ids_np,fs)
            unitIDPerTimeStamp=sorting.get_all_spike_trains()
            currChTimeStamps=nap.Tsd(t=unitIDPerTimeStamp[0][0]/fs, d=unitIDPerTimeStamp[0][1])

            unit_ids=np.unique(currChTimeStamps.values)

            rate_over_session_per_unit=currChTimeStamps.to_tsgroup().count(1) #1 sec bins
            spike_amp_tsd_per_unit=[]
            #################################################
            #compute tuning curves
            #################################################
            #find intervals corresponding to first N trials and last N trials

            rasterSuperBrickPerSceneSetPerUnit={}
            earlyPsthPerSceneSetPerUnit={}
            latePsthPerSceneSetPerUnit={}
            deltaPsthPerSceneSetPerUnit={}
            PSTH_ratePerSceneSetPerUnit_PerScenePerTrialPerTime={}
            tuningCurve1PerSceneSetPerUnit={}
            tuningCurve2PerSceneSetPerUnit={}
            baselineCurve1PerSceneSetPerUnit={}
            baselineCurve2PerSceneSetPerUnit={}
            tuningToBaselineCurve1PerSceneSetPerUnit={}
            tuningToBaselineCurve2PerSceneSetPerUnit={}
            initialPeakSceneNumPerUnitPerCondition={}
            trialBrick=pw.Brick(figsize=(3,1))

            #only pt3... has one set of 150 scene displays
            if 'pt3' not in sessID:
                scene_set_nums=np.array([1,2])
            else:
                scene_set_nums=np.array([1])

            for scene_set_num in scene_set_nums: 
                N=3
                initial_trial_range=(1,N)
                final_trial_range=(15-N+1,15)
                trial_interval_epDict=sta.find_intervals_tsd_range(trial_num_tsd, [initial_trial_range,final_trial_range])
              
                curr_scene_set_ep=sta.find_intervals_tsd(sceneSet_num_tsd,[scene_set_num])[scene_set_num]
                
                ep1stNTrials=trial_interval_epDict[initial_trial_range]
                epLastNTrials=trial_interval_epDict[final_trial_range]
               
                ep1stNTrials=ep1stNTrials.intersect(curr_scene_set_ep)
                epLastNTrials=epLastNTrials.intersect(curr_scene_set_ep)

                ep1stNTrials=ep1stNTrials.time_span()
                epLastNTrials=epLastNTrials.time_span()
                
                trialBrick.plot(scene_feature_tsd.restrict(curr_scene_set_ep)+10*(scene_set_num-1),alpha=0.5,color='green',linewidth=0.25); 
                trialBrick.axvline(ep1stNTrials["start"][0],color=condNumToColor[scene_set_num],linewidth=3)
                trialBrick.axvline(ep1stNTrials["end"][0],color=condNumToColor[scene_set_num],linewidth=3)
                trialBrick.axvline(epLastNTrials["start"][0],color=condNumToColor[scene_set_num],linestyle='--',linewidth=3)
                trialBrick.axvline(epLastNTrials["end"][0],color=condNumToColor[scene_set_num],linestyle='--',linewidth=3)

                trialBrick.set_xlabel('Time in session (s)')
                trialBrick.set_ylabel('Scene no.')
                #trialBrick.set_title('Scene timing (-1=fixation cue)')
                #trialBrick.set_title('Scene timing (per condition)')
                trialBrick.yaxis.set_major_locator(MaxNLocator(integer=True))
                trialBrick.spines['top'].set_visible(False)
                trialBrick.spines['right'].set_visible(False)
                
                currChAndSceneSetTimeStamps=currChTimeStamps.restrict(curr_scene_set_ep)
                #print(currChAndSceneSetTimeStamps)
                currChTsGroup=currChAndSceneSetTimeStamps.to_tsgroup()
                
                #tuning_curve1=nap.compute_1d_tuning_curves(currChAndSceneSetTimeStamps.to_tsgroup(),scene_feature_tsd,nb_bins=12,ep=ep1stNTrials)

                curr_trial_gp_scene_ep1stN_dict=sta.find_intervals_tsd(scene_feature_tsd.restrict(ep1stNTrials),np.arange(10)+1)
                curr_trial_gp_scene_epLastN_dict=sta.find_intervals_tsd(scene_feature_tsd.restrict(epLastNTrials),np.arange(10)+1)
                tuning_curve1=nap.compute_discrete_tuning_curves(currChTsGroup,curr_trial_gp_scene_ep1stN_dict)
                tuning_curve2=nap.compute_discrete_tuning_curves(currChTsGroup,curr_trial_gp_scene_epLastN_dict)
                tuningCurve1PerSceneSetPerUnit[scene_set_num]=tuning_curve1
                tuningCurve2PerSceneSetPerUnit[scene_set_num]=tuning_curve2
                
                curr_trial_gp_baseline_ep1stN_dict=ti.get_baseline_ep_from_scene_ep_dict(curr_trial_gp_scene_ep1stN_dict)
                curr_trial_gp_baseline_epLastN_dict=ti.get_baseline_ep_from_scene_ep_dict(curr_trial_gp_scene_epLastN_dict)

                baseline_curve1=nap.compute_discrete_tuning_curves(currChTsGroup,curr_trial_gp_baseline_ep1stN_dict)
                baseline_curve2=nap.compute_discrete_tuning_curves(currChTsGroup,curr_trial_gp_baseline_epLastN_dict)
                baselineCurve1PerSceneSetPerUnit[scene_set_num]=baseline_curve1
                baselineCurve2PerSceneSetPerUnit[scene_set_num]=baseline_curve2
                
                tuningToBaselineCurve1PerSceneSetPerUnit[scene_set_num]=tuning_curve1-baseline_curve1
                tuningToBaselineCurve2PerSceneSetPerUnit[scene_set_num]=tuning_curve2-baseline_curve2
                
                #t=time.time();
                #elapsed=time.time()-t; print(f'{elapsed} sec')
                
                #################################################
                #plot rasters
                #################################################
                rasterSuperBrickPerSceneSetPerUnit[scene_set_num]={}
                earlyPsthPerSceneSetPerUnit[scene_set_num]={}
                latePsthPerSceneSetPerUnit[scene_set_num]={}
                deltaPsthPerSceneSetPerUnit[scene_set_num]={}
                PSTH_ratePerSceneSetPerUnit_PerScenePerTrialPerTime[scene_set_num]={}
                maxTuningRatePerUnit={}
                initialPeakSceneNumPerUnitPerCondition[scene_set_num]={}
                for ui,unit_id in enumerate(unit_ids):
                    currUnitTs=currChTsGroup.values()[ui]
                    currBaselinedTuning1=tuning_curve1[unit_id]-baseline_curve1[unit_id]
                    currBaselinedTuning2=tuning_curve2[unit_id]-baseline_curve2[unit_id]
                    maxTuningRate1=np.max(tuning_curve1[unit_id])
                    maxTuningRate2=np.max(tuning_curve2[unit_id])
                    maxTuningRate=np.max([1,maxTuningRate1,maxTuningRate2])
                    maxTuningRatePerUnit[unit_id]=maxTuningRate
                    earlyTuningResponseMean=np.mean(tuning_curve1[unit_id])

                    #maxBaselinedTuningRate=np.max([np.max(currBaselinedTuning1),np.max(currBaselinedTuning2)])
                    #minBaselinedTuningRate=np.min([np.min(currBaselinedTuning1),np.min(currBaselinedTuning2)])
                    
                    #maxBaselinedTuningRate=2*(maxTuningRate-earlyTuningResponseMean)
                    maxBaselinedTuningRate=1.5*(maxTuningRate-earlyTuningResponseMean)

                    initialPeakSceneNum=np.argmax(tuning_curve1[unit_id].values)+1
                    initialPeakSceneNumPerUnitPerCondition[scene_set_num][unit_id]=initialPeakSceneNum
                    #initialPeakSceneNum=np.argmax(currBaselinedTuning1.values)+1
                    #maxTuningRate=maxTuningRate1
                    superBrick={}
                    perieventTsGpPerScene=[]
                    for sci in range(10):
                        currSceneStr=f'Scene {sci+1}'
                        #currSceneTimes=ti.get_crossing_times_nap(scene_feature_tsd,sci+1)
                        currSceneTimes=ti.get_scene_start_times(scene_feature_tsd,sci+1)
                        currSceneTimeStamps=nap.Ts(t=currSceneTimes)
                        relTimeBounds=[-1,1]
                        trialBounds=[-1, numTrialsPerScene+1]
                        perieventTsGp=nap.compute_perievent(currChAndSceneSetTimeStamps.to_tsgroup(),currSceneTimeStamps,relTimeBounds)

                        perieventTsGpPerScene.append(perieventTsGp)
                    #end scene loop
                  
                    
                    rasterSuperBrick,psthPerSceneEarly,psthPerSceneLate,psthPerSceneChange,PSTH_ratePerScenePerTrialPerTime=sta.plot_all_scene_colored_raster(perieventTsGpPerScene,unit_id,relTimeBounds,scene_set_num,N,initialPeakSceneNumPerUnitPerCondition[scene_set_num][unit_id],-maxBaselinedTuningRate,maxBaselinedTuningRate,earlyTuningResponseMean,sessID)            
                    #May 3, 2024
                    rasterSuperBrick.savefig('exampleUnitRaster.pdf')
                    print('saved to exampleUnitRaster.pdf')
                    sys.exit(1)
                            
#sorter_names = ["klusta", "mountainsort5", "waveclus", "combinato"]
sorter_names = ["waveclus"]
sessID=sys.argv[1]
channels=sys.argv[2]

print(channels)
#processed_data_dir=local_paths.get_processed_data_dir()
time_axis,scene_num_per_time_step,trial_num_per_time_step,sceneSet_num_per_time_step,epoch1Bounds,epoch2Bounds,ttl_pulse_times=ti.load_scene_times_info(sessID,cacheDir=processed_data_dir)

ep1=nap.IntervalSet(start=epoch1Bounds[0],end=epoch1Bounds[1])
try:
    np.empty(epoch2Bounds)
    epoch2Bounds=epoch1Bounds
    ep2=ep1
    numTrialsPerScene=15    
except: #*** TypeError: 'numpy.float64' object cannot be interpreted as an integer
    ep2=nap.IntervalSet(start=epoch2Bounds[0],end=epoch2Bounds[1])
    #numTrialsPerScene=30
    numTrialsPerScene=15

epBuffer=5
epBuffer=0
epBuffer=1
epBoth=nap.IntervalSet(start=epoch1Bounds[0]-epBuffer,end=epoch2Bounds[1]+epBuffer)
scene_feature_tsd=nap.Tsd(t=time_axis,d=scene_num_per_time_step)
trial_num_tsd=nap.Tsd(t=time_axis,d=trial_num_per_time_step)
sceneSet_num_tsd=nap.Tsd(t=time_axis,d=sceneSet_num_per_time_step)

process_channels(sorter_names, sessID, [channels],scene_feature_tsd,trial_num_tsd,sceneSet_num_tsd,epBoth,numTrialsPerScene)

