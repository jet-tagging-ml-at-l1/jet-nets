from imports import *
from dataset import *
import argparse
from models import *
import tensorflow_model_optimization as tfmot
# from tensorflow_model_optimization.sparsity import keras as sparsity
# from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks, pruning_wrapper,  pruning_schedule
# from tensorflow_model_optimization.sparsity.keras import strip_pruning


from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
# import pandas as pd
# import shap
import json
import glob

import pdb

# All PF candidate properties
pfcand_fields_all = ['puppiweight','pt_rel','pt_rel_log',
                    'z0','dxy','dxy_custom','id','charge','pperp_ratio','ppara_ratio','deta','dphi','etarel','track_chi2',
                    'track_chi2norm','track_qual','track_npar','track_nstubs','track_vx','track_vy','track_vz','track_pterror',
                    'cluster_hovere','cluster_sigmarr','cluster_abszbarycenter','cluster_emet',
                    # 'cluster_egvspion','cluster_egvspu'
                    # 'pt','pt_log','px','py','pz','eta','phi','mass','energy','energy_log','pt_log',
                    ]
# A slightly reduced set
pfcand_fields_baseline = ['pt_rel','deta','dphi','charge','id',"track_vx","track_vy","track_vz"]
pfcand_fields_ext1 = ['pt_rel','deta','dphi','charge','id',"track_vx","track_vy","track_vz",
            'puppiweight','dxy_custom','etarel','pperp_ratio','ppara_ratio']


modelnamesDict = {
    "DeepSet": "QDeepSets_PermutationInv",
    "DeepSet-MHA": "QDeepSetsWithAttention_PermutationInv",
}

nconstit = 16

def doPlots(
        filetag,
        flavs,
        inputSetTags,
        modelNames,
        nnConfig,
        outname,
        splitTau,
        splitGluon,
        save = True,
        workdir = "./",):

    modelsAndNames = {}

    for inputSetTag in inputSetTags:

        tempflav = "btg"

        PATH = workdir + '/datasets_notreduced_chunked/' + filetag + "/" + tempflav + "/"
        outFolder = "performancePlots/"+outname+"/"
        if not os.path.exists(outFolder):
            os.makedirs(outFolder)

        if inputSetTag == "baseline":
            feature_names = pfcand_fields_baseline
        elif inputSetTag == "ext1":
            feature_names = pfcand_fields_ext1
        elif inputSetTag == "all":
            feature_names = pfcand_fields_all

        chunksmatching = glob.glob(PATH+"X_"+inputSetTag+"_test*.parquet")
        chunksmatching = [chunksm.replace(PATH+"X_"+inputSetTag+"_test","").replace(".parquet","").replace("_","") for chunksm in chunksmatching]

        filter = "/(jet)_(eta|phi|pt|pt_raw|mass|energy|bjetscore|tauscore|pt_corr/"

        print ("Loading data in all",len(chunksmatching),"chunks.")

        X_test = None
        X_test_global = None
        Y_test = None
        # for c in chunksmatching[:3]:
        for c in chunksmatching:
            if X_test is None:
                X_test = ak.from_parquet(PATH+"X_"+inputSetTag+"_test_"+c+".parquet")
                X_test_global = ak.from_parquet(PATH+"X_global_"+inputSetTag+"_test_"+c+".parquet")
                Y_test = ak.from_parquet(PATH+"Y_"+inputSetTag+"_test_"+c+".parquet")
            else:
                X_test =ak.concatenate((X_test, ak.from_parquet(PATH+"X_"+inputSetTag+"_test_"+c+".parquet")))
                X_test_global =ak.concatenate((X_test_global, ak.from_parquet(PATH+"X_global_"+inputSetTag+"_test_"+c+".parquet")))
                Y_test =ak.concatenate((Y_test, ak.from_parquet(PATH+"Y_"+inputSetTag+"_test_"+c+".parquet")))

        X_test = ak.to_numpy(X_test)
        Y_test = ak.to_numpy(Y_test)

        print("Loaded X_test      ----> shape:", X_test.shape)
        print("Loaded Y_test      ----> shape:", Y_test.shape)

        modelsAndNames[inputSetTag] = {}
        for flav in flavs:
            modelsAndNames[inputSetTag][flav] = {}
            for modelname in modelNames:
                modelsAndNames[inputSetTag][flav][modelname] = {}

                print ("Get performance for", inputSetTag, flav, modelname)

                # splitTau = "t" in flav
                # splitGluon = "g" in flav


                custom_objects_ = {
                    "AAtt": AAtt,
                    "QDense": QDense,
                    "QActivation": QActivation,
                    "quantized_bits": quantized_bits,
                    "ternary": ternary,
                    "binary": binary,
                    "QBatchNormalization": QBatchNormalization
                    }

                ncands = 16
                nfeatures = len(feature_names)
                nbits = 8

                labels = ["Bkg", "b"]
                if splitTau:
                    labels.append("Tau")
                if splitGluon:
                    labels.append("Gluon")

                # Get inference of some models

                trainingBasePath = "trainings/" + filetag + "_" + flav + "_" + inputSetTag + "_"

                print ("Load model",
                    trainingBasePath+""+modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)+'/model_'+modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)+'.h5')

                modelsAndNames[inputSetTag][flav][modelname]["model"] = tf.keras.models.load_model(
                    trainingBasePath+""+modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)+'/model_'+modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)+'.h5',
                    custom_objects=custom_objects_)
                modelsAndNames[inputSetTag][flav][modelname]["Y_predict"] = modelsAndNames[inputSetTag][flav][modelname]["model"].predict(X_test)

                # Plot the ROC curves
                fpr = {}
                tpr = {}
                auc1 = {}
                NN = {}
                NP = {}
                TP = {}
                FP = {}
                TN = {}
                FN = {}
                tresholds = {}

                # Loop over classes(labels) to get metrics per class
                for i, label in enumerate(labels):
                    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,i], modelsAndNames[inputSetTag][flav][modelname]["Y_predict"][:,i])
                    _ , N = np.unique(Y_test[:,i], return_counts=True) # count the NEGATIVES and POSITIVES samples in your test set
                    NN[label] = N[0]                   # number of NEGATIVES 
                    NP[label] = N[1]                   # number of POSITIVES
                    TP[label] = tpr[label]*NP[label]
                    FP[label] = fpr[label]*NN[label] 
                    TN[label] = NN[label] - FP[label]
                    FN[label] = NP[label] - TP[label]
                    auc1[label] = auc(fpr[label], tpr[label])
                    # plt.plot(tpr[label], fpr[label], label='%s tagger, AUC = %.1f%%'%(label, auc1[label]*100.))

                modelsAndNames[inputSetTag][flav][modelname]["ROCs"] = {}
                modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["tpr"] = tpr
                modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["fpr"] = fpr
                modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["auc"] = auc1

                modelsAndNames[inputSetTag][flav]["Reference"] = {}
                fpr = {}
                tpr = {}
                auc1 = {}
                NN = {}
                NP = {}
                TP = {}
                FP = {}
                TN = {}
                FN = {}
                tresholds = {}

                # Get reference ROCs
                label = "b"
                fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,1], X_test_global["jet_bjetscore"])
                _ , N = np.unique(Y_test[:,i], return_counts=True) # count the NEGATIVES and POSITIVES samples in your test set
                NN[label] = N[0] # number of NEGATIVES 
                NP[label] = N[1] # number of POSITIVES
                TP[label] = tpr[label]*NP[label]
                FP[label] = fpr[label]*NN[label] 
                TN[label] = NN[label] - FP[label]
                FN[label] = NP[label] - TP[label]
                auc1[label] = auc(fpr[label], tpr[label])

                if splitTau:
                    label = "Tau"
                    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,2], X_test_global["jet_tauscore"])
                    _ , N = np.unique(Y_test[:,i], return_counts=True) # count the NEGATIVES and POSITIVES samples in your test set
                    NN[label] = N[0] # number of NEGATIVES 
                    NP[label] = N[1] # number of POSITIVES
                    TP[label] = tpr[label]*NP[label]
                    FP[label] = fpr[label]*NN[label] 
                    TN[label] = NN[label] - FP[label]
                    FN[label] = NP[label] - TP[label]
                    auc1[label] = auc(fpr[label], tpr[label])

                modelsAndNames[inputSetTag][flav]["Reference"]["ROCs"] = {}
                modelsAndNames[inputSetTag][flav]["Reference"]["ROCs"]["tpr"] = tpr
                modelsAndNames[inputSetTag][flav]["Reference"]["ROCs"]["fpr"] = fpr
                modelsAndNames[inputSetTag][flav]["Reference"]["ROCs"]["auc"] = auc1


    # make the ROC plots
    # make one plot per truth tagger (b/tau)
    truthclass = "b"
    plt.figure()
    for inputSetTag in inputSetTags:
        for flav in flavs:
        # reference tagger, only once
            if inputSetTag == inputSetTags[0] and flav == flavs[0]:
                tpr = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["tpr"]
                fpr = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["fpr"]
                auc1 = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["auc"]
                plotlabel ="Reference"
                plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
            for modelname in modelNames:
                tpr = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["tpr"]
                fpr = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["fpr"]
                auc1 = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["auc"]
                plotlabel =modelname + " " + flav + " " + inputSetTag
                plt.plot(tpr[truthclass],fpr[truthclass],label='%s tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='lower right')
    hep.cms.label("Private Work", data=False, com = 14)
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
    plt.cla()

    if splitTau:
        truthclass = "Tau"
        for inputSetTag in inputSetTags:
            for flav in flavs:
            # reference tagger, only once
                if inputSetTag == inputSetTags[0] and flav == flavs[0]:
                    tpr = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["tpr"]
                    fpr = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["fpr"]
                    auc1 = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["auc"]
                    plotlabel ="Reference"
                    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
                for modelname in modelNames:
                    tpr = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["tpr"]
                    fpr = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["fpr"]
                    auc1 = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["auc"]
                    plotlabel =modelname + " " + flav + " " + inputSetTag
                    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
        plt.semilogy()
        plt.xlabel("Signal efficiency")
        plt.ylabel("Mistag rate")
        plt.xlim(0.,1.)
        plt.ylim(0.001,1)
        plt.grid(True)
        plt.legend(loc='lower right')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
        plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
        plt.cla()

    if splitGluon:
        truthclass = "Gluon"
        for inputSetTag in inputSetTags:
            for flav in flavs:
            # reference tagger, only once
                # if inputSetTag == inputSetTags[0] and flav == flavs[0]:
                #     tpr = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["tpr"]
                #     fpr = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["fpr"]
                #     auc1 = modelsAndNames[inputSetTags[0]][flavs[0]]["Reference"]["ROCs"]["auc"]
                #     plotlabel ="Reference"
                #     plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
                for modelname in modelNames:
                    tpr = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["tpr"]
                    fpr = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["fpr"]
                    auc1 = modelsAndNames[inputSetTag][flav][modelname]["ROCs"]["auc"]
                    plotlabel =modelname + " " + flav + " " + inputSetTag
                    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
        plt.semilogy()
        plt.xlabel("Signal efficiency")
        plt.ylabel("Mistag rate")
        plt.xlim(0.,1.)
        plt.ylim(0.001,1)
        plt.grid(True)
        plt.legend(loc='lower right')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
        plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
        plt.cla()


if __name__ == "__main__":
    from args import get_common_parser, handle_common_args
    # parser = argparse.ArgumentParser()
    parser = get_common_parser()
    parser.add_argument('-f','--file', help = 'input file name part')
    parser.add_argument('-o','--outname', help = 'output file name part')
    parser.add_argument('-c','--classes',nargs='+', help = 'Which flavors to run, options are b, bt, btg.')
    parser.add_argument('-i','--inputs',nargs='+', help = 'Which inputs to run, options are baseline, ext1, all.')
    parser.add_argument('-m','--models',nargs='+', help = 'Which models to evaluate, options are DeepSet, DeepSet-MHA.')
    parser.add_argument('--splitTau', dest = 'splitTau', default = False)
    parser.add_argument('--splitGluon', dest = 'splitGluon', default = False)
    parser.add_argument('--classweights', dest = 'classweights', default = False)


    args = parser.parse_args()
    handle_common_args(args)

    print('#'*30)
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('#'*30)


    nnConfig = {
        "classweights": args.classweights,
    }

    doPlots(
        args.file,
        args.classes,
        args.inputs,
        args.models,
        nnConfig,
        args.outname,
        args.splitTau,
        args.splitGluon,
        )