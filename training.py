from imports import *
from dataset import *
import argparse
from models import *
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks, pruning_wrapper,  pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning


from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd
import shap
import json
import glob
from datetime import datetime

import pdb

def plotInputFeatures(Xb, Xuds, Xtau, Xgluon, Xcharm, featureNames, outFolder, outputAddName = ""):
    print ("Plot all input features w/ add. name", outputAddName)
    plt.figure()
    # pdb.set_trace()
    for idxFeature, name in enumerate(featureNames):
        b_ = ak.flatten(Xb[:,:,idxFeature][Xb[:,:,0]!=0.],axis=-1)
        uds_ = ak.flatten(Xuds[:,:,idxFeature][Xuds[:,:,0]!=0.],axis=-1)
        g_ = ak.flatten(Xgluon[:,:,idxFeature][Xgluon[:,:,0]!=0.],axis=-1)
        tau_ = ak.flatten(Xtau[:,:,idxFeature][Xtau[:,:,0]!=0.],axis=-1)
        charm_ = ak.flatten(Xcharm[:,:,idxFeature][Xcharm[:,:,0]!=0.],axis=-1)
        min_ = min(min(b_), min(uds_))
        max_ = max(max(b_), max(uds_))
        # if splitTau:
        min_ = min(min_, min(tau_))
        max_ = max(max_, max(tau_))
        # if splitGluon:
        min_ = min(min_, min(g_))
        max_ = max(max_, max(g_))
        # if splitCharm:
        min_ = min(min_, min(charm_))
        max_ = max(max_, max(charm_))
        range = (min_, max_)
        plt.hist(b_, label='b', bins = 200, density = True, log = True, histtype = "step", range = range, color="blue")
        plt.hist(uds_, label='uds', bins = 200, density = True, log = True, histtype = "step", range = range, color="orange")
        plt.hist(g_, label='Gluon', bins = 200, density = True, log = True, histtype = "step", range = range, color="green")
        plt.hist(tau_, label='Tau', bins = 200, density = True, log = True, histtype = "step", range = range, color="red")
        plt.hist(charm_, label='Charm', bins = 200, density = True, log = True, histtype = "step", range = range, color="black")
        plt.legend(loc = "upper right")
        plt.xlabel(f"{name}")
        plt.ylabel('Jets (Normalized to 1)')
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+".png")
        plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+".pdf")
        plt.cla()


def rms(array):
   return np.sqrt(np.mean(array ** 2))

# All PF candidate properties
pfcand_fields_all = ['puppiweight','pt_rel','pt_rel_log',
                    'z0','dxy','dxy_custom','id','charge','pperp_ratio','ppara_ratio','deta','dphi','etarel','track_chi2',
                    'track_chi2norm','track_qual','track_npar','track_nstubs','track_vx','track_vy','track_vz','track_pterror',
                    'cluster_hovere','cluster_sigmarr','cluster_abszbarycenter','cluster_emet',
                    # 'cluster_egvspion','cluster_egvspu'
                    # 'pt_log','px','py','pz','eta','phi','mass','energy_log',
                    'pt_log','eta','phi',
                    ]
# A slightly reduced set
pfcand_fields_baseline = ['pt_rel','deta','dphi','charge','id',"track_vx","track_vy","track_vz"]
pfcand_fields_ext1 = ['pt_rel','deta','dphi','charge','id',"track_vx","track_vy","track_vz",
            'puppiweight','dxy_custom','etarel','pperp_ratio','ppara_ratio']
pfcand_fields_ext2 = ['pt_rel','deta','dphi','charge','id',"track_vx","track_vy","track_vz",
            'pt_log','eta','phi',]

def doTraining(
        filetag,
        flavs,
        inputSetTag,
        nnConfig,
        save = True,
        strstamp = "strstamp",
        workdir = "./",):

    # timestamp = datetime.now()
    # year = str(timestamp.year)
    # month = str(timestamp.month)
    # day = str(timestamp.day)
    # hour = str(timestamp.hour)
    # min = str(timestamp.minute)

    # strstamp = year+"_"+month+"_"+day+"-"+hour+"_"+min

    splitTau = "t" in flavs
    splitGluon = "g" in flavs
    splitCharm= "c" in flavs

    PATH = workdir + '/datasets/' + filetag + "/" + flavs + "/"
    if nnConfig["classweights"]:
        PATH = workdir + '/datasets_notreduced/' + filetag + "/" + flavs + "/"
    outFolder = "trainings/"
    if nnConfig["classweights"]:
        outFolder = "trainings_notreduced/"

    if nnConfig["regression"]:
        outFolder = outFolder.replace("trainings_","trainings_regression_")

    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)

    if inputSetTag == "baseline":
        feature_names = pfcand_fields_baseline
    elif inputSetTag == "ext1":
        feature_names = pfcand_fields_ext1
    elif inputSetTag == "ext2":
        feature_names = pfcand_fields_ext2
    elif inputSetTag == "all":
        feature_names = pfcand_fields_all

    nconstit = 16

    PATH_load = workdir + '/datasets_notreduced_chunked/' + filetag + "/" + flavs + "/"
    chunksmatching = glob.glob(PATH_load+"X_"+inputSetTag+"_test*.parquet")
    chunksmatching = [chunksm.replace(PATH_load+"X_"+inputSetTag+"_test","").replace(".parquet","").replace("_","") for chunksm in chunksmatching]

    # chunksmatching = chunksmatching[:5]

    # filter = "/(jet)_(eta|phi|pt|pt_raw|mass|energy|bjetscore|tauscore|pt_corr|genmatch_lep_vis_pt|genmatch_pt|label_b|label_uds|label_g|label_c|label_tau/"
    filter = "/(jet)_(eta|phi|pt|pt_raw|bjetscore|tauscore|pt_corr|genmatch_lep_vis_pt|genmatch_pt|label_b|label_uds|label_g|label_c|label_tau/"

    print ("Loading data in all",len(chunksmatching),"chunks.")

    X_train_val = None
    X_test = None
    # X_test_global = None
    Y_train_val = None
    Y_train_val_reg = None
    Y_test = None
    Y_test_reg = None
    X_train_global = None
    X_test_global = None
    x_b = None
    x_tau = None
    x_bkg = None
    x_gluon = None
    x_charm = None

    for c in chunksmatching:
        if X_test is None:
            X_test = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_test_"+c+".parquet")
            X_train_global = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_train_"+c+".parquet")
            X_test_global = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_test_"+c+".parquet")
            X_train_val = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_train_"+c+".parquet")
            # X_test_global = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_test_"+c+".parquet")
            Y_test = ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_test_"+c+".parquet")
            Y_test_reg = ak.from_parquet(PATH_load+"Y_target_"+inputSetTag+"_test_"+c+".parquet")
            Y_train_val = ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_train_"+c+".parquet")
            Y_train_val_reg = ak.from_parquet(PATH_load+"Y_target_"+inputSetTag+"_train_"+c+".parquet")
        else:
            X_test =ak.concatenate((X_test, ak.from_parquet(PATH_load+"X_"+inputSetTag+"_test_"+c+".parquet")))
            X_test_global =ak.concatenate((X_test_global, ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_test_"+c+".parquet")))
            X_train_global =ak.concatenate((X_train_global, ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_train_"+c+".parquet")))
            X_train_val =ak.concatenate((X_train_val, ak.from_parquet(PATH_load+"X_"+inputSetTag+"_train_"+c+".parquet")))
            # X_test_global =ak.concatenate((X_test_global, ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_test_"+c+".parquet")))
            Y_test =ak.concatenate((Y_test, ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_test_"+c+".parquet")))
            Y_test_reg =ak.concatenate((Y_test_reg, ak.from_parquet(PATH_load+"Y_target_"+inputSetTag+"_test_"+c+".parquet")))
            Y_train_val =ak.concatenate((Y_train_val, ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_train_"+c+".parquet")))
            Y_train_val_reg =ak.concatenate((Y_train_val_reg, ak.from_parquet(PATH_load+"Y_target_"+inputSetTag+"_train_"+c+".parquet")))

        x_b_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_b_"+c+".parquet")
        if len(x_b_) > 0:
            if x_b is None:
                x_b = x_b_
            else:
                x_b =ak.concatenate((x_b, x_b_))

        x_bkg_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_bkg_"+c+".parquet")
        if len(x_bkg_) > 0:
            if x_bkg is None:
                x_bkg = x_bkg_
            else:
                x_bkg =ak.concatenate((x_bkg, x_bkg_))

        if splitTau:
            x_tau_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_tau_"+c+".parquet")
            if len(x_tau_) > 0:
                if x_tau is None:
                    x_tau = x_tau_
                else:
                    x_tau =ak.concatenate((x_tau, x_tau_))

        if splitGluon:
            x_gluon_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_gluon_"+c+".parquet")
            if len(x_gluon_) > 0:
                if x_gluon is None:
                    x_gluon = x_gluon_
                else:
                    x_charm =ak.concatenate((x_gluon, x_gluon_))

        if splitCharm:
            x_charm_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_charm_"+c+".parquet")
            if len(x_charm_) > 0:
                if x_charm is None:
                    x_charm = x_charm_
                else:
                    x_charm =ak.concatenate((x_charm, x_charm_))


    x_b = ak.to_numpy(x_b)
    x_bkg = ak.to_numpy(x_bkg)
    if splitTau:
        x_tau = ak.to_numpy(x_tau)
    if splitGluon:
        x_gluon = ak.to_numpy(x_gluon)
    if splitCharm:
        x_charm = ak.to_numpy(x_charm)


    # rebalance data set
    X_train_val = ak.to_numpy(X_train_val)
    X_test = ak.to_numpy(X_test)
    Y_train_val = ak.to_numpy(Y_train_val)
    Y_train_val_reg = ak.to_numpy(Y_train_val_reg)
    Y_test = ak.to_numpy(Y_test)
    Y_test_reg = ak.to_numpy(Y_test_reg)

    print("Loaded X_train_val ----> shape:", X_train_val.shape)
    print("Loaded X_test      ----> shape:", X_test.shape)
    print("Loaded Y_train_val ----> shape:", Y_train_val.shape)
    print("Loaded Y_test      ----> shape:", Y_test.shape)

    nbits = nnConfig["nbits"]
    integ = nnConfig["integ"]
    nfeat = X_train_val.shape[-1]

    if nnConfig["model"] == "DeepSet":
        model, modelname, custom_objects = getDeepSet(nclasses = len(Y_train_val[0]), input_shape = (nconstit, nfeat),
                                                      nnodes_phi = nnConfig["nNodes"], nnodes_rho = nnConfig["nNodes"],
                                                      nbits = nbits, integ = integ, addRegression = nnConfig["regression"])
    elif nnConfig["model"] == "MLP":
        model, modelname, custom_objects = getMLP(nclasses = len(Y_train_val[0]), input_shape = (nconstit, nfeat),
                                                      nnodes_phi = nnConfig["nNodes"], nnodes_rho = nnConfig["nNodes"],
                                                      nbits = nbits, integ = integ, addRegression = nnConfig["regression"])
    elif nnConfig["model"] == "DeepSet-MHA":
        model, modelname, custom_objects = getDeepSetWAttention(nclasses = len(Y_train_val[0]), input_shape = (nconstit, nfeat),
                                                                nnodes_phi = nnConfig["nNodes"], nnodes_rho = nnConfig["nNodes"],
                                                                nbits = nbits, integ = integ,
                                                                n_head = nnConfig["nHeads"], dim = nfeat, dim2 = nnConfig["nNodesHead"], addRegression = nnConfig["regression"])
    elif nnConfig["model"] == "MLP-MHA":
        model, modelname, custom_objects = getMLPWAttention(nclasses = len(Y_train_val[0]), input_shape = (nconstit, nfeat),
                                                                nnodes_phi = nnConfig["nNodes"], nnodes_rho = nnConfig["nNodes"],
                                                                nbits = nbits, integ = integ,
                                                                n_head = nnConfig["nHeads"], dim = nfeat, dim2 = nnConfig["nNodesHead"], addRegression = nnConfig["regression"])
        

    if nnConfig["pruning"]:
        modelname = modelname + "_pruned"
    if nnConfig["inputQuant"]:
        modelname = modelname + "_inputQuant"

    print('Model name :', modelname)


    # outFolder = outFolder + "/" + filetag + "_" + flavs + "_" + inputSetTag + "_" + modelname + "/"
    outFolder = outFolder + "/"+ strstamp + "_" + filetag + "_" + flavs + "_" + inputSetTag + "_" + modelname + "/"
    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)
    print ("Use the following output folder:", outFolder)

    plotInputFeatures(x_b, x_bkg, x_tau, x_gluon, x_charm, feature_names, outFolder, outputAddName = "")

    if nnConfig["inputQuant"]:
        input_quantizer = quantized_bits(bits=16, integer=6, symmetric=0, alpha=1)
        x_b = input_quantizer(x_b.astype(np.float32)).numpy()
        x_bkg = input_quantizer(x_bkg.astype(np.float32)).numpy()
        x_tau = input_quantizer(x_tau.astype(np.float32)).numpy()
        x_gluon = input_quantizer(x_gluon.astype(np.float32)).numpy()
        x_charm = input_quantizer(x_charm.astype(np.float32)).numpy()


        plotInputFeatures(x_b, x_bkg, x_tau, x_gluon, x_charm, feature_names, outFolder, outputAddName = "_quant")

    # calculate class weights
    # if nnConfig["classweights"]:
    # lengths = [sum(Y_train_val[:,idx]) for idx in range(Y_train_val.shape[1])]
    # class_weights = [s/sum(lengths) for s in lengths]
    # class_weights = [w/np.mean(class_weights) for w in class_weights]
    # class_weights = {idx: class_weights[idx] for idx in range(len(class_weights))}
    # print ("Use the following class weights")
    # print (class_weights)

    # c_w = np.array([class_weights[key] for key in class_weights])

    # pdb.set_trace()
    # sample_weights  = np.sum(Y_train_val * c_w, axis =-1)

    # class_weights = np.array([1., 1., 1., 1., 2.])
    class_weights = np.array([1., 1., 1., 1., 1.])

    # get pT weights, weighting all to b spectrum
    bins_pt_weights = np.array([15, 20, 25, 30, 38, 48, 60, 76, 97, 122, 154, 195, 246, 311, 393, 496, 627, 792, 9999999999999999999])
    bins_eta_weights = np.array([-99, -1.5, -0.5, 0., 0.5, 1.5, 99])
    counts_b, edges_b = np.histogram(X_train_global[X_train_global["label_b"]>0]["jet_pt"], bins = bins_pt_weights) 
    counts_uds, edges_uds = np.histogram(X_train_global[X_train_global["label_uds"]>0]["jet_pt"], bins = bins_pt_weights) 
    counts_g, edges_g = np.histogram(X_train_global[X_train_global["label_g"]>0]["jet_pt"], bins = bins_pt_weights) 
    counts_c, edges_c = np.histogram(X_train_global[X_train_global["label_c"]>0]["jet_pt"], bins = bins_pt_weights) 
    counts_tau, edges_tau = np.histogram(X_train_global[X_train_global["label_tau"]>0]["jet_pt"], bins = bins_pt_weights) 
    # counts_b, edges_b, edges_b2 = np.histogram2d(X_train_global[X_train_global["label_b"]>0]["jet_pt"],X_train_global[X_train_global["label_b"]>0]["jet_eta"], bins = (bins_pt_weights, bins_eta_weights)) 
    # counts_uds, edges_uds, edges_uds2 = np.histogram2d(X_train_global[X_train_global["label_uds"]>0]["jet_pt"],X_train_global[X_train_global["label_uds"]>0]["jet_eta"], bins = (bins_pt_weights, bins_eta_weights)) 
    # counts_g, edges_g, edges_g2 = np.histogram2d(X_train_global[X_train_global["label_g"]>0]["jet_pt"],X_train_global[X_train_global["label_g"]>0]["jet_eta"], bins = (bins_pt_weights, bins_eta_weights)) 
    # counts_c, edges_c, edges_c2 = np.histogram2d(X_train_global[X_train_global["label_c"]>0]["jet_pt"],X_train_global[X_train_global["label_c"]>0]["jet_eta"], bins = (bins_pt_weights, bins_eta_weights)) 
    # counts_tau, edges_tau, edges_tau2 = np.histogram2d(X_train_global[X_train_global["label_tau"]>0]["jet_pt"],X_train_global[X_train_global["label_tau"]>0]["jet_eta"], bins = (bins_pt_weights, bins_eta_weights)) 

    print(counts_b, counts_uds, counts_g, counts_c, counts_tau)

    w_b = np.nan_to_num(counts_b/counts_b * class_weights[0], nan = 1., posinf = 1., neginf = 1.)
    w_uds =  np.nan_to_num(counts_b/counts_uds * class_weights[1], nan = 1., posinf = 1., neginf = 1.)
    w_g =  np.nan_to_num(counts_b/counts_g * class_weights[2], nan = 1., posinf = 1., neginf = 1.)
    w_c =  np.nan_to_num(counts_b/counts_c * class_weights[3], nan = 1., posinf = 1., neginf = 1.)
    w_tau =  np.nan_to_num(counts_b/counts_tau * class_weights[4], nan = 1., posinf = 1., neginf = 1.)

    # pt_binidx = np.digitize(X_train_global["jet_pt"], bins_pt_weights)-1
    # pt_onehot = to_categorical(pt_binidx)

    # w_pt_b = np.sum(pt_onehot*w_b, axis=-1)
    # w_pt_uds = np.sum(pt_onehot*w_uds, axis=-1)
    # w_pt_g = np.sum(pt_onehot*w_g, axis=-1)
    # w_pt_c = np.sum(pt_onehot*w_c, axis=-1)
    # w_pt_tau = np.sum(pt_onehot*w_tau, axis=-1)

    X_train_global["weight_jetpT_binidx"] = to_categorical( np.digitize(X_train_global["jet_pt"], bins_pt_weights)-1)

    X_train_global["weight_pt"] = (X_train_global["label_b"]*ak.sum(w_b*X_train_global["weight_jetpT_binidx"], axis =-1) + X_train_global["label_uds"]*ak.sum(w_uds*X_train_global["weight_jetpT_binidx"], axis =-1) + X_train_global["label_g"]*ak.sum(w_g*X_train_global["weight_jetpT_binidx"], axis =-1) + X_train_global["label_c"]*ak.sum(w_c*X_train_global["weight_jetpT_binidx"], axis =-1) + X_train_global["label_tau"]*ak.sum(w_tau*X_train_global["weight_jetpT_binidx"], axis =-1))

    # w_all = np.array([w_pt_b,w_pt_uds,w_pt_g,w_pt_c,w_pt_tau])

    sample_weights = ak.to_numpy(X_train_global["weight_pt"])
    sample_weights = (sample_weights/np.mean(sample_weights))

    print ("Using sample weights:", sample_weights)
    # sample_weights = np.clip(np.nan_to_num(sample_weights, nan = 1., posinf = 1., neginf = 1.), 0.01, 100.)
    sample_weights = np.nan_to_num(sample_weights, nan = 1., posinf = 1., neginf = 1.)
    print ("Using sample weights fixed:", sample_weights)

    # plot the weight distributions
    plt.figure()
    bins = np.linspace(0., 20., 1000)
    # pdb.set_trace()
    plt.hist(X_train_global[X_train_global["label_b"] > 0]["weight_pt"], label='b', bins = bins)
    plt.hist(X_train_global[X_train_global["label_uds"] > 0]["weight_pt"], label='uds', bins = bins)
    plt.hist(X_train_global[X_train_global["label_g"] > 0]["weight_pt"], label='Gluon', bins = bins)
    plt.hist(X_train_global[X_train_global["label_tau"] > 0]["weight_pt"], label='Tau', bins = bins)
    plt.hist(X_train_global[X_train_global["label_c"] > 0]["weight_pt"], label='Charm', bins = bins)
    plt.legend(loc = "upper right")
    plt.xlabel('Weights')
    plt.ylabel('Counts')
    hep.cms.label("Private Work", data = False, com = 14)
    plt.savefig(outFolder+"/weights_"+inputSetTag+".png")
    plt.savefig(outFolder+"/weights_"+inputSetTag+".pdf")
    plt.cla()

    # nfeat = X_train_val.shape[-1]

    # Define the optimizer ( minimization algorithm )
    if nnConfig["optimizer"] == "adam":
        optim = Adam(learning_rate = nnConfig["learning_rate"])
    elif nnConfig["optimizer"] == "ranger":
        radam = tfa.optimizers.RectifiedAdam(learning_rate = nnConfig["learning_rate"])
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        optim = ranger

    if nnConfig["regression"] == True:
        # model.compile(optimizer = optim, loss = ['categorical_crossentropy', 'mean_squared_error'],
        #             #   metrics = ['categorical_accuracy', 'mae'],
        #               loss_weights=[1., 1.]
        # #               )
        # model.compile(optimizer=optim, loss={'output_class': 'categorical_crossentropy', 'output_reg': 'mean_squared_error'},
        #               metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']}, loss_weights=[1., 1.])
        model.compile(optimizer=optim, loss={'output_class': 'categorical_crossentropy', 'output_reg': 'log_cosh'},
                      metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']}, loss_weights=[1., 1.])
    else:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # if nnConfig["pruning"]:
    #     # The code bellow allows pruning of selected layers of the model
        
    #     def pruneFunction(layer):
    #         pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.5, begin_step=6000, frequency=10)}
                
    #         # Apply prunning to Dense layers type excluding the output layer
    #         if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'dense_out': # exclude output_dense
    #             return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    #         return layer

    #     # Clone model to apply "pruneFunction" to model layers 
    #     model = tf.keras.models.clone_model(model, clone_function=pruneFunction)

    # if nnConfig["regression"] == False:
    #     # compile the model
    #     model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print ("Model summary:")
    # print the model summary
    model.summary()

    
    # outFolder = outFolder + "/" + filetag + "_" + flavs + "_" + inputSetTag + "_" + modelname + "/"
    # if not os.path.exists(outFolder):
    #     os.makedirs(outFolder, exist_ok=True)
    # print ("Use the following output folder:", outFolder)

    print ("Dump nnConfig to file:", outFolder+"/nnConfig.json")
    json_object = json.dumps(nnConfig, indent=4)
    with open(outFolder+"/nnConfig.json", "w") as outfile:
        outfile.write(json_object)
    # ------------------------------------------------------
    # Run training


    # Figure o merit to monitor during training
    merit = 'val_loss'

    # early stopping callback
    es = EarlyStopping(monitor=merit, patience = 10)
    # Learning rate scheduler 
    ls = ReduceLROnPlateau(monitor=merit, factor=0.2, patience=10)
    # model checkpoint callback
    # this saves our model architecture + parameters into mlp_model.h5
    chkp = ModelCheckpoint(outFolder+'/model_'+modelname+'.h5', monitor = merit, 
                                    verbose = 0, save_best_only = True, 
                                    save_weights_only = False, mode = 'auto', 
                                    save_freq = 'epoch')

    #tb = TensorBoard("/Users/sznajder/WorkM1/miniforge3/tensorflow_macos/arm64/workdir/logs")

    # callbacks_=[es,ls,chkp])
    # callbacks_=[es,chkp])
    callbacks_ = [chkp]
    if nnConfig["pruning"]:
        # Prunning callback
        pr = pruning_callbacks.UpdatePruningStep()
        callbacks_.append(pr)

    # print ("sample_weights",min(sample_weights),max(sample_weights))
    # print ("Y_train_val",min(Y_train_val),max(Y_train_val))
    # print ("Y_train_val_reg",min(Y_train_val_reg),max(Y_train_val_reg))

    if nnConfig["inputQuant"]:
        input_quantizer = quantized_bits(bits=16, integer=6, symmetric=0, alpha=1)
        X_train_val = input_quantizer(X_train_val.astype(np.float32)).numpy()
        X_test = input_quantizer(X_test.astype(np.float32)).numpy()

    # Train classifier
    if nnConfig["regression"] == True:
        history = model.fit(x = X_train_val,
                            y = [Y_train_val, Y_train_val_reg], 
                            epochs = nnConfig["epochs"], 
                            batch_size = nnConfig["batch_size"], 
                            verbose = 1,
                            validation_split = nnConfig["validation_split"],
                            callbacks = callbacks_,
                            # class_weight = class_weights,
                            sample_weight = sample_weights,
                            shuffle=True)
    else:
        history = model.fit(X_train_val, Y_train_val, 
                            epochs = nnConfig["epochs"], 
                            batch_size = nnConfig["batch_size"], 
                            verbose = 1,
                            validation_split = nnConfig["validation_split"],
                            callbacks=callbacks_,
                            class_weight = class_weights,
                            shuffle=True)
    
    custom_objects_ = {}
    if custom_objects is not None:
        for co in custom_objects:   
            custom_objects_[co] = custom_objects[co]

    if nnConfig["pruning"]:
        # Strip the model 
        model = strip_pruning(model)
        if nnConfig["regression"] == True:
            model.compile(optimizer=optim, loss={'output_class': 'categorical_crossentropy', 'output_reg': 'log_cosh'},
                        metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']}, loss_weights=[1., 1.])
        else:
            model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        # Print the stripped model summary
        model.summary()
        # Save the stripped and prunned model
        # model.save(outFolder+'/model_'+modelname+'.h5', custom_objects=custom_objects_)
        model.save(outFolder+'/model_'+modelname+'.h5')


    # Plot performance

    # Here, we plot the history of the training and the performance in a ROC curve using the best saved model

    # Load the best saved model
    model = tf.keras.models.load_model(outFolder+'/model_'+modelname+'.h5', custom_objects=custom_objects_)



    #plt.rcParams['axes.unicode_minus'] = False

    if nnConfig["regression"] == False:
        # Plot loss vs epoch
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend(loc="upper right")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/loss_"+inputSetTag+".png")
        plt.savefig(outFolder+"/loss_"+inputSetTag+".pdf")
        plt.cla()

        # Plot accuracy vs epoch
        plt.plot(history.history['categorical_accuracy'], label='Accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='Validation accuracy')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/acc_"+inputSetTag+".png")
        plt.savefig(outFolder+"/acc_"+inputSetTag+".pdf")
        plt.cla()

    if nnConfig["regression"]:
        # Plot loss vs epoch
        plt.plot(history.history['output_class_loss'], label='class loss')
        plt.plot(history.history['val_output_class_loss'], label='val class loss')
        plt.legend(loc="upper right")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/loss_class_"+inputSetTag+".png")
        plt.savefig(outFolder+"/loss_class_"+inputSetTag+".pdf")
        plt.cla()

        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend(loc="upper right")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/loss_"+inputSetTag+".png")
        plt.savefig(outFolder+"/loss_"+inputSetTag+".pdf")
        plt.cla()

        # Plot accuracy vs epoch
        plt.plot(history.history['output_class_categorical_accuracy'], label='Accuracy')
        plt.plot(history.history['val_output_class_categorical_accuracy'], label='Validation accuracy')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/acc_"+inputSetTag+".png")
        plt.savefig(outFolder+"/acc_"+inputSetTag+".pdf")
        plt.cla()

        # Plot reg loss vs epoch
        plt.plot(history.history['output_reg_loss'], label='Regression loss')
        plt.plot(history.history['val_output_reg_loss'], label='Validation regression loss')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/loss_reg_"+inputSetTag+".png")
        plt.savefig(outFolder+"/loss_reg_"+inputSetTag+".pdf")
        plt.cla()

        # Plot reg loss (MAE) vs epoch
        plt.plot(history.history['output_reg_mae'], label='MAE')
        plt.plot(history.history['val_output_reg_mae'], label='Validation MAE')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/mae_reg_"+inputSetTag+".png")
        plt.savefig(outFolder+"/mae_reg_"+inputSetTag+".pdf")
        plt.cla()

        # Plot reg loss (MSE) vs epoch
        plt.plot(history.history['output_reg_mean_squared_error'], label='MSE')
        plt.plot(history.history['val_output_reg_mean_squared_error'], label='Validation MSE')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/mse_reg_"+inputSetTag+".png")
        plt.savefig(outFolder+"/mse_reg_"+inputSetTag+".pdf")
        plt.cla()

    # Plot the ROC curves
    labels = ["Bkg", "b"]
    if splitTau:
        labels.append("Tau")
    if splitGluon:
        labels.append("Gluon")
    if splitCharm:
        labels.append("Charm")
    fpr = {}
    tpr = {}
    auc1 = {}
    precision = {}
    recall = {}
    NN = {}
    NP = {}
    TP = {}
    FP = {}
    TN = {}
    FN = {}
    tresholds = {}


    if nnConfig["regression"]:
        Y_predict = model.predict(X_test)
        Y_predict_reg = Y_predict[1]
        Y_predict = Y_predict[0]
    else:
        Y_predict = model.predict(X_test)
    

    # Loop over classes(labels) to get metrics per class
    for i, label in enumerate(labels):
        # print (Y_test[:,i],Y_predict[:,i])
        fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,i], Y_predict[:,i])
        #precision[label], recall[label], tresholds = precision_recall_curve(Y_test[:,i], Y_predict[:,i]) 
        # print( np.unique(Y_test[:,i], return_counts=True) )
        _ , N = np.unique(Y_test[:,i], return_counts=True) # count the NEGATIVES and POSITIVES samples in your test set
        NN[label] = N[0]                   # number of NEGATIVES 
        NP[label] = N[1]                   # number of POSITIVES
        TP[label] = tpr[label]*NP[label]
        FP[label] = fpr[label]*NN[label] 
        TN[label] = NN[label] - FP[label]
        FN[label] = NP[label] - TP[label]
        plt.grid()
        auc1[label] = auc(fpr[label], tpr[label])
        plt.plot(tpr[label],fpr[label],label='%s tagger, AUC = %.1f%%'%(label,auc1[label]*100.))


    #fpr, tpr, tresholds = roc_curve(Y_test, Y_predict)
    #auc1 = auc(fpr, tpr)
    #plt.plot(tpr,fpr,label='%s tagger, auc = %.1f%%'%("b",auc1*100.))

    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='lower right')
    hep.cms.label("Private Work", data=False, com = 14)
    plt.savefig(outFolder+"/ROC_"+inputSetTag+".png")
    plt.savefig(outFolder+"/ROC_"+inputSetTag+".pdf")
    plt.cla()



    # Plot DNN output 
    y_b_predict = model.predict(x_b)
    y_bkg_predict = model.predict(x_bkg)
    if splitTau: y_tau_predict = model.predict(x_tau)
    if splitGluon: y_gluon_predict = model.predict(x_gluon)
    if splitCharm:y_charm_predict = model.predict(x_charm)


    if nnConfig["regression"]:
        y_b_predict_reg = y_b_predict[1]
        y_b_predict = y_b_predict[0]
        y_bkg_predict_reg = y_bkg_predict[1]
        y_bkg_predict = y_bkg_predict[0]
        if splitTau:
            y_tau_predict_reg = y_tau_predict[1]
            y_tau_predict = y_tau_predict[0]
        if splitCharm:
            y_charm_predict_reg = y_charm_predict[1]
            y_charm_predict = y_charm_predict[0]
        if splitGluon:
            y_gluon_predict_reg = y_gluon_predict[1]
            y_gluon_predict = y_gluon_predict[0]

    X = np.linspace(0.0, 1.0, 100)
    histo = plt.hist(y_b_predict[:,0], bins=X, label='b' ,histtype='step', density = True)
    if splitTau:
        histo = plt.hist(y_tau_predict[:,0], bins=X, label='Tau' ,histtype='step', density = True)
    histo = plt.hist(y_bkg_predict[:,0], bins=X, label='uds' ,histtype='step', density = True)
    if splitGluon:
        histo = plt.hist(y_gluon_predict[:,0], bins=X, label='Gluon' ,histtype='step', density = True)
    if splitCharm:
        histo = plt.hist(y_charm_predict[:,0], bins=X, label='Charm' ,histtype='step', density = True)
    plt.xlabel('uds score')
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data=False, com = 14)
    plt.savefig(outFolder+"/score_bkg_"+inputSetTag+".png")
    plt.savefig(outFolder+"/score_bkg_"+inputSetTag+".pdf")
    plt.cla()

    X = np.linspace(0.0, 1.0, 100)
    histo = plt.hist(y_b_predict[:,1], bins=X, label='b ' ,histtype='step', density = True)
    if splitTau:
        histo = plt.hist(y_tau_predict[:,1], bins=X, label='Tau' ,histtype='step', density = True)
    histo = plt.hist(y_bkg_predict[:,1], bins=X, label='uds' ,histtype='step', density = True)
    if splitGluon:
        histo = plt.hist(y_gluon_predict[:,1], bins=X, label='Gluon' ,histtype='step', density = True)
    if splitCharm:
        histo = plt.hist(y_charm_predict[:,1], bins=X, label='Charm' ,histtype='step', density = True)
    plt.xlabel('b score')
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data=False, com = 14)
    plt.savefig(outFolder+"/score_b_"+inputSetTag+".png")
    plt.savefig(outFolder+"/score_b_"+inputSetTag+".pdf")
    plt.cla()

    if splitTau:
        X = np.linspace(0.0, 1.0, 100)
        histo = plt.hist(y_b_predict[:,2], bins=X, label='b' ,histtype='step', density = True)
        if splitTau:
            histo = plt.hist(y_tau_predict[:,2], bins=X, label='Tau' ,histtype='step', density = True)
        histo = plt.hist(y_bkg_predict[:,2], bins=X, label='uds' ,histtype='step', density = True)
        if splitGluon:
            histo = plt.hist(y_gluon_predict[:,2], bins=X, label='Gluon' ,histtype='step', density = True)
        if splitCharm:
            histo = plt.hist(y_charm_predict[:,2], bins=X, label='Charm' ,histtype='step', density = True)
        plt.xlabel('tau score')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/score_tau_"+inputSetTag+".png")
        plt.savefig(outFolder+"/score_tau_"+inputSetTag+".pdf")
        plt.cla()

    if splitGluon:
        X = np.linspace(0.0, 1.0, 100)
        histo = plt.hist(y_b_predict[:,3], bins=X, label='b' ,histtype='step', density = True)
        if splitTau:
            histo = plt.hist(y_tau_predict[:,3], bins=X, label='Tau' ,histtype='step', density = True)
        histo = plt.hist(y_bkg_predict[:,3], bins=X, label='uds' ,histtype='step', density = True)
        if splitGluon:
            histo = plt.hist(y_gluon_predict[:,3], bins=X, label='Gluon' ,histtype='step', density = True)
        if splitCharm:
            histo = plt.hist(y_charm_predict[:,3], bins=X, label='Charm' ,histtype='step', density = True)
        #plt.semilogy()
        plt.xlabel('gluon score')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/score_gluon_"+inputSetTag+".png")
        plt.savefig(outFolder+"/score_gluon_"+inputSetTag+".pdf")
        plt.cla()

    if splitCharm:
        X = np.linspace(0.0, 1.0, 100)
        histo = plt.hist(y_b_predict[:,4], bins=X, label='b' ,histtype='step', density = True)
        if splitTau:
            histo = plt.hist(y_tau_predict[:,4], bins=X, label='Tau' ,histtype='step', density = True)
        histo = plt.hist(y_bkg_predict[:,4], bins=X, label='uds' ,histtype='step', density = True)
        if splitGluon:
            histo = plt.hist(y_gluon_predict[:,4], bins=X, label='Gluon' ,histtype='step', density = True)
        if splitGluon:
            histo = plt.hist(y_charm_predict[:,4], bins=X, label='Charm' ,histtype='step', density = True)
        #plt.semilogy()
        plt.xlabel('charm score')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/score_charm_"+inputSetTag+".png")
        plt.savefig(outFolder+"/score_charm_"+inputSetTag+".pdf")
        plt.cla()

    if nnConfig["regression"]:
        # a quick response plot before and after...
        X_test_global["jet_pt_reg"] = Y_predict_reg[:,0]
        X_test_global["jet_pt_cor_reg"] = X_test_global["jet_pt"] * X_test_global["jet_pt_reg"]
        mean_uncor = np.mean(np.array(X_test_global["jet_pt"] / X_test_global["jet_genmatch_pt"]))
        std_uncor = rms(X_test_global["jet_pt"] / X_test_global["jet_genmatch_pt"])
        mean_cor = np.mean(X_test_global["jet_pt_corr"] / X_test_global["jet_genmatch_pt"])
        std_cor = rms(X_test_global["jet_pt_corr"] / X_test_global["jet_genmatch_pt"])
        mean_reg = np.mean(X_test_global["jet_pt_cor_reg"] / X_test_global["jet_genmatch_pt"])
        std_reg = rms(X_test_global["jet_pt_cor_reg"] / X_test_global["jet_genmatch_pt"])
        print("uncor", mean_uncor, std_uncor)
        print("cor", mean_cor, std_cor)
        print("reg", mean_reg, std_reg)
        X = np.linspace(0.0, 2.0, 100)
        histo = plt.hist(X_test_global["jet_pt"] / X_test_global["jet_genmatch_pt"], bins=X, label='Uncorrected' ,histtype='step', density=True)
        histo = plt.hist(X_test_global["jet_pt_corr"] / X_test_global["jet_genmatch_pt"], bins=X, label='JEC LOT' ,histtype='step', density=True)
        histo = plt.hist(X_test_global["jet_pt_cor_reg"] / X_test_global["jet_genmatch_pt"], bins=X, label='Regression' ,histtype='step', density=True)
        plt.xlabel('Jet response (reco/gen)')
        plt.ylabel('Jets')
        plt.xlim(0.,2.)
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        plt.text(0.7, 0.7, "mean: "+str(np.round(mean_uncor,3))+" rms:"+str(np.round(std_uncor,3)), color = '#1f77b4')
        plt.text(0.7, 0.8, "mean: "+str(np.round(mean_cor,3))+" rms:"+str(np.round(std_cor,3)), color = '#ff7f0e')
        plt.text(0.7, 0.9, "mean: "+str(np.round(mean_reg,3))+" rms:"+str(np.round(std_reg,3)), color = '#2ca02c')
        hep.cms.label("Private Work", data=False, com = 14)
        plt.savefig(outFolder+"/response_"+inputSetTag+".png")
        plt.savefig(outFolder+"/response_"+inputSetTag+".pdf")
        plt.cla()



    # class_names = labels
    # # max_display = X_test.shape[1]
    # max_display = len(labels)
    # print (X_test.shape)
    # print (X_test[:1000].shape)
    # print (X_test[:1000, 0, :].shape)
    # print (class_names)
    # print (feature_names)
    # for explainer, name  in [(shap.GradientExplainer(model, X_test[:10]), "GradientExplainer"), ]:
    #     shap.initjs()
    #     print("... {0}: explainer.shap_values(X)".format(name))
    #     shap_values = explainer.shap_values(X_test[:10])
    #     print (len(shap_values))
    #     print (len(shap_values[0]))
    #     print (len(shap_values[0][0]))
    #     print (len(shap_values[0][0][0]))
    #     print("... shap.summary_plot")
    #     plt.clf()
    #     # shap.summary_plot(shap_values[:, :, 0], X_test[:1000,0,:], plot_type="bar",
    #     # shap.summary_plot(shap_values[0][:, 0, :], X_test[:100][:, 0, :], plot_type="bar",
    #     pdb.set_trace()
    #     shap.summary_plot(shap_values[:, :, 0, :], X_test[:10][:, 0, :], plot_type="bar",
    #         feature_names = feature_names,
    #         class_names = class_names, show=False,
    #         # max_display = max_display,
    #         # )
    #         # max_display = max_display, plot_size=[15.0, 0.4 * max_display + 1.5])
    #         plot_size=[15.0, 0.4 * 10. + 1.5])
    #     plt.savefig(outFolder+"/shap_summary_"+inputSetTag+"_{0}.pdf".format(name))
    #     plt.savefig(outFolder+"/shap_summary_"+inputSetTag+"_{0}.png".format(name))


if __name__ == "__main__":
    from args import get_common_parser, handle_common_args
    # parser = argparse.ArgumentParser()
    parser = get_common_parser()
    parser.add_argument('-f','--file', help = 'input file name part')
    parser.add_argument('-c','--classes', help = 'Which flavors to run, options are b, bt, btg, btgc.')
    parser.add_argument('-i','--inputs', help = 'Which inputs to run, options are baseline, ext1, ext2, all.')
    parser.add_argument('--model', dest = 'model', default = "deepset")
    parser.add_argument('--train-batch-size', dest = 'batch_size', default = 1024)
    parser.add_argument('--train-epochs', dest = 'epochs', default = 50)
    parser.add_argument('--train-validation-split', dest = 'validation_split', default = .25)
    parser.add_argument('--learning-rate', dest = 'learning_rate', default = 0.0001)
    parser.add_argument('--optimizer', dest = 'optimizer', default = "adam")
    parser.add_argument('--classweights', dest = 'classweights', default = False, action='store_true')
    parser.add_argument('--regression', dest = 'regression', default = False, action='store_true')
    parser.add_argument('--pruning', dest = 'pruning', default = False, action='store_true')
    parser.add_argument('--inputQuant', dest = 'inputQuant', default = False, action='store_true')
    parser.add_argument('--nbits', dest = 'nbits', default = 8)
    parser.add_argument('--integ', dest = 'integ', default = 0)
    parser.add_argument('--nNodes', dest = 'nNodes', default = 15)
    parser.add_argument('--nHeads', dest = 'nHeads', default = 3)
    parser.add_argument('--nNodesHead', dest = 'nNodesHead', default = 12)
    parser.add_argument('--strstamp', dest = 'strstamp', default = "")

    args = parser.parse_args()
    handle_common_args(args)

    print('#'*30)
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('#'*30)

    allowedModels = ["DeepSet", "DeepSet-MHA", "MLP", "MLP-MHA"]
    allowedClasses = ["b", "bt", "btg", "btgc"]
    allowedFiles = ["All200", "extendedAll200", "baselineAll200", "AllHIG200", "AllQCD200", "AllTT200", "TT_PU200", "TT1L_PU200", "TT2L_PU200", "ggHtt_PU200"]
    allowedInputs = ["baseline", "ext1", 'ext2', "all"]
    allowedOptimizer = ["adam", "ranger"]

    if args.model not in allowedModels:
        raise ValueError("args.model not in allowed models! Options are", allowedModels)
    if args.classes not in allowedClasses:
        raise ValueError("args.classes not in allowed classes! Options are", allowedClasses)
    if args.file not in allowedFiles:
        raise ValueError("args.file not in allowed file! Options are", allowedFiles)
    if args.inputs not in allowedInputs:
        raise ValueError("args.inputs not in allowed inputs! Options are", allowedInputs)
    if args.optimizer not in allowedOptimizer:
        raise ValueError("args.optimizer not in allowed optimizer! Options are", allowedOptimizer)

    nnConfig = {
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "validation_split": float(args.validation_split),
        "model": args.model,
        "learning_rate": float(args.learning_rate),
        "optimizer": args.optimizer,
        "classweights": args.classweights,
        "regression": args.regression,
        "pruning": args.pruning,
        "inputQuant": args.inputQuant,
        "nbits": int(args.nbits),
        "integ": int(args.integ),
        "nNodes": int(args.nNodes),
        "nHeads": int(args.nHeads),
        "nNodesHead": int(args.nNodesHead),
    }

    doTraining(
        args.file,
        args.classes,
        args.inputs,
        nnConfig,
        args.save,
        args.strstamp,
        args.workdir,
        )