from imports import *
from dataset import *
import argparse
gc.set_threshold(0)

def createDataset(filetag, flavs, not_reduce):

    # Open the input ROOT files and check its contents
    # filetag = "TT_PU200"
    # filetag = "All200"
    # filetag = "AllTT200"
    # fname = "/eos/cms/store/cmst3/user/sewuchte/l1teg/fp_ntuples_11_01_24/"+filetag+".root"
    fname = "../nTuples/"+filetag+".root"
    # flavs = "b"
    splitTau = "t" in flavs
    splitGluon = "g" in flavs
    splitCharm = "c" in flavs

    outFolder = "datasets/"
    if not_reduce:
        outFolder = "datasets_notreduced"
    outFolder = outFolder + "/" + filetag + "/" + flavs +"/"

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    print ("Use the following outfolder", outFolder)

    # Transform into Awkward arrays and filter its contents
    filter = "/(jet)_(eta|phi|pt|pt_log|pt_raw|mass|energy|px|py|pz|bjetscore|tauscore|pt_corr|tauflav|muflav|elflav|taudecaymode|lepflav|taucharge|genmatch_pt|genmatch_eta|genmatch_phi|genmatch_mass|genmatch_hflav|genmatch_pflav|npfcand|pfcand_pt|pfcand_pt_rel|pfcand_pt_rel_log|pfcand_px|pfcand_py|pfcand_pz|pfcand_pt_log|pfcand_eta|pfcand_phi|pfcand_mass|pfcand_energy|pfcand_energy_log|pfcand_puppiweight|pfcand_z0|pfcand_dxy|pfcand_dxy_custom|pfcand_id|pfcand_charge|pfcand_pperp_ratio|pfcand_ppara_ratio|pfcand_deta|pfcand_dphi|pfcand_etarel|pfcand_track_chi2|pfcand_track_chi2norm|pfcand_track_qual|pfcand_track_npar|pfcand_track_nstubs|pfcand_track_vx|pfcand_track_vy|pfcand_track_vz|pfcand_track_pterror|pfcand_cluster_hovere|pfcand_cluster_sigmarr|pfcand_cluster_abszbarycenter|pfcand_cluster_emet|pfcand_cluster_egvspion|pfcand_cluster_egvspu)/"
    #filter ="jet_*"

    nconstit = 16

    data = readDataFromFile(fname, filter = filter, applyBaseCut = True)

    addResponseVars(data)

    # data = ak.concatenate([data1,data2])
    print("Data Fields:", data.fields)
    print("JET PFcand Fields:", data.jet_pfcand.fields)

    data_split = splitFlavors(data, splitTau = splitTau, splitGluon = splitGluon, splitCharm = splitCharm)
    data_b = data_split["b"]
    data_tau = data_split["tau"]
    data_gluon = data_split["gluon"]
    data_charm = data_split["charm"]
    data_bkg = data_split["bkg"]

    # Validation: check that signal only contains jets with hadron flavour 5

    jethflav = ak.to_numpy(data_b['jet_genmatch_hflav'])
    flat_jethflav=jethflav.flatten()
    plt.hist(flat_jethflav, bins=np.arange(0, 10, 1), edgecolor='black', density = True)
    # hep.histplot(flat_jethflav, yerr=True)
    plt.xlabel('Jet genmatch hadron flavor')
    plt.ylabel('Jets')
    hep.cms.label("Private Work", data = False, com = 14)
    plt.savefig(outFolder+"/signal_b_hflav.pdf")
    plt.savefig(outFolder+"/signal_b_hflav.png")
    plt.cla()

    if splitTau:
        jettauflav = ak.to_numpy(data_tau['jet_tauflav'])
        flat_jettauflav=jettauflav.flatten()
        plt.hist(flat_jettauflav, bins=np.arange(0, 10, 1), edgecolor='black', density = True)
        plt.xlabel('Jet tau flavor')
        plt.ylabel('Jets')
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/signal_tau_tauflav.pdf")
        plt.savefig(outFolder+"/signal_tau_tauflav.png")
        plt.cla()

    if splitGluon:
        jetgflav = ak.to_numpy(data_gluon['jet_genmatch_pflav'])
        flat_jetgflav=jetgflav.flatten()
        plt.hist(flat_jetgflav, bins=np.arange(0, 25, 1), edgecolor='black', density = True)
        plt.xlabel('Jet genmatch parton flavor')
        plt.ylabel('Jets')
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/signal_gluon_pflav.pdf")
        plt.savefig(outFolder+"/signal_gluon_pflav.png")
        plt.cla()

        jetgflav = ak.to_numpy(data_gluon['jet_genmatch_hflav'])
        flat_jetgflav=jetgflav.flatten()
        plt.hist(flat_jetgflav, bins=np.arange(0, 25, 1), edgecolor='black', density = True)
        plt.xlabel('Jet genmatch parton flavor')
        plt.ylabel('Jets')
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/signal_gluon_hflav.pdf")
        plt.savefig(outFolder+"/signal_gluon_hflav.png")
        plt.cla()

    if splitCharm:
        jetgflav = ak.to_numpy(data_charm['jet_genmatch_pflav'])
        flat_jetgflav=jetgflav.flatten()
        plt.hist(flat_jetgflav, bins=np.arange(0, 25, 1), edgecolor='black', density = True)
        plt.xlabel('Jet genmatch parton flavor')
        plt.ylabel('Jets')
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/signal_charm_pflav.pdf")
        plt.savefig(outFolder+"/signal_charm_pflav.png")
        plt.cla()

        jetgflav = ak.to_numpy(data_charm['jet_genmatch_hflav'])
        flat_jetgflav=jetgflav.flatten()
        plt.hist(flat_jetgflav, bins=np.arange(0, 25, 1), edgecolor='black', density = True)
        plt.xlabel('Jet genmatch parton flavor')
        plt.ylabel('Jets')
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/signal_charm_hflav.pdf")
        plt.savefig(outFolder+"/signal_charm_hflav.png")
        plt.cla()

    if not not_reduce:
        reduceDatasetToMin(data_split)
    
    data_b = data_split["b"]
    if splitTau:
        data_tau = data_split["tau"]
    if splitGluon:
        data_gluon = data_split["gluon"]
    if splitCharm:
        data_charm = data_split["charm"]
    data_bkg = data_split["bkg"]

    datas = [data_bkg, data_b]
    labels = ["Bkg", "b"]
    if splitTau:
        labels.append("Tau")
        datas.append(data_tau)
    if splitGluon:
        labels.append("Gluon")
        datas.append(data_gluon)
    if splitCharm:
        labels.append("Charm")
        datas.append(data_charm)

    gc.collect()

    # make some plots
    for obj in ['jet_pfcand']:
        for label, data in zip(labels, datas):
            num = ak.num(data[obj])
            plt.hist(num, label = label, bins = np.arange(0, 30, 1), density = True, 
                    #log = True, 
                    histtype = "step")
            
        plt.xlabel(f"N of {obj}")
        plt.legend()
        # plt.grid()
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/"+obj+".pdf")
        plt.savefig(outFolder+"/"+obj+".png")
        plt.cla()

    # for obj in data.fields:
    for obj in ['jet_eta', 'jet_phi', 'jet_pt', 'jet_pt_raw', 'jet_mass', 'jet_energy', 'jet_px', 'jet_py', 'jet_pz',
                'jet_bjetscore', 'jet_tauscore', 'jet_tauflav', 'jet_muflav', 'jet_elflav', 'jet_taudecaymode', 'jet_lepflav',
                'jet_taucharge', 'jet_genmatch_pt', 'jet_genmatch_eta', 'jet_genmatch_phi', 'jet_genmatch_mass', 'jet_genmatch_hflav', 'jet_genmatch_pflav',
                'jet_pt_corr', 'jet_npfcand',
                'jet_ptUncorr_div_ptGen', 'jet_ptCorr_div_ptGen', 'jet_ptRaw_div_ptGen',
                ]:
        min_ = min(min(ak.flatten(data_b[obj],axis=-1)), min(ak.flatten(data_bkg[obj],axis=-1)))
        max_ = max(max(ak.flatten(data_b[obj],axis=-1)), max(ak.flatten(data_bkg[obj],axis=-1)))
        if splitTau:
            min_ = min(min_, min(ak.flatten(data_tau[obj],axis=-1)))
            max_ = max(max_, max(ak.flatten(data_tau[obj],axis=-1)))
        if splitGluon:
            min_ = min(min_, min(ak.flatten(data_gluon[obj],axis=-1)))
            max_ = max(max_, max(ak.flatten(data_gluon[obj],axis=-1)))
        if splitCharm:
            min_ = min(min_, min(ak.flatten(data_charm[obj],axis=-1)))
            max_ = max(max_, max(ak.flatten(data_charm[obj],axis=-1)))
        range = (min_, max_)
        if 'div_ptGen' in obj:
            range = (0, 5)
        for label, data in zip(labels, datas):
            # notice the [:,:1] below -> we slice the array and select no more than the first entry per event
            # ak.ravel makes the array flat such that we can fill a histogram
            # plt.hist(ak.ravel(data[obj].puppiweight[:,:1]), label = label, bins = 50, density = True, log = True, histtype = "step")
            plt.hist(ak.flatten(data[obj],axis=-1), label = label, bins = 100, density = True, log = True, histtype = "step", range = range)
            
        if 'div_ptGen' in obj:
            plt.xlim(0.,5.)
        plt.xlabel(f"{obj}")
        plt.legend()
        # plt.grid()
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/"+obj+".pdf")
        plt.savefig(outFolder+"/"+obj+".png")
        plt.cla()


    del data_b, data_bkg, datas
    if splitTau:
        del data_tau
    if splitGluon:
        del data_gluon
    if splitCharm:
        del data_charm

    gc.collect()

    import boost_histogram as bh
    hist = bh.Histogram(
        bh.axis.Regular(100, 0.0, 150.0),
        bh.axis.Regular(100, 0.0, 5.0)
    )
    hist.fill(data['jet_genmatch_pt'], data['jet_ptUncorr_div_ptGen'])
    hep.hist2dplot(hist, labels = False, cbar = False)
    hep.cms.label("Private Work", data = False, com = 14)
    plt.savefig(outFolder+"/"+"responseUncorr"+".pdf")
    plt.savefig(outFolder+"/"+"responseUncorr"+".png")
    plt.cla()

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


    # Create Datasets for plotting
    classes_plotting, var_names_all, x_all, y_all, x_global = createAndSaveTrainingData(data_split, pfcand_fields_all)
    x_b_all = classes_plotting["b"]["x"]
    if splitTau:
        x_tau_all = classes_plotting["tau"]["x"]
    if splitGluon:
        x_gluon_all = classes_plotting["gluon"]["x"]
    if splitCharm:
        x_charm_all = classes_plotting["charm"]["x"]
    x_bkg_all = classes_plotting["bkg"]["x"]

    # Plot Input Data distributions
    for i, name in enumerate(var_names_all[:100]):
        if "_0_" not in name: continue
        _ = plt.hist(x_bkg_all[:,i], bins = 100, log = True, density = True, label = "Bkg")
        _ = plt.hist(x_b_all[:,i], bins = _[1], histtype = "step", density = True, label = "b")
        if splitGluon:
            _ = plt.hist(x_gluon_all[:,i], bins = _[1], histtype = "step", density = True, label = "Gluon")
        if splitTau:
            _ = plt.hist(x_tau_all[:,i], bins = _[1], histtype = "step", density = True, label = "Tau")
        if splitCharm:
            _ = plt.hist(x_charm_all[:,i], bins = _[1], histtype = "step", density = True, label = "Charm")
        
        plt.xlabel(name)
        plt.ylabel("PF Candidates / bin (normalized to 1)")
        plt.legend()
        hep.cms.label("Private Work", data = False, com = 14)
        plt.savefig(outFolder+"/"+name+".pdf")
        plt.savefig(outFolder+"/"+name+".png")
        plt.cla()

    del classes_plotting, var_names_all, x_all, y_all
    gc.collect()

    # Create datasets for training
    classes_baseline, var_names_baseline, x_baseline, y_baseline, x_global_baseline = createAndSaveTrainingData(data_split, pfcand_fields_baseline)
    classes_ext1, var_names_ext1, x_ext1, y_ext1, x_global_ext1 = createAndSaveTrainingData(data_split, pfcand_fields_ext1)
    classes_all, var_names_all, x_all, y_all, x_global_all = createAndSaveTrainingData(data_split, pfcand_fields_all)

    del data_split
    gc.collect()

    # print(x_all.shape)
    # print(y_all.shape)
    # print(len(x_global_all))

    X_train_baseline, X_test_baseline, Y_train_baseline, Y_test_baseline, x_global_train_baseline, x_global_test_baseline = splitAndShuffle(x_baseline, y_baseline, x_global_baseline, len(pfcand_fields_baseline))
    X_train_ext1, X_test_ext1, Y_train_ext1, Y_test_ext1, x_global_train_ext1, x_global_test_ext1  = splitAndShuffle(x_ext1, y_ext1, x_global_ext1, len(pfcand_fields_ext1))
    X_train_all, X_test_all, Y_train_all, Y_test_all, x_global_train_all, x_global_test_all  = splitAndShuffle(x_all, y_all, x_global_all, len(pfcand_fields_all))

    # -----------------------------------------
    # Save baseline
    x_b_baseline = np.reshape(classes_baseline["b"]["x"],[-1, nconstit, len(pfcand_fields_baseline)])
    if splitTau:
        x_tau_baseline = np.reshape(classes_baseline["tau"]["x"],[-1, nconstit, len(pfcand_fields_baseline)])
    if splitGluon:
        x_gluon_baseline = np.reshape(classes_baseline["gluon"]["x"],[-1, nconstit, len(pfcand_fields_baseline)])
    if splitCharm:
        x_charm_baseline = np.reshape(classes_baseline["charm"]["x"],[-1, nconstit, len(pfcand_fields_baseline)])
    x_bkg_baseline = np.reshape(classes_baseline["bkg"]["x"],[-1, nconstit, len(pfcand_fields_baseline)])

    # Save Data to Parket files
    ak.to_parquet(X_train_baseline, outFolder+"/X_baseline_train.parquet")
    ak.to_parquet(Y_train_baseline, outFolder+"/Y_baseline_train.parquet")
    ak.to_parquet(X_test_baseline, outFolder+"/X_baseline_test.parquet")
    ak.to_parquet(Y_test_baseline, outFolder+"/Y_baseline_test.parquet")
    ak.to_parquet(x_global_train_baseline, outFolder+"/X_global_baseline_train.parquet")
    ak.to_parquet(x_global_test_baseline, outFolder+"/X_global_baseline_test.parquet")
    ak.to_parquet(x_b_baseline, outFolder+"/X_baseline_b.parquet")
    if splitTau:
        ak.to_parquet(x_tau_baseline, outFolder+"/X_baseline_tau.parquet")
    if splitGluon:
        ak.to_parquet(x_gluon_baseline, outFolder+"/X_baseline_gluon.parquet")
    if splitCharm:
        ak.to_parquet(x_charm_baseline, outFolder+"/X_baseline_charm.parquet")
    ak.to_parquet(x_bkg_baseline, outFolder+"/X_baseline_bkg.parquet")

    # Save data in npy for HLS4ML
    np.save(outFolder+"/X_test_baseline_btag_nconst_{}".format(nconstit), X_test_baseline)
    np.save(outFolder+"/Y_test_baseline_btag_nconst_{}".format(nconstit), Y_test_baseline)
    # -----------------------------------------

    del X_train_baseline, Y_train_baseline, X_test_baseline, Y_test_baseline, x_b_baseline, x_tau_baseline, x_gluon_baseline, x_bkg_baseline, x_charm_baseline
    gc.collect()

    # Save ext1
    x_b_ext1 = np.reshape(classes_ext1["b"]["x"],[-1, nconstit, len(pfcand_fields_ext1)])
    if splitTau:
        x_tau_ext1 = np.reshape(classes_ext1["tau"]["x"],[-1, nconstit, len(pfcand_fields_ext1)])
    if splitGluon:
        x_gluon_ext1 = np.reshape(classes_ext1["gluon"]["x"],[-1, nconstit, len(pfcand_fields_ext1)])
    if splitCharm:
        x_charm_ext1 = np.reshape(classes_ext1["charm"]["x"],[-1, nconstit, len(pfcand_fields_ext1)])
    x_bkg_ext1 = np.reshape(classes_ext1["bkg"]["x"],[-1, nconstit, len(pfcand_fields_ext1)])

    # Save Data to Parket files
    ak.to_parquet(X_train_ext1, outFolder+"/X_ext1_train.parquet")
    ak.to_parquet(Y_train_ext1, outFolder+"/Y_ext1_train.parquet")
    ak.to_parquet(X_test_ext1, outFolder+"/X_ext1_test.parquet")
    ak.to_parquet(Y_test_ext1, outFolder+"/Y_ext1_test.parquet")
    ak.to_parquet(x_global_train_ext1, outFolder+"/X_global_ext1_train.parquet")
    ak.to_parquet(x_global_test_ext1, outFolder+"/X_global_ext1_test.parquet")
    ak.to_parquet(x_b_ext1, outFolder+"/X_ext1_b.parquet")
    if splitTau:
        ak.to_parquet(x_tau_ext1, outFolder+"/X_ext1_tau.parquet")
    if splitGluon:
        ak.to_parquet(x_gluon_ext1, outFolder+"/X_ext1_gluon.parquet")
    if splitCharm:
        ak.to_parquet(x_charm_ext1, outFolder+"/X_ext1_charm.parquet")
    ak.to_parquet(x_bkg_ext1, outFolder+"/X_ext1_bkg.parquet")

    # Save data in npy for HLS4ML
    np.save(outFolder+"/X_test_ext1_btag_nconst_{}".format(nconstit), X_test_ext1)
    np.save(outFolder+"/Y_test_ext1_btag_nconst_{}".format(nconstit), Y_test_ext1)
    # -----------------------------------------

    del X_train_ext1, Y_train_ext1, X_test_ext1, Y_test_ext1, x_b_ext1, x_tau_ext1, x_gluon_ext1, x_bkg_ext1, x_charm_ext1
    gc.collect()

    # Save all
    x_b_all = np.reshape(classes_all["b"]["x"],[-1, nconstit, len(pfcand_fields_all)])
    if splitTau:
        x_tau_all = np.reshape(classes_all["tau"]["x"],[-1, nconstit, len(pfcand_fields_all)])
    if splitGluon:
        x_gluon_all = np.reshape(classes_all["gluon"]["x"],[-1, nconstit, len(pfcand_fields_all)])
    if splitCharm:
        x_charm_all = np.reshape(classes_all["charm"]["x"],[-1, nconstit, len(pfcand_fields_all)])
    x_bkg_all = np.reshape(classes_all["bkg"]["x"],[-1, nconstit, len(pfcand_fields_all)])

    # Save Data to Parket files
    ak.to_parquet(X_train_all, outFolder+"/X_all_train.parquet")
    ak.to_parquet(Y_train_all, outFolder+"/Y_all_train.parquet")
    ak.to_parquet(X_test_all, outFolder+"/X_all_test.parquet")
    ak.to_parquet(Y_test_all, outFolder+"/Y_all_test.parquet")
    ak.to_parquet(x_global_train_all, outFolder+"/X_global_all_train.parquet")
    ak.to_parquet(x_global_test_all, outFolder+"/X_global_all_test.parquet")
    ak.to_parquet(x_b_all, outFolder+"/X_all_b.parquet")
    if splitTau:
        ak.to_parquet(x_tau_all, outFolder+"/X_all_tau.parquet")
    if splitGluon:
        ak.to_parquet(x_gluon_all, outFolder+"/X_all_gluon.parquet")
    if splitCharm:
        ak.to_parquet(x_charm_all, outFolder+"/X_all_charm.parquet")
    ak.to_parquet(x_bkg_all, outFolder+"/X_all_bkg.parquet")

    # Save data in npy for HLS4ML
    np.save(outFolder+"/X_test_all_btag_nconst_{}".format(nconstit), X_test_all)
    np.save(outFolder+"/Y_test_all_btag_nconst_{}".format(nconstit), Y_test_all)

    del X_train_all, Y_train_all, X_test_all, Y_test_all, x_b_all, x_tau_all, x_gluon_all, x_bkg_all, x_charm_all
    gc.collect()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', help='input file name part')
    parser.add_argument('-c','--classes', help='Which flavors to run, options are b, bt, btg.')
    parser.add_argument('--not-reduce', dest = 'not_reduce', default = False)
    args = parser.parse_args()

    print('#'*30)
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('#'*30)

    createDataset(args.file, args.classes, args.not_reduce)
