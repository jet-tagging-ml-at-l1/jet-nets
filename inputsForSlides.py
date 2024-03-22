











# All field stored
pfcand_fields_all = [
    'pt_rel','deta','dphi','charge',
    'id',"track_vx","track_vy","track_vz"
    'puppiweight','dxy_custom','etarel','pperp_ratio','ppara_ratio',
    'pt_rel_log','z0','dxy','track_chi2',
    'track_chi2norm','track_qual','track_npar','track_nstubs','track_pterror',
    'cluster_hovere','cluster_sigmarr','cluster_abszbarycenter','cluster_emet',
    # 'cluster_egvspion','cluster_egvspu'
    # 'pt','pt_log','px','py','pz','eta','phi','mass','energy','energy_log','pt_log',
]

# A reduced set - what the baseline model uses
pfcand_fields_baseline = [
    'pt_rel','deta','dphi','charge',
    'id',"track_vx","track_vy","track_vz"
]

# A set w/ more features inspired by DeepJet
pfcand_fields_ext1 = [
    'pt_rel','deta','dphi','charge',
    'id',"track_vx","track_vy","track_vz",
    'puppiweight','dxy_custom','etarel','pperp_ratio','ppara_ratio'
]