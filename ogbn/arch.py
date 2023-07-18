archs = {
    "ogb_e50_06": [['PAPF'], ['PFP', 'PPP']],
    "ogb_e50": [['PF', 'PAPF', 'PFPP', 'PPPPF'], ['PFP', 'PPP']],  # 0.4
    "ogb_e50_02": [['PA', 'PF', 'PAPF', 'PFPP', 'PPAP', 'PPPF', 'PAPAP', 'PAPFP', 'PAPPF', 'PFPAP', 'PFPPF', 'PPFPF', 'PPFPP', 'PPPAI', 'PPPFP', 'PPPPF', 'PPPPP'],
    ['PP', 'PFP', 'PPP']], # 0.2
    "ogb_e50_nodrop": [['PF', 'PP', 'PAI', 'PAP', 'PAPF', 'PAPP', 'PFPF', 'PPAP', 'PPPF', 'PPPP'], ['PFP', 'PPP']],   # hop=3 ns=8 ratio=0.2
    "ogb_e50_nodrop1": [['PF', 'PAI', 'PAPP', 'PAPAI'], ['PP', 'PAP', 'PFP', 'PPP']], # hop=4 ns=8 ratio=0.2
    "ogb_e50_nodrop2": [['PF', 'PP', 'PPF', 'PPP', 'PAPF', 'PAPP', 'PFPF', 'PFPP', 'PPAP', 'PPFP', 'PPPF', 'PPPP'], ['P', 'PP', 'PAP', 'PFP', 'PPP']], # hop=3 ns=8 ratio=0.2  no label
    "ogb_e50_nodrop3": [['PAI', 'PPA', 'PPF', 'PFPP', 'PAPPA', 'PAPPP', 'PFPAI', 'PFPPP'], ['P', 'PP', 'PAP', 'PFP', 'PPP']],
    "ogb_e50_nodrop4": [['PF', 'PAPF', 'PAPP', 'PFPF', 'PPPF', 'PPPP', 'PAPPF', 'PPAPF', 'PPPAP', 'PPPFP', 'PPPPF', 'PPPPP'], []], # hop=4  ns = 16  Val 55.4232, Test 53.9617
    "ogb_e100_nodrop": [['PF', 'PAPF', 'PAPP', 'PFPF', 'PPPF', 'PPPP', 'PPPFP', 'PPPPF', 'PPPPP'], []], # hop=4  ns = 16  0.4
    "ogb_e100_nodrop": [['PF', 'PPF', 'PAPF', 'PAPP', 'PFPF', 'PPPF', 'PPPP', 'PAPPF', 'PPAPF', 'PPPAP', 'PPPFP', 'PPPPF', 'PPPPP'], []], # hop=4  ns = 16  0.3  Val 55.8301, Test 54.4147
    "ogb_h3_0": [['PA', 'PF', 'PP', 'PAI', 'PAP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPA', 'PPAP', 'PPFP', 'PPPA', 'PPPF', 'PPPP'], ['PAP', 'PFP']],
    "ogb_h3_1": [['P', 'PF', 'PP', 'PAI', 'PAP', 'PPF', 'PPP', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPA', 'PFPF', 'PPAP', 'PPFP', 'PPPF', 'PPPP'], ['PAP', 'PFP', 'PPP']],
    "ogb_h4_0": [['PF', 'PAI', 'PAP', 'PPA', 'PPF', 'PAPF', 'PAPP', 'PFPF', 'PPPF', 'PAIAP', 'PAPAI', 'PAPAP', 'PPAPP', 'PPFPP', 'PPPAP', 'PPPFP', 'PPPPF', 'PPPPP'], ['PFP', 'PPP']],
    "ogb_h4_1":[['P', 'PF', 'PPF', 'PAPF', 'PAPP', 'PFPF', 'PPPF', 'PAIAI', 'PAIAP', 'PAPAI', 'PAPAP', 'PAPFP', 'PAPPP', 'PPAPP', 'PPFPP', 'PPPFP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h3_2": [['P', 'PF', 'PP', 'PAI', 'PAP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPF', 'PAPP', 'PFPF', 'PPAP', 'PPFP', 'PPPF', 'PPPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogb_h1": [['P', 'PA', 'PF', 'PP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h2": [['P', 'PA', 'PF', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h3": [['P', 'PF', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP', 'PAPF', 'PAPP', 'PFPF', 'PPAP', 'PPFP', 'PPPF', 'PPPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h4": [['PF', 'PP', 'PAI', 'PAP', 'PPF', 'PAPF', 'PAPP', 'PPAP', 'PPPF', 'PAIAP', 'PAPAI', 'PAPAP', 'PAPFP', 'PAPPP', 'PPAPP', 'PPFPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h5": [['PF', 'PP', 'PAI', 'PPF', 'PAPF', 'PAPP', 'PPAP', 'PAIAI', 'PAIAP', 'PAPAI', 'PAIAPP', 'PAPAPP', 'PAPPAP', 'PAPPFP', 'PPAPAP', 'PPPPAP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h6_0": [['PPA', 'PFPP', 'PAPPP', 'PAPAPF', 'PAPPPF', 'PPPAPP', 'PAIAIAP', 'PAIAPAP', 'PAPAPAI', 'PFPAPAP', 'PPAPPPP', 'PPFPPAP', 'PPFPPFP', 'PPPAPAI', 'PPPAPPF', 'PPPPAPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    
    "ogb_h6_top20": [['PF', 'PAP', 'PAIAI', 'PAPAI', 'PAPAP', 'PAPFP', 'PAIAPP', 'PAIAPAI', 'PAIAPAP', 'PAIAPPP', 'PAPAIAI', 'PPAPPFP', 'PPPAIAP', 'PPPAPPF', 'PPPPAPP', 'PPPPPFP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h6_all": [['PAIAPAP', 'PF', 'PAIAPP', 'PAPAIAI', 'PPAPPFP', 'PPPAIAP', 'PAIAPPP', 'PAP', 'PAIAPAI', 'PAPFP', 'PAIAI', 'PAPAI', 'PPPPPFP', 'PAPAP', 'PPPPAPP', 'PPPAPPF', 'PPAPP', 'PPAP', 'PFP', 'PPAPPPF', 'PAI', 'PAPAPAI', 'PAPPFPP', 'PPPPPAP', 'PPPPFP', 'PAPPPP', 'PP', 'PAPAPP', 'PPAIAPP', 'PFPP', 'PFPPAP', 'PAPPP', 'PAPF', 'PPPP', 'PAPPFP', 'PPPAIAI', 'PPPPPPP', 'PAIAIAP', 'PAPPPPP', 'PAPPAP', 'PPP', 'PPFPPAP', 'PFPPAPP', 'PPPFPPA', 'PPFPFP', 'PAPAPPP', 'PFPPPP', 'PPPF', 'PPFPPP', 'PPAI', 'PPFPPFP', 'PAPPPFP', 'PFPAPP', 'PAPPPAP', 'PPPPPP', 'PAPFPAP', 'PAIAP', 'PPFPF', 'PPPPAPF', 'PAPPPA', 'PAIAPFP', 'PPPPAP', 'PAPPAPP', 'PPAPPP', 'PAPAIAP', 'PAPPPPF', 'PFPFPAP', 'PPAPPF', 'PPAPFP', 'PPAPPPP', 'PFPAPAP', 'PPAPAI', 'PAIAPF', 'PPAPFPP', 'PFPAPPP', 'PPFPP', 'PPFPAP', 'PFPFPPP', 'PFPAPFP', 'PFPPPAP', 'PPPAP', 'PAPAPF', 'PPFP', 'PPPAPP', 'PFPAP', 'PPAIAI', 'PAPFPP', 'PAIA', 'PAPPPAI', 'PPPFPFP', 'PPFPPPP', 'PAPPPF', 'PAPPF', 'PFPFP', 'PPAPAPF', 'PPPAPAI', 'PPPAPAP', 'PPPAI', 'PAPFPFP', 'PFPAI', 'PPPPPAI', 'PFPPFP', 'P', 'PAPFPPP', 'PPPFPPP', 'PAPPAI', 'PFPPPPP', 'PPPFPP', 'PPPAPF', 'PPAIAP', 'PFPPF', 'PPPPFPA', 'PAPA', 'PPAPAP', 'PPPFPF', 'PPPPPF', 'PFPPP', 'PPPPFPP', 'PPPAPFP', 'PAPP', 'PFPFPFP', 'PFPF', 'PA', 'PPAPPAP', 'PPAIAPF', 'PPPAPPP', 'PPAPAPP', 'PPFPAPF', 'PPPFP', 'PPFPA', 'PFPAIAP', 'PPPPAI', 'PPPPPA', 'PFPPPF', 'PAPPPPA', 'PPAPFPF', 'PFPPFPF', 'PFPPFPP', 'PFPFPAI', 'PAIAIAI', 'PPFPFPF', 'PAIAPA', 'PFPPPFP', 'PPPPP', 'PPFPFPP', 'PAPPAIA', 'PAPPFPA', 'PPA', 'PPFPFPA', 'PPAPPA', 'PAPPA', 'PFPPPA', 'PFPPPPA', 'PAPFPPF', 'PPAPAPA', 'PPAPF', 'PPFPPPA', 'PPFPAIA', 'PPPAPA', 'PAPAPFP', 'PPAPA', 'PPPFPAI', 'PPPAIA', 'PPPFPPF', 'PAPPFPF', 'PPFPAPP', 'PPPFPAP', 'PAPAPAP', 'PAIAPPA', 'PPPPPPA', 'PAPAPA', 'PPAPPAI', 'PFPFPPF', 'PPPPF', 'PPPPAPA', 'PPAIAPA', 'PFPPPAI', 'PPAIAIA', 'PFPAPA', 'PFPPAI', 'PAIAPPF', 'PAPFPF', 'PFPAPPA', 'PPPAPPA', 'PPPA', 'PPFPPPF', 'PPPPA', 'PPAPFPA', 'PFPPAPF', 'PFPPPPF', 'PAPAPPA', 'PAPFPPA', 'PPFPPA', 'PFPAPAI', 'PPFPAPA', 'PFPPAPA', 'PPAPAIA', 'PFPFPP', 'PPFPPAI', 'PAPAPPF', 'PFPPFPA', 'PAIAIA', 'PPPFPA', 'PFPPAIA', 'PAPPAPA', 'PPFPAI', 'PFPFPA', 'PAPAIA', 'PPPPPPF', 'PFPAIA', 'PPPPAIA', 'PPAPPPA', 'PFPFPPA', 'PAPPAPF', 'PFPAPPF', 'PAPFPAI', 'PFPFPF', 'PPAIA', 'PFPAIAI', 'PFPAPF', 'PPF', 'PPFPPF', 'PFPA', 'PAPFPA', 'PPPPFPF', 'PFPPA'], ['PP', 'PPP', 'PAP', 'PFP']],
    "ogb_h6_lin": [['PAIAPAP', 'PF', 'PAIAPP', 'PAPAIAI', 'PPAPPFP', 'PPPAIAP', 'PAIAPPP', 'PAP', 'PAIAPAI', 'PAPFP', 'PAIAI', 'PAPAI', 'PPPPPFP', 'PAPAP', 'PPPPAPP', 'PPPAPPF', 'PPAPP', 'PPAP', 'PFP', 'PPAPPPF', 'PAI', 'PAPAPAI', 'PAPPFPP', 'PPPPPAP', 'PPPPFP', 'PAPPPP', 'PP', 'PAPAPP', 'PPAIAPP', 'PFPP', 'PFPPAP', 'PAPPP', 'PAPF', 'PPPP', 'PAPPFP', 'PPPAIAI', 'PPPPPPP'], ['PP', 'PPP', 'PAP', 'PFP']],


    "ogb_h1_nsl_hidden128_top20": [['PF', 'PP', 'P', 'PA'], ['PFP', 'PP', 'PAP', 'PPP']],
    "ogb_h2_nsl_hidden128_top20": [['PF', 'P', 'PAP', 'PAI', 'PPA', 'PP', 'PA', 'PPP', 'PPF', 'PFP'], ['PAP', 'PFP', 'PPP', 'PP']],

    "ogb_h3_nsl_hidden128_top20": [['P', 'PA', 'PF', 'PP', 'PAP', 'PPF', 'PAIA', 'PAPF', 'PAPP', 'PFPA', 'PFPF', 'PFPP', 'PPAI', 'PPAP', 'PPPF', 'PPPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h4_nsl_hidden128_top20":[['PF', 'PAI', 'PAPF', 'PFPF', 'PFPP', 'PPAI', 'PPFP', 'PAIAI', 'PAIAP', 'PAPAI', 'PAPFP', 'PAPPP', 'PFPAP', 'PPFPP', 'PPPAI', 'PPPFP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h5_nsl_hidden128_top20":[['PF', 'PAPF', 'PFPF', 'PPFP', 'PPAPA', 'PPPPF', 'PAIAPF', 'PAPFPA', 'PAPFPP', 'PAPPAI', 'PAPPPP', 'PFPAPA', 'PFPFPA', 'PPFPAI', 'PPPAPA', 'PPPPPA'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h6_nsl_hidden128_top20":[['PAP', 'PFPP', 'PAPAP', 'PAPPP', 'PAPPPF', 'PPPAPP', 'PPPPFP', 'PAIAIAP', 'PAIAPAP', 'PAPAPAI', 'PAPPPPF', 'PFPAPAP', 'PFPFPPP', 'PPFPPFP', 'PPPAPPF', 'PPPPAPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogb_h1_nsl_hidden128_lin": [['PF', 'PP', 'P', 'PA'], ['PFP', 'PP', 'PAP', 'PPP']],
    "ogb_h2_nsl_hidden128_lin": [['PF', 'P', 'PAP', 'PAI', 'PPA', 'PP', 'PA', 'PPP', 'PPF', 'PFP'], ['PAP', 'PFP', 'PPP', 'PP']],
    "ogb_h3_nsl_hidden128_lin":[['PAPF', 'PPAP', 'PP', 'PF', 'PPF', 'PFPP', 'PAP', 'PFPF', 'PPPP', 'P', 'PAPP', 'PPPF', 'PPAI', 'PAIA', 'PA', 'PFPA', 'PAI', 'PPFP', 'PAPA'], ['PP', 'PPP', 'PAP', 'PFP']],
    "ogb_h4_nsl_hidden128_lin":[['PAIAP', 'PF', 'PAPAI', 'PAPF', 'PPAI', 'PPFPP', 'PPPAI', 'PAI', 'PFPP', 'PPPFP', 'PAPPP', 'PAPFP', 'PAIAI', 'PFPF', 'PFPAP', 'PPFP', 'PAPPA', 'PPAP', 'PAPP', 'PPFPF', 'PPF', 'PFP', 'PAPA', 'PPPAP', 'PPPA', 'PPPF'], ['PFP', 'PAP', 'PP', 'PPP']],
    "ogb_h5_nsl_hidden128_lin":[['PF', 'PAIAPF', 'PAPFPP', 'PFPF', 'PAPPPP', 'PPPPPA', 'PPAPA', 'PAPPAI', 'PPPAPA', 'PFPAPA', 'PAPFPA', 'PAPF', 'PPFPAI', 'PPFP', 'PPPPF', 'PFPFPA', 'PPAIA', 'PPAIAI', 'PPFPPA', 'PPPFPA', 'PAPAPA', 'PPFPP', 'PAPAI', 'PPAPAP', 'PAPPFP', 'PPFPAP', 'PPA', 'PAPPF', 'PAIAPA', 'PPF', 'PFP', 'PAPPA', 'PFPPF'], ['PFP', 'PAP', 'PP', 'PPP']],
    "ogb_h6_nsl_hidden128_lin":[['PPPAPPF', 'PAPAPAI', 'PPPPAPP', 'PAIAPAP', 'PAPPP', 'PPFPPFP', 'PFPP', 'PAPPPF', 'PFPAPAP', 'PPPPFP', 'PAP', 'PAIAIAP', 'PAPPPPF', 'PFPFPPP', 'PPPAPP', 'PAPAP', 'PPFPPP', 'PAPPFPP', 'PPFPPAP', 'PAPPPFP', 'PPPPF', 'PAIAPAI', 'PAIAPPP', 'PPAPPFP', 'PAPPAP', 'PPPPAPF', 'PPPAPAI', 'PAPAPPP', 'PAPPPPP', 'PFPFPAP', 'PAIAPP', 'PPPFPPP', 'PAPAPF', 'PPPPPPP', 'PPAIAP', 'PPAP', 'PPPAIAI', 'PP', 'PPAPPPP', 'PPPF', 'PAPPAPP'], ['PFP', 'PPP', 'PAP', 'PP']],

    "ogb_h6_nsl_hidden128_lin_seed42": [['PAIAIAP', 'PAPAP', 'PAPAPF', 'PAPPPPP', 'PAIAI', 'PFP', 'PPAPFPP', 'PAPPAP', 'PF', 'PAIAPP', 'PAPAPAI', 'PAPF', 'PAIAP', 'PAIAPPP', 'PPPAPPP', 'PAPPFP', 'PAPAPAP', 'PPAI', 'PPPAPP', 'PPAPAPP', 'PPPPPFP', 'PPPP', 'PAPAIAP', 'PPAIAI', 'PPPPPPF', 'PFPPAP', 'PAPAPP', 'PPAPAPA', 'PPAPP', 'PAIAIAI', 'PFPAPP', 'PPF', 'PPAPAPF', 'PFPPF', 'PPPPP', 'PPPAIAP', 'PAIAPAP'], ['PAP', 'PP', 'PFP', 'PPP']],
    "ogb_h6_nsl_hidden128_top20_seed42": [['PF', 'PFP', 'PAPF', 'PAIAI', 'PAIAP', 'PAPAP', 'PAIAPP', 'PAPAPF', 'PAPPAP', 'PAPPFP', 'PAIAIAP', 'PAIAPPP', 'PAPAPAI', 'PAPPPPP', 'PPAPFPP', 'PPPAPPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogb_h3_nsl_hidden128_top20_seed2": [['P', 'PF', 'PAI', 'PAP', 'PFP', 'PPF', 'PPP', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPF', 'PPAP', 'PPFP', 'PPPF', 'PPPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h3_nsl_hidden128_top20_seed3": [['P', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPP', 'PPAP', 'PPFP', 'PPPF'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h3_nsl_hidden128_top20_seed4": [['PF', 'PP', 'PAI', 'PAP', 'PPA', 'PPF', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPA', 'PFPF', 'PPAI', 'PPFP', 'PPPA', 'PPPF'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h3_nsl_hidden128_top20_seed5": [['P', 'PA', 'PP', 'PAI', 'PAP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPF', 'PAPP', 'PFPF', 'PFPP', 'PPAI', 'PPFP', 'PPPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogb_h3_nsl_hidden128_top20_seed6": [['P', 'PA', 'PF', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP', 'PAPP', 'PFPF', 'PFPP', 'PPAP', 'PPPF', 'PPPP'], ['PP', 'PAP', 'PFP', 'PPP']],


    "h3_top20_ns20_se128_nolabel": [['P', 'PF', 'PP', 'PAI', 'PAP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPA', 'PFPF', 'PPAI', 'PPAP', 'PPFP', 'PPPA', 'PPPF', 'PPPP'], []],
    "h4_top20_ns20_se128_nolabel": [['PF', 'PFP', 'PPF', 'PAPF', 'PAPP', 'PFPF', 'PPPF', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPF', 'PAPPP', 'PFPAP', 'PFPPF', 'PPAPF', 'PPAPP', 'PPFPF', 'PPFPP', 'PPPAP', 'PPPPP'], []],
    "h5_top20_ns20_se128_nolabel": [['PF', 'PP', 'PPPF', 'PPPP', 'PAPPF', 'PPAPF', 'PPPAP', 'PPPFP', 'PAPPAP', 'PAPPPF', 'PAPPPP', 'PFPPAP', 'PPAPAP', 'PPAPFP', 'PPAPPF', 'PPAPPP', 'PPFPPF', 'PPPAPF', 'PPPPAP', 'PPPPPP'], []],
    "h6_top20_ns20_se128_nolabel": [['PPP', 'PPPF', 'PPAPP', 'PAPPFP', 'PPAPAP', 'PPFPPF', 'PPFPPP', 'PPPFPP', 'PPPPAP', 'PAPAPAP', 'PAPFPFP', 'PAPPAPF', 'PAPPAPP', 'PAPPFPF', 'PFPFPAP', 'PPAPAPF', 'PPAPPPP', 'PPFPAPF', 'PPPAPFP', 'PPPPPPP'], []],

    "ogb_h4_ns20_se512_nolabel": [['PF', 'PAI', 'PPF', 'PAPF', 'PAPP', 'PFPF', 'PFPP', 'PPPF', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPP', 'PFPAP', 'PFPPF', 'PFPPP', 'PPAPF', 'PPAPP', 'PPFPP', 'PPPAP', 'PPPPP'], []],
    "ogb_h4_ns20_se512_label": [['PF', 'PP', 'PAI', 'PAP', 'PPF', 'PAPF', 'PAPP', 'PPAP', 'PPPF', 'PAIAP', 'PAPAI', 'PAPAP', 'PAPFP', 'PAPPP', 'PPAPP', 'PPFPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogb_h4_ns20_se512_nolabel_fix": [['PP', 'PAP', 'PFP', 'PAPP', 'PFPP', 'PPAP', 'PPFP', 'PPPP', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPP', 'PFPAP', 'PFPFP', 'PFPPP', 'PPAPP', 'PPFPP', 'PPPAP', 'PPPFP', 'PPPPP'], []],
    
    "ogb_h4_ns20_se512_label_fix": [['PP', 'PAP', 'PPP', 'PAPP', 'PPAP', 'PPFP', 'PPPP', 'PAIAP', 'PAPAP', 'PAPPP', 'PPAPP', 'PPFPP', 'PPPAP', 'PPPFP', 'PPPPF', 'PPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogb_h4_ns20_se512_label_fix_after": [['PF', 'PP', 'PAP', 'PPP', 'PAPF', 'PAPP', 'PPFP', 'PPPF', 'PAIAP', 'PAPFP', 'PAPPP', 'PPAPP', 'PPFPP', 'PPPFP', 'PPPPA', 'PPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogb_h4_ns20_se512_label_fix_fix":[['PF', 'PAP', 'PPP', 'PAPF', 'PAPP', 'PPFP', 'PPPF', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPP', 'PPAPP', 'PPFPP', 'PPPAP', 'PPPFP', 'PPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogb_h4_all": [['P', 'PA', 'PF', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPA', 'PFPF', 'PFPP', 'PPAI', 'PPAP', 'PPFP', 'PPPA', 'PPPF', 'PPPP', 'PAIAI', 'PAIAP', 'PAPAI', 'PAPAP', 'PAPFP', 'PAPPA', 'PAPPF', 'PAPPP', 'PFPAI', 'PFPAP', 'PFPFP', 'PFPPA', 'PFPPF', 'PFPPP', 'PPAIA', 'PPAPA', 'PPAPF', 'PPAPP', 'PPFPA', 'PPFPF', 'PPFPP', 'PPPAI', 'PPPAP', 'PPPFP', 'PPPPA', 'PPPPF', 'PPPPP'], []],

    "ogb_h4_reduce_top20": [['PF', 'PPF', 'PAPF', 'PAPP', 'PFPF', 'PPFP', 'PPPF', 'PPPP', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPF', 'PAPPP', 'PFPPP', 'PPAPF', 'PPAPP', 'PPFPP', 'PPPAP', 'PPPFP', 'PPPPP'], []],

    "ogbn_h3_ns20_se512_mask1_top20": [['P', 'PA', 'PF', 'PP', 'PAI', 'PAP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPA', 'PPAI', 'PPAP', 'PPFP', 'PPPA', 'PPPF', 'PPPP'], []],
    "ogbn_h3_ns20_se512_mask3_top20": [['PA', 'PF', 'PP', 'PAI', 'PAP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPA', 'PAPF', 'PAPP', 'PFPA', 'PFPF', 'PPAI', 'PPAP', 'PPFP', 'PPPA', 'PPPF', 'PPPP'], []],
    "ogbn_h3_ns20_se512_mask5_top20": [['PF', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPF', 'PAPP', 'PFPA', 'PFPF', 'PFPP', 'PPAI', 'PPAP', 'PPFP', 'PPPA', 'PPPF', 'PPPP'], []],
    "ogbn_h3_ns20_se512_mask7_top20": [['P', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPF', 'PAPP', 'PFPA', 'PFPF', 'PFPP', 'PPAI', 'PPAP', 'PPFP', 'PPPA', 'PPPF', 'PPPP'], []],
    "ogbn_h3_ns20_se512_mask9_top20": [['P', 'PA', 'PF', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP', 'PAIA', 'PAPP', 'PFPA', 'PFPF', 'PFPP', 'PPAI', 'PPAP', 'PPFP', 'PPPF', 'PPPP'], []],
    "ogbn_h4_ns20_se512_mask1_top20": [['PF', 'PFP', 'PPF', 'PAPF', 'PAPP', 'PPAI', 'PPPF', 'PPPP', 'PAIAP', 'PAPAI', 'PAPFP', 'PAPPF', 'PAPPP', 'PPAPF', 'PPAPP', 'PPFPF', 'PPPAI', 'PPPAP', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_ns20_se512_mask3_top20": [['PF', 'PFP', 'PPF', 'PAPF', 'PAPP', 'PFPF', 'PFPP', 'PPPF', 'PPPP', 'PAIAP', 'PAPPF', 'PAPPP', 'PPAPF', 'PPAPP', 'PPFPF', 'PPPAI', 'PPPAP', 'PPPFP', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_ns20_se512_mask5_top20": [['PFP', 'PAPF', 'PAPP', 'PFPF', 'PFPP', 'PPPF', 'PPPP', 'PAIAP', 'PAPPF', 'PAPPP', 'PFPAP', 'PFPPF', 'PPAPF', 'PPAPP', 'PPFPF', 'PPPAI', 'PPPAP', 'PPPFP', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_ns20_se512_mask7_top20": [['PFP', 'PAPF', 'PAPP', 'PFPF', 'PFPP', 'PPFP', 'PPPF', 'PPPP', 'PAIAP', 'PAPPF', 'PAPPP', 'PFPPF', 'PPAPF', 'PPAPP', 'PPFPF', 'PPPAI', 'PPPAP', 'PPPFP', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_ns20_se512_mask9_top20": [['PP', 'PFP', 'PPP', 'PAPP', 'PFPF', 'PFPP', 'PPAP', 'PPFP', 'PPPF', 'PPPP', 'PAIAP', 'PAPPP', 'PFPFP', 'PFPPF', 'PFPPP', 'PPAPP', 'PPFPP', 'PPPAP', 'PPPFP', 'PPPPP'], []],

    "ogbn_h4_deg10": [['PF', 'PAP', 'PPF', 'PPP', 'PAPF', 'PAPP', 'PPAP', 'PPPF', 'PPPP', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPF', 'PAPPP', 'PPAPF', 'PPAPP', 'PPFPP', 'PPPAI', 'PPPAP', 'PPPPP'], []],
    "ogbn_h4_deg5": [['PAP', 'PPF', 'PPP', 'PAPP', 'PPAP', 'PPPA', 'PPPP', 'PAIAP', 'PAPAI', 'PAPAP', 'PAPPA', 'PAPPF', 'PAPPP', 'PFPFP', 'PPAPP', 'PPPAI', 'PPPAP', 'PPPPA', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_deg15": [['PF', 'PFP', 'PPF', 'PAPF', 'PAPP', 'PPPA', 'PPPF', 'PPPP', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPF', 'PAPPP', 'PPAPF', 'PPAPP', 'PPFPP', 'PPPAI', 'PPPAP', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_deg20": [['PF', 'PFP', 'PPF', 'PAPF', 'PAPP', 'PPPF', 'PPPP', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPF', 'PAPPP', 'PPAPF', 'PPAPP', 'PPFPF', 'PPFPP', 'PPPAI', 'PPPAP', 'PPPPF', 'PPPPP'], []],

    "ogbn_h5_deg5": [['PP', 'PAP', 'PPP', 'PAPP', 'PAIAP', 'PAPAP', 'PPAPP', 'PPPAP', 'PPPPP', 'PAIAPP', 'PAPAPP', 'PAPPAP', 'PAPPPP', 'PPAIAP', 'PPAPAP', 'PPAPPP', 'PPPAPP', 'PPPPAI', 'PPPPAP', 'PPPPPP'], []],


    "ogbn_h4_outdeg5": [['PF', 'PFP', 'PPF', 'PAPF', 'PAPP', 'PFPP', 'PPAP', 'PPPF', 'PPPP', 'PAIAP', 'PAPPP', 'PFPAP', 'PFPPF', 'PPAPF', 'PPAPP', 'PPFPF', 'PPPAI', 'PPPAP', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_indeg5": [['PP', 'PAP', 'PPP', 'PAPA', 'PAPP', 'PPAI', 'PPAP', 'PPPA', 'PPPP', 'PAIAI', 'PAPAP', 'PAPFP', 'PAPPA', 'PAPPP', 'PPAPA', 'PPAPP', 'PPPAP', 'PPPPA', 'PPPPF', 'PPPPP'], []],

    "ogbn_h4_indeg3": [['PP', 'PAP', 'PPP', 'PAPA', 'PAPP', 'PPAP', 'PPPA', 'PPPP', 'PAIAI', 'PAPAP', 'PAPPA', 'PAPPP', 'PFPFP', 'PPAPA', 'PPAPP', 'PPPAI', 'PPPAP', 'PPPPA', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_indeg10": [['PP', 'PAI', 'PAP', 'PPP', 'PAPP', 'PPAI', 'PPAP', 'PPPA', 'PPPP', 'PAIAI', 'PAPAP', 'PAPFP', 'PAPPF', 'PAPPP', 'PPAPA', 'PPAPP', 'PPPAP', 'PPPPA', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_indeg20": [['PP', 'PAI', 'PAP', 'PPP', 'PAPP', 'PPAI', 'PPAP', 'PPPA', 'PPPP', 'PAPAP', 'PAPFP', 'PAPPF', 'PAPPP', 'PPAPP', 'PPFPP', 'PPPAI', 'PPPAP', 'PPPPA', 'PPPPF', 'PPPPP'], []],


    "ogbn_h6_indeg5": [['PP', 'PAP', 'PPP', 'PAPAP', 'PPAPP', 'PPPAP', 'PAPAPP', 'PAPPPP', 'PPAPAP', 'PPAPPP', 'PPPAPP', 'PPPPAP', 'PPPPPP', 'PAPAPPP', 'PAPPPAP', 'PAPPPFP', 'PAPPPPP', 'PPAPPPP', 'PPPPAPP', 'PPPPPPP'], []],

    "ogbn_h6_indeg10": [['PAP', 'PAPAP', 'PAPPP', 'PPAPP', 'PPPAP', 'PAPAPP', 'PAPPAP', 'PAPPPP', 'PPAPAP', 'PPAPPP', 'PPPAPP', 'PPPPAP', 'PPPPPP', 'PAPAPPP', 'PAPPPAP', 'PAPPPFP', 'PAPPPPP', 'PPAPPPP', 'PPPPAPP', 'PPPPPPP'], []],
    "ogbn_h6_indeg20": [['PP', 'PAP', 'PAPPP', 'PPAPP', 'PPPAP', 'PAPAPP', 'PAPPAP', 'PAPPPP', 'PPAPAP', 'PPAPPP', 'PPPAPP', 'PPPPAP', 'PPPPPP', 'PAPPPAP', 'PAPPPFP', 'PAPPPPP', 'PPAPPFP', 'PPAPPPP', 'PPPPAPP', 'PPPPPPP'], []],
    "ogbn_h6_indeg50": [['PP', 'PAP', 'PAPAP', 'PPAPP', 'PPPAP', 'PAPAPP', 'PAPPPP', 'PPAPAP', 'PPAPPP', 'PPPAPP', 'PPPPAP', 'PPPPPP', 'PAPPPAP', 'PAPPPFP', 'PAPPPPP', 'PPAPPFP', 'PPAPPPP', 'PPPPAPP', 'PPPPPFP', 'PPPPPPP'], []],

    "ogbn_h6_indeg50_seed2": [['PPP', 'PPAP', 'PAPAP', 'PPAPP', 'PPPAP', 'PAPPFP', 'PPAPAP', 'PPPAPP', 'PPPFPP', 'PPPPAP', 'PPPPPP', 'PAPAPAP', 'PAPAPPP', 'PAPFPPP', 'PAPPAPP', 'PPAPPAP', 'PPAPPPP', 'PPPAPFP', 'PPPAPPP', 'PPPPPPP'], []],
    "ogbn_h6_indeg50_seed42":[['PPPAPP', 'PAPPPAP', 'PPPAPPP', 'PAPPPPP', 'PAPPAP', 'PAPPPFP', 'PPPPAP', 'PPAPP', 'PPPPPAP', 'PAPPFP', 'PPAPPP', 'PPAPAPP', 'PAPAPP', 'PAPAP', 'PPPPPAI', 'PPPPAPP', 'PAPPFPP', 'PPAPFPP', 'PPPAP', 'PPPPPFP'], []],

    "ogbn_h6_indeg40_seed42": [['PPPAPP', 'PAPPPAP', 'PPPAPPP', 'PAPPPPP', 'PAPPAP', 'PPAPP', 'PPPPAP', 'PPPPPAP', 'PAPPPFP', 'PPAPPP', 'PPAPAPP', 'PAPAPP', 'PAPPFP', 'PAPAP', 'PPPAP', 'PPPPAPP', 'PPPPPFP', 'PPAPFPP', 'PPAPAP', 'PPAP'], []],
    "ogbn_h6_indeg30_seed42":[['PAPPPPP', 'PAPPPAP', 'PPPAPPP', 'PPAPP', 'PPPPAP', 'PPPPPAP', 'PAPPAP', 'PPPAPP', 'PAPPPFP', 'PAPAPP', 'PPAPPP', 'PAPAP', 'PPAPAPP', 'PAPPP', 'PPPPAPP', 'PAPPFP', 'PPAP', 'PPPPPFP', 'PPPAP', 'PAPPAPP'], []],
    
    "ogbn_h6_indeg5_label": [['PF', 'PP', 'PAPF', 'PAPP', 'PAPPP', 'PAIAIA', 'PAIAPA', 'PPPAPP', 'PAPAPPP', 'PAPPAPA', 'PAPPAPF', 'PAPPAPP', 'PAPPPAP', 'PPAPPPP', 'PPPPAPP', 'PPPPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogbn_h6_indeg10_label": [['PAPPP', 'PAIAIA', 'PAPFPP', 'PAPPAI', 'PAPPFP', 'PPPAPA', 'PPPAPP', 'PAPAPPP', 'PPAPPPP', 'PPFPFPP', 'PPPFPFP', 'PPPPAPA', 'PPPPAPP', 'PPPPPAP', 'PPPPPPA', 'PPPPPPF'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogbn_h6_indeg20_label": [['PAP', 'PAPPP', 'PPAPF', 'PAIAIA', 'PAPAPP', 'PAPPAI', 'PPFPAP', 'PPFPPF', 'PPPAPP', 'PAPAPPP', 'PAPPPAI', 'PAPPPPP', 'PFPFPFP', 'PFPFPPF', 'PPAPPPP', 'PPPPAPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogbn_h6_indeg50_label": [['PAP', 'PAPAP', 'PAPPP', 'PAIAIA', 'PAPPAI', 'PAPPFP', 'PPFPAP', 'PPFPPF', 'PPPAPA', 'PPPAPP', 'PAPPAPP', 'PAPPPPP', 'PPAPPPP', 'PPFPPAP', 'PPFPPFP', 'PPPPAPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogbn_h6_indeg5_label_seed2": [['PAP', 'PPP', 'PPFPP', 'PPPAP', 'PAPPAP', 'PPAPAP', 'PPAPPA', 'PPFPFP', 'PPPAIA', 'PPPPAP', 'PAIAIAP', 'PAIAPFP', 'PAPAPAI', 'PAPFPAI', 'PPAPPPP', 'PPPPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogbn_h6_nomask_label_top20": [['PF', 'PAIAP', 'PAPPP', 'PPPPF', 'PAIAPF', 'PAIAPP', 'PAPAPF', 'PAPPPF', 'PFPPPF', 'PPPAPP', 'PPPPAP', 'PAPPPPF', 'PPAPPPF', 'PPPAPPF', 'PPPPAPP', 'PPPPPAP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogbn_h6_nomask_nolabel_top20": [['PF', 'PPPF', 'PPPAP', 'PPPPF', 'PAPAPF', 'PAPPPF', 'PAPPPP', 'PPPAPF', 'PPPAPP', 'PPPFPF', 'PPPPAP', 'PPPPPP', 'PAPPPFP', 'PAPPPPF', 'PAPPPPP', 'PPAPAPF', 'PPAPPPF', 'PPPAPAI', 'PPPAPPF', 'PPPPAPP'], []],
    
    "ogbn_h4_nomask_label_top20": [['PF', 'PPF', 'PAPF', 'PAPP', 'PPAP', 'PPPF', 'PPPP', 'PAIAP', 'PAPAP', 'PAPPF', 'PAPPP', 'PPAPF', 'PPAPP', 'PPPAI', 'PPPPF', 'PPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    
    "ogbn_h6_nomask_label_top40": [['PAPPPPF', 'PAPPPF', 'PPPAPPF', 'PPPPAPP', 'PAIAPF', 'PF', 'PPAPPPF', 'PPPPAP', 'PPPPPAP', 'PAIAPP', 'PAPAPF', 'PAIAP', 'PPPPF', 'PFPPPF', 'PAPPP', 'PPPAPP', 'PAIAPPP', 'PAPPAPF', 'PPAPPPP', 'PAPAP', 'PAPF', 'PAPAIAP', 'PAIAPAP', 'PFPPPAI', 'PFPAPPP', 'PFPPFPA', 'PFPPPAP', 'PFPPPPF', 'PFPAIAI', 'PAPPFPF', 'PFPPAPF', 'PPPPPPA', 'PPAIAIA', 'PPAPF', 'PPAIA', 'PA'], ['PPP', 'PFP', 'PAP', 'PP']],
    "ogbn_h6_nomask_label_top60":[['PAPPPPF', 'PAPPPF', 'PPPAPPF', 'PPPPAPP', 'PAIAPF', 'PF', 'PPAPPPF', 'PPPPAP', 'PPPPPAP', 'PAIAPP', 'PAPAPF', 'PAIAP', 'PPPPF', 'PFPPPF', 'PAPPP', 'PPPAPP', 'PAIAPPP', 'PAPPAPF', 'PPAPPPP', 'PAPAP', 'PAPF', 'PAPAIAP', 'PAIAPAP', 'PFPPPAI', 'PFPAPPP', 'PFPPFPA', 'PFPPPAP', 'PFPPPPF', 'PFPAIAI', 'PAPPFPF', 'PFPPAPF', 'PPPPPPA', 'PPAIAIA', 'PPAPF', 'PPAIA', 'PA', 'PFPFPF', 'PAIAPPF', 'PAPAPAP', 'PPPPAI', 'PAPAPFP', 'PFPAI', 'PFP', 'PAPP', 'PAPAPPF', 'PAPFPAP', 'PPPAPA', 'PPFPAP', 'PFPPF', 'PFPFPP', 'PFPAIA', 'PFPAPP', 'PPFPP', 'PPPPAPF', 'PPFPPPP', 'PAIAIAP'], ['PPP', 'PFP', 'PAP', 'PP']],
    "ogbn_h6_nomask_label_top30": [['PAPPPPF', 'PAPPPF', 'PPPAPPF', 'PPPPAPP', 'PAIAPF', 'PF', 'PPAPPPF', 'PPPPAP', 'PPPPPAP', 'PAIAPP', 'PAPAPF', 'PAIAP', 'PPPPF', 'PFPPPF', 'PAPPP', 'PPPAPP', 'PAIAPPP', 'PAPPAPF', 'PPAPPPP', 'PAPAP', 'PAPF', 'PAPAIAP', 'PAIAPAP', 'PFPPPAI', 'PFPAPPP', 'PFPPFPA'], ['PPP', 'PFP', 'PAP', 'PP']],


    "ogbn_h6_indeg5_old": [['PP', 'PAP', 'PPPP', 'PAPPP', 'PPAPP', 'PPPAP', 'PAPAPP', 'PAPPPP', 'PPAPPP', 'PPPAPP', 'PPPPAP', 'PPPPPP', 'PAPAPPP', 'PAPPAPP', 'PAPPPAP', 'PAPPPPP', 'PPAPAPP', 'PPAPPPP', 'PPPPAPP', 'PPPPPPP'], []],
    "ogbn_h4_nomask_label_top20_old": [['PF', 'PPF', 'PAPF', 'PAPP', 'PPAP', 'PPFP', 'PPPF', 'PAIAP', 'PAPAP', 'PAPFP', 'PAPPP', 'PPAPP', 'PPFPP', 'PPPAP', 'PPPFP', 'PPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogbn_h6_nomask_label_top20_old": [['PAIAPAP', 'PP', 'PF', 'PPPPF', 'PAIAPPP', 'PPPF', 'PPFPP', 'PPFPAPP', 'PPFPPP', 'PAPAIAP', 'PPAPFPP', 'PAPPFPP', 'PAPPPP', 'PAPFPFP', 'PAIAIAP', 'PAPAPF'], ['PPP', 'PAP', 'PP', 'PFP']],
    "ogbn_h6_nomask_label_top40_old":[['PAIAPAP', 'PP', 'PF', 'PPPPF', 'PAIAPPP', 'PPPF', 'PPFPP', 'PPFPAPP', 'PPFPPP', 'PAPAIAP', 'PPAPFPP', 'PAPPFPP', 'PAPPPP', 'PAPFPFP', 'PAIAIAP', 'PAPAPF', 'PAPF', 'PPPFPP', 'PPPFPAP', 'PPPFPPF', 'PPPFPPP', 'PPFPFP', 'PAPAPPP', 'PAPPAPP', 'PAIAPP', 'PAP', 'PAPFPPP', 'PPPAPPF', 'PAPPPPP', 'PAPPPAP', 'PPFPAPF', 'PPPPFP', 'PPAPPPP', 'PPPPAPF', 'PPPAPP', 'PPF'], ['PPP', 'PAP', 'PP', 'PFP']],
    "ogbn_h6_nomask_label_top60_old":[['PAIAPAP', 'PP', 'PF', 'PPPPF', 'PAIAPPP', 'PPPF', 'PPFPP', 'PPFPAPP', 'PPFPPP', 'PAPAIAP', 'PPAPFPP', 'PAPPFPP', 'PAPPPP', 'PAPFPFP', 'PAIAIAP', 'PAPAPF', 'PAPF', 'PPPFPP', 'PPPFPAP', 'PPPFPPF', 'PPPFPPP', 'PPFPFP', 'PAPAPPP', 'PAPPAPP', 'PAIAPP', 'PAP', 'PAPFPPP', 'PPPAPPF', 'PAPPPPP', 'PAPPPAP', 'PPFPAPF', 'PPPPFP', 'PPAPPPP', 'PPPPAPF', 'PPPAPP', 'PPF', 'PAPFP', 'PAPPAP', 'PAPAP', 'PPAPPFP', 'PAPPP', 'PPFPPAP', 'PPFPFPF', 'PPAPPPA', 'PFPPPPF', 'PFPPFPP', 'PPAPPAP', 'PPAPAPP', 'PAPFPAP', 'PAPAPPF', 'PAPAPAP', 'PPFPPF', 'PPFPAP', 'PPAPPF', 'PPPPPAP', 'PPPPPPA'], ['PPP', 'PAP', 'PP', 'PFP']],


    "ogbn_h4_nomask_nolabel_top20_new_seed42": [['PF', 'PAP', 'PPF', 'PPP', 'PAPF', 'PAPP', 'PFPF', 'PPAP', 'PPPF', 'PPPP', 'PAIAP', 'PAPPF', 'PPAPF', 'PPAPP', 'PPPAI', 'PPPAP', 'PPPFP', 'PPPPA', 'PPPPF', 'PPPPP'], []],
    "ogbn_h4_nomask_label_top20_new_seed42": [['PF', 'PAP', 'PPF', 'PPP', 'PAPF', 'PAPP', 'PFPF', 'PPAP', 'PPPF', 'PAIAP', 'PAPPF', 'PFPPF', 'PPAPP', 'PPPAP', 'PPPPF', 'PPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],
    "ogbn_h6_nomask_label_top20_new_seed42": [['PF', 'PAPF', 'PFPF', 'PPPP', 'PAPAP', 'PFPPF', 'PPAPF', 'PAPAPF', 'PAPAPP', 'PAPFPP', 'PPAPPF', 'PPPAPP', 'PAIAIAP', 'PAPPPPP', 'PPPAPPP', 'PPPPPPF'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogbn_h4_nomask_labelh3_top20_new_seed42":[['PF', 'PP', 'PAP', 'PPF', 'PPP', 'PAPF', 'PAPP', 'PFPF', 'PPPF', 'PPPP', 'PAIAP', 'PAPPF', 'PPAPP', 'PPPAI', 'PPPAP', 'PPPPF'], ['PAP', 'PPP', 'PAPP', 'PPAP']],
    "ogbn_h4_nomask_labelh3_top20_new_seed42_fix": [['PPPF', 'PFPF', 'PAPF', 'PF', 'PAPPF', 'PPPAP', 'PPPPF', 'PPAPF', 'PAPP', 'PAIAP', 'PPAPP', 'PPPP', 'PAPPP', 'PAP'], ['PPAP', 'PPP', 'PAP', 'PPPP', 'PAPP', 'PP']],

    "ogbn_h4_nomask_label_top20_old_seed42": [['P', 'PF', 'PAP', 'PPF', 'PAPF', 'PAPP', 'PPAP', 'PPFP', 'PPPP', 'PAIAP', 'PAPFP', 'PAPPP', 'PPFPP', 'PPPAP', 'PPPFP', 'PPPPP'], ['PP', 'PAP', 'PFP', 'PPP']],

    "ogbn_h4_nomask_labelh4_top20_new_seed42_fix": [['PPPF', 'PAPF', 'PF', 'PFPF', 'PPPAP', 'PPPPF', 'PAPP', 'PAIAP', 'PP', 'PAPPF', 'PPPP'], ['PPP', 'PAPPP', 'PPPAP', 'PPPPP', 'PAP', 'PPPP', 'PAPAP', 'PPAP', 'PAPP']],

    "ogbn_h6_nomask_labelh4_top30": [['PPPAPP', 'PAPPPAP', 'PAPAPF', 'PPAPF', 'PAPPPPP', 'PPPAPPP', 'PPPPP', 'PAPAPPF', 'PPPPAP', 'PAPPPPF', 'PPPPPAP', 'PPAPPF', 'PPPPFPF', 'PPAPP', 'PPPPPAI', 'PF', 'PPPPPPF', 'PFPPAPF', 'PAPF', 'PFP', 'PAPPAP', 'PAPFPF', 'PPPPAPF', 'PPF', 'PPAP', 'PAPAPP'], ['PAPAP', 'PAPP', 'PP', 'PPAPP']],

    "ogbn_h4_nomask_labelh4_top30_new_seed42_fix": [['PPPF', 'PAPF', 'PF', 'PFPF', 'PPPAP', 'PPPPF', 'PAPP', 'PAIAP', 'PP', 'PAPPF', 'PPPP', 'PFPPF', 'PAPPP', 'PPPPP', 'PAP', 'PPP', 'PPAP', 'PPPFP'], ['PPP', 'PAPPP', 'PPPAP', 'PPPPP', 'PAP', 'PPPP', 'PAPAP', 'PPAP', 'PAPP', 'PAIAP', 'PP', 'PPFP']],

    "ogbn_h1_nolabel_all": [['P', 'PA', 'PF', 'PP'], []],
    "ogbn_h2_nolabel_all": [['P', 'PA', 'PF', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP'], []],
    "ogbn_h3_nolabel_all": [['P', 'PF', 'PP', 'PAI', 'PAP', 'PFP', 'PPA', 'PPF', 'PPP', 'PAPF', 'PAPP', 'PFPF', 'PPAP', 'PPFP', 'PPPF', 'PPPP'], []],

    "ogbn_h5_nolabel_top20":[['PAPF', 'PPPPF', 'PAPPF', 'PPPAPF', 'PPAPAP', 'PPF', 'PPPAPP', 'PPPF', 'PPPPFP', 'PAPP', 'PPFP', 'PFPAPF', 'PPAPPF', 'PAPFP', 'PF', 'PAPAPP', 'PAPPPF', 'PPPPPP', 'PPAPPP', 'PPPAP'], []],
    "ogbn_h5_nolabel_all": [['PAPF', 'PPPPF', 'PAPPF', 'PPPAPF', 'PPAPAP', 'PPF', 'PPPAPP', 'PPPF', 'PPPPFP', 'PAPP', 'PPFP', 'PFPAPF', 'PPAPPF', 'PAPFP', 'PF', 'PAPAPP', 'PAPPPF', 'PPPPPP', 'PPAPPP', 'PPPAP', 'PPAI', 'PPAIAI', 'PPAPF', 'PP', 'PAIAP', 'PPFPAP', 'PPPPPF', 'PAPPAI', 'PFPPPF', 'PFPPP', 'PFPPF', 'PPPAIA', 'PAPPFP', 'PPPFP', 'PAIAPP', 'PPAPAI', 'PPPAI', 'PFPPPP', 'PPAP', 'PAP', 'PPAPP', 'PPP', 'PAPAP', 'PFPAI', 'PPPP', 'PPPA', 'PPPPP', 'PAPAPF', 'PPPPAI', 'PFPPAP', 'PPAIAP', 'PAIAIA', 'PFPAP', 'PFPF', 'PPPAPA', 'PAIAI', 'PPAPA', 'PFPPAI', 'PPPPPA', 'PFPAPP', 'PAPPAP', 'PPPFPP', 'PAPPPA', 'PAPPPP', 'PPFPF', 'PPPPAP', 'PAPAI', 'PPPFPF', 'PAPPP', 'PPFPAI', 'PA', 'PPPPA', 'PAI', 'PPFPA', 'PPFPFP', 'PPAPPA', 'PAIAPF', 'PAPPA', 'PFPFPA', 'PFPP', 'PFPFPF', 'PFPPFP', 'PAPFPP', 'PAPAPA', 'PFPPA', 'PPFPPF', 'PFPFPP', 'PPAIA', 'PAPA', 'PAIA', 'PPAPFP', 'PPPFPA', 'PPFPPA', 'PAPFPA', 'PAPFPF', 'PPA', 'PFPAIA', 'PFP', 'PFPPPA', 'PAIAPA', 'P', 'PFPA', 'PFPFP', 'PPFPPP', 'PAPAIA', 'PFPAPA', 'PPFPP'], []],

    "ogbn_h6_nolabel_top20": [['PAPPPAP', 'PPPPPAP', 'PPPAPPP', 'PAPPAP', 'PPPAPP', 'PPAPF', 'PPPPPAI', 'PAPAPF', 'PAPPPPP', 'PPPPAP', 'PPAPP', 'PPPPPPF', 'PF', 'PAIAIAP', 'PAPF', 'PPAPPP', 'PPAPPF', 'PAPAPPF', 'PAPAPP', 'PAPPF'], []],
    "ogbn_h6_nolabel_all": [['PAPPPAP', 'PPPPPAP', 'PPPAPPP', 'PAPPAP', 'PPPAPP', 'PPAPF', 'PPPPPAI', 'PAPAPF', 'PAPPPPP', 'PPPPAP', 'PPAPP', 'PPPPPPF', 'PF', 'PAIAIAP', 'PAPF', 'PPAPPP', 'PPAPPF', 'PAPAPPF', 'PAPAPP', 'PAPPF', 'PPAPPPF', 'PAIAPP', 'PPF', 'PPAPAPP', 'PPPPAPP', 'PFPPAPF', 'PAPAP', 'PAPFPF', 'PPPF', 'PPPPF', 'PPFPAPP', 'PPAIAPF', 'PPPPPFP', 'PPPAPF', 'PPAPAP', 'PPFPPPF', 'PAPAIAP', 'PPPPAI', 'PAIAP', 'PFPF', 'PAPPAPF', 'PPAPPAI', 'PAPPPPF', 'PFP', 'PAIAPAP', 'PPPPAPF', 'PFPAPP', 'PPPP', 'PFPPPAI', 'PPFPPF', 'PPAPFPP', 'PPPFPP', 'PPAPFPA', 'PPPFPPF', 'PAPPFPF', 'PPPPPF', 'PAIAPPP', 'PAPPP', 'PFPFPAP', 'PFPAPAP', 'PAPPPFP', 'PPPFPPP', 'PPFPAPF', 'PFPAPPP', 'PAPPAI', 'PAPPPAI', 'PFPPF', 'PPFPPP', 'PAIAPFP', 'PPFPPPP', 'PPAPPFP', 'PAPFPPF', 'PPPAPAP', 'PAI', 'PFPPP', 'PAP', 'PPAPFPF', 'PPPPP', 'PFPPAI', 'PP', 'PPPFPAP', 'PFPFPPF', 'PPFPPAI', 'PAPAIAI', 'PFPFPPA', 'PPPAP', 'PPAPPPP', 'PFPAPFP', 'PAPPFP', 'PPPAIAI', 'PAPPA', 'PPFPPAP', 'PAPAPPP', 'PPAIAP', 'PPPFP', 'PPPFPF', 'PPPPA', 'PPPPFPF', 'PPPPAPA', 'PAPFPAI', 'PPPPPP', 'PAPPPF', 'PPAPA', 'PAPAPPA', 'PAPPPP', 'PPPAIAP', 'PPAPPPA', 'PPPAI', 'PAPPAPP', 'PFPAPF', 'PFPFPAI', 'PAPP', 'PFPPAPA', 'PPAPAPF', 'PPFPF', 'PPPPPPP', 'PPFPP', 'PPA', 'PAIAPAI', 'PPAPAI', 'PPPAPA', 'PPPAPPF', 'PPAP', 'PAIAPF', 'PAPFPAP', 'PFPP', 'PAPFPPP', 'PPPFPAI', 'PFPPFPP', 'PPFPAP', 'PAPPPA', 'PAIAPPA', 'PPFPPFP', 'PAIAIAI', 'PFPAI', 'PAPAPA', 'PAPAPFP', 'PFPPFPA', 'PPFPAPA', 'PPFPFPA', 'PPPAPPA', 'PPFPPA', 'PAPFPFP', 'PPPPFPP', 'PFPPA', 'PAIAPA', 'PFPAPAI', 'PFPPPPA', 'PFPFPP', 'PPFPAI', 'PPAI', 'PAPPPPA', 'PPPFPPA', 'PAPFPPA', 'PPAPPAP', 'PAPAPAI', 'PFPPPFP', 'PFPPAPP', 'PPFP', 'PAPAI', 'PAPPFPP', 'PFPFP', 'PFPAPPF', 'PPPA', 'PAPPFPA', 'PFPPFPF', 'PPP', 'PPAIAPP', 'PPAIAI', 'PAPAPAP', 'PFPPPAP', 'PPPPFP', 'PFPPAP', 'PFPPPA', 'PPAPAIA', 'PA', 'PPPAPAI', 'PFPPAIA', 'PFPAIAI', 'PPAPPA', 'PFPFPFP', 'PPAPAPA', 'PFPPPP', 'PFPAP', 'PAPFP', 'PAIA', 'PFPFPPP', 'PFPPPPF', 'PPPAPFP', 'PPFPFP', 'PPPPPA', 'PPAIA', 'PAIAI', 'PAPA', 'PFPA', 'PPPFPA', 'PPPAIA', 'PPFPPPA', 'PAPFPA', 'PFPPPF', 'PFPAPA', 'PPPPPPA', 'PFPFPF', 'PAIAPPF', 'PFPPPPP', 'PFPPFP', 'PAIAIA', 'PPPFPFP', 'PFPAPPA', 'PAPPAPA', 'PPFPFPP', 'PFPAIAP', 'PAPFPP', 'PPFPFPF', 'PPFPAIA', 'PPFPA', 'P', 'PPPPAIA', 'PPAPFP', 'PAPPAIA', 'PPPPFPA', 'PPAIAIA', 'PAPAIA', 'PPAIAPA', 'PFPFPA', 'PFPAIA'], []],

    "ogbn_h3_nolabel_top20": [['PAPF', 'PPPF', 'PPF', 'PPAP', 'PF', 'PAPP', 'PPPP', 'PPP', 'PAP', 'PFPF', 'PPFP', 'PP', 'PPAI', 'PPPA', 'PAIA', 'P', 'PFPA', 'PPA', 'PFPP', 'PAI'], []],
    "ogbn_h3_nolabel_all": [['PAPF', 'PPPF', 'PPF', 'PPAP', 'PF', 'PAPP', 'PPPP', 'PPP', 'PAP', 'PFPF', 'PPFP', 'PP', 'PPAI', 'PPPA', 'PAIA', 'P', 'PFPA', 'PPA', 'PFPP', 'PAI', 'PFP', 'PAPA', 'PA'], []],

    "ogbn_h5_nolabel_all_seed1": [['PAPAPP', 'PPPF', 'PPPPF', 'PPAPPF', 'PAPF', 'PPAPF', 'PAPP', 'PAPPPF', 'PF', 'PAPPAP', 'PAPPF', 'PPPPAP', 'PPPPP', 'PP', 'PPF', 'PPPFPF', 'PPPPPP', 'PAIAPP', 'PPPAI', 'PPPAPP', 'PPPPPF', 'PPAPPP', 'PPPAPF', 'PAPAPF', 'PFPF', 'PPPAP', 'PAPPFP', 'PAPFPF', 'PAPPPP', 'PFPAPF', 'PPFPPF', 'PAPPAI', 'PPAPAP', 'PPAI', 'PPAIAI', 'PAP', 'PPPPAI', 'PAIAPF', 'PPP', 'PPAPP', 'PAIAP', 'PPAPAI', 'PPAIAP', 'PPPFPP', 'PFPPAP', 'PAPAP', 'PPPFP', 'P', 'PPFPAP', 'PFPPP', 'PPPPA', 'PFPAIA', 'PPPAIA', 'PPFPF', 'PFPFPA', 'PFP', 'PAPPPA', 'PAPPA', 'PAPAI', 'PPPPFP', 'PPA', 'PFPPF', 'PPFP', 'PAIAI', 'PAPAIA', 'PAI', 'PPPA', 'PAPFPP', 'PFPFP', 'PPFPPP', 'PPPPPA', 'PAPFP', 'PPPP', 'PPPFPA', 'PPAPFP', 'PAPA', 'PAIA', 'PPAP', 'PFPP', 'PFPFPF', 'PAPPP', 'PAIAPA', 'PFPPAI', 'PPFPAI', 'PFPPPA', 'PFPAI', 'PAPAPA', 'PPFPFP', 'PPPAPA', 'PPAPA', 'PAIAIA', 'PFPPFP', 'PPFPP', 'PPAPPA', 'PFPPPP', 'PFPAPA', 'PPFPPA', 'PAPFPA', 'PPFPA', 'PFPFPP', 'PFPA', 'PA', 'PFPPA', 'PFPPPF', 'PPAIA', 'PFPAPP', 'PFPAP'], []],
    "ogbn_h5_nolabel_top20_seed1":[['PAPAPP', 'PPPF', 'PPPPF', 'PPAPPF', 'PAPF', 'PPAPF', 'PAPP', 'PAPPPF', 'PF', 'PAPPAP', 'PAPPF', 'PPPPAP', 'PPPPP', 'PP', 'PPF', 'PPPFPF', 'PPPPPP', 'PAIAPP', 'PPPAI', 'PPPAPP'], []],

    "ogbn_h4_nolabel_top20": [['PF', 'PAP', 'PPF', 'PPP', 'PAPF', 'PAPP', 'PFPF', 'PPAP', 'PPPF', 'PPPP', 'PAIAP', 'PAPPF', 'PPAPF', 'PPAPP', 'PPPAI', 'PPPAP', 'PPPFP', 'PPPPA', 'PPPPF', 'PPPPP'], []],


    "ogbn_h4_nomask_labelh4_top20_emb_seed42": [['PF', 'PPPA', 'PAPF', 'PPPF', 'PAPP', 'PPPPA', 'PPPPF', 'PAPPF', 'PAPPA', 'PPAPA', 'PFPPF', 'PA'], ['PAPAP', 'PAP', 'PAPPP', 'PPAP', 'PPPP', 'PPPPP', 'PAPP', 'PP']],
    "ogbn_h4_nomask_labelh4_top30_emb_seed42": [['PF', 'PPPA', 'PAPF', 'PPPF', 'PAPP', 'PPPPA', 'PPPPF', 'PAPPF', 'PAPPA', 'PPAPA', 'PFPPF', 'PA', 'PP', 'PFPF', 'PPAPF', 'PAPA', 'PAP', 'PAIA', 'PAIAP', 'PPP'], ['PAPAP', 'PAP', 'PAPPP', 'PPAP', 'PPPP', 'PPPPP', 'PAPP', 'PP', 'PPPAP', 'PPP']],
    "ogbn_h4_nomask_labelh4_all_emb_seed42": [['PF', 'PPPA', 'PAPF', 'PPPF', 'PAPP', 'PPPPA', 'PPPPF', 'PAPPF', 'PAPPA', 'PPAPA', 'PFPPF', 'PA', 'PP', 'PFPF', 'PPAPF', 'PAPA', 'PAP', 'PAIA', 'PAIAP', 'PPP', 'PPA', 'PPF', 'PPPPP', 'PAPPP', 'PFPAP', 'PPPP', 'PFPAI', 'PPAI', 'PPAP', 'PPFPA', 'PPPFP', 'PAPAP', 'PFPA', 'PPPAI', 'PAIAI', 'PPAIA', 'PAPFP', 'PFPPP', 'PPFPF', 'PAPAI', 'PPPAP', 'PFPPA', 'PPAPP', 'P', 'PFP', 'PAI', 'PFPP', 'PFPFP', 'PPFPP', 'PPFP'], ['PAPAP', 'PAP', 'PAPPP', 'PPAP', 'PPPP', 'PPPPP', 'PAPP', 'PP', 'PPPAP', 'PPP', 'PAIAP', 'PPPFP', 'PPAPP', 'PFP', 'PPFP', 'PFPAP', 'PFPPP', 'PAPFP', 'PFPFP', 'PPFPP', 'PFPP']],
    
    "ogbn_h4_nomask_labelh4_top20_emb_seed1": [['PF', 'PAPF', 'PPPF', 'PPPP', 'PPPPF', 'PFPF', 'PAPP', 'PPPPP', 'PAPPA'], ['PPPAP', 'PPPP', 'PAPP', 'PPP', 'PAP', 'PAPPP', 'PPPPP', 'PPAP', 'PAPAP', 'PP', 'PPAPP']],
    "ogbn_h4_nomask_labelh4_top30_emb_seed1":[['PF', 'PAPF', 'PPPF', 'PPPP', 'PPPPF', 'PFPF', 'PAPP', 'PPPPP', 'PAPPA', 'PAPPF', 'PAIAP', 'PPAPF', 'PPF', 'PPFPA', 'PAPPP', 'PPPA', 'PFPPF', 'PPPPA', 'PAPAP'], ['PPPAP', 'PPPP', 'PAPP', 'PPP', 'PAP', 'PAPPP', 'PPPPP', 'PPAP', 'PAPAP', 'PP', 'PPAPP']],
    "ogbn_h4_nomask_labelh4_all_emb_seed1": [['PF', 'PAPF', 'PPPF', 'PPPP', 'PPPPF', 'PFPF', 'PAPP', 'PPPPP', 'PAPPA', 'PAPPF', 'PAIAP', 'PPAPF', 'PPF', 'PPFPA', 'PAPPP', 'PPPA', 'PFPPF', 'PPPPA', 'PAPAP', 'PA', 'PPAPP', 'PPPAP', 'PPAPA', 'PAPA', 'PAIAI', 'PPPAI', 'PPA', 'PAPAI', 'PFPA', 'PAIA', 'PAI', 'PPAP', 'PFP', 'PFPPP', 'PPPFP', 'PPP', 'PAPFP', 'PP', 'PPAI', 'PFPAI', 'PPAIA', 'PFPAP', 'PFPP', 'P', 'PPFPF', 'PAP', 'PPFP', 'PPFPP', 'PFPPA', 'PFPFP'], ['PPPAP', 'PPPP', 'PAPP', 'PPP', 'PAP', 'PAPPP', 'PPPPP', 'PPAP', 'PAPAP', 'PP', 'PPAPP', 'PAIAP', 'PPFP', 'PAPFP', 'PFPPP', 'PFPFP', 'PFP', 'PFPAP', 'PFPP', 'PPPFP', 'PPFPP']],

}
