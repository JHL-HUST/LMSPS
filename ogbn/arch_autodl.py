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


    "ogb_h1_nsl_hidden128": [['PF', 'PP', 'P', 'PA'], ['PFP', 'PP', 'PAP', 'PPP']],
    "ogb_h2_nsl_hidden128": [['PF', 'P', 'PAP', 'PAI', 'PPA', 'PP', 'PA', 'PPP', 'PPF', 'PFP'], ['PAP', 'PFP', 'PPP', 'PP']],

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

    "ogb_h3_nsl_hidden128_lin_seed42": [['PF', 'PAPF', 'PP', 'PAPP', 'PFPF', 'PAP', 'PPF', 'PPPF', 'PAI', 'PFP', 'PPFP', 'PFPA', 'PPA', 'PPPA', 'PPPP'], ['PFP', 'PPP', 'PAP', 'PP']]
}