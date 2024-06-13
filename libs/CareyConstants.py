import seaborn as sns

class CareyConstants:

    paw_labels = ['FR', 'HR', 'FL', 'HL']              # paw labels
    paw_colors = [[1,0,0], [1,0,1], [0,0,1], [0,1,1]]  # paw colors

    cpaw = [3, 4, 1, 2]                                # contralateral paw indexes
    cpaw_labels = ['FL', 'HL', 'FR', 'HR']             # contralateral paw labels
    cpaw_colors = [[0,0,1], [0,1,1], [1,0,0], [1,0,1]] # contralateral paw colors

    dpaw = [4, 3, 2, 1]                                # diagonal paw indexes
    dpaw_labels = ['HL', 'FL', 'HR', 'FR']             # diagonal paw labels
    dpaw_colors = [[0,1,1], [0,0,1], [1,0,1], [1,0,0]] # diagonal paw colors

    hpaw = [2, 1, 4, 3]                                # homolateral paw indexes
    hpaw_labels = ['HR', 'FR', 'HL', 'FL']             # homolateral paw labels
    hpaw_colors = [[1,0,1], [1,0,0], [0,1,1], [0,0,1]] # homolateral paw colors

    subplot_topview_order = [2,4,1,3]  # the order a subplot should be filled so that
    # the FL paw takes position top left, etc

    paw_idx = {}
    paw_idx["FR"] = 0
    paw_idx["HR"] = 1
    paw_idx["FL"] = 2
    paw_idx["HL"] = 3

    DEF_NPX_FS = 30000 # default sampling rate for neuropixels
    DEF_BEHAV_FS = 430

    sns_color_palette = 'deep'

    paw_colors_sns = [sns.color_palette(sns_color_palette)[3], sns.color_palette(sns_color_palette)[6],
                      sns.color_palette(sns_color_palette)[0], sns.color_palette(sns_color_palette)[9]]
    paw_colors_sns_dict = {'FR': sns.color_palette(sns_color_palette)[3],
                           'HR': sns.color_palette(sns_color_palette)[6],
                           'FL': sns.color_palette(sns_color_palette)[0],
                           'HL': sns.color_palette(sns_color_palette)[9]}

    # pc to mm scale
    ## 210.81973994640725 in pixel distance corresponds to 12cm
    NPXRIG_PX_PER_MM = 210.81973994640725 / 120.0 # 1.7568311662200604
    NPXRIG_MM_PER_PX = 1/NPXRIG_PX_PER_MM