import CareyPlots

# app = CareyPlots.exploreVideo(r"X:\data\2022\BATCH5\behavior\VIV_23058\S10\VIV_23058_BodyCamVideo2022-03-24_S10_1.avi")
app = CareyPlots.exploreVideo(r"X:\data\2022\BATCH5\behavior\VIV_23058\S10\VIV_23058_BodyCamVideo2022-03-24_S10_1DLC_resnet50_npx_cage_bodycamFeb8shuffle1_500000_labeled")
app.run_server(debug=True)