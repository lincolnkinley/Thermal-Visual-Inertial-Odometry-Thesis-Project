from boson_driver import *

#x = pyClient.Initialize(manualport=16, useDll=False)
x = pyClient.Initialize(manualport="/dev/ttyBoson", useDll=False)
#pyClient.bosonRunFFC()
#

pyClient.dvoSetDisplayMode(FLR_DVO_DISPLAY_MODE_E.FLR_DVO_ONE_SHOT)
print(pyClient.dvoGetDisplayMode())

pyClient.bosonSetExtSyncMode(FLR_BOSON_EXT_SYNC_MODE_E.FLR_BOSON_EXT_SYNC_SLAVE_MODE)
print(pyClient.bosonGetExtSyncMode())

#print(pyClient.gaoSetAveragerState(FLR_ENABLE_E.FLR_ENABLE))
print(pyClient.gaoGetAveragerState())
