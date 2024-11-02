# from .E_TRFNetv0 import E_TRFNetv0
from .EVDI import EVDI
from .REFID.REFID_twosharpInterp import REFID

# RGB method
from .SuperSlomo.runSuperSlomo import SuperSlomo
from .RIFE.runRIFE import RIFE

from models.Expv7_dataprop.runExpv7_dataprop import Expv7_dataprop
from models.Expv7_dataprop_fuse.runExpv7_dataprop_fuse import Expv7_datafuse
from models.Expv7_dataprop_fuseFeat.runExpv7_dataprop_fuseFeat import Expv7_datafuseFeat
from models.Expv7_dataprop_fuseFeat_large.runExpv7_dataprop_fuseFeat_large import Expv7_datafuseFeat_large
from models.Expv7_dataprop_fuseFeat_onlyEvents.runExpv7_dataprop_fuseFeat_onlyEvents import Expv7_datafuseFeat_onlyEvents
from models.Expv7_dataprop_fuseFeat_direct.runExpv7_dataprop_fuseFeat_direct import Expv7_datafuseFeat_direct

# from models.Expv8.runExpv8 import Expv8
# from models.Expv9.runExpv9 import Expv9
# from models.Expv8_Light.runExpv8_Light import Expv8_Light
from models.Expv8_Lights2.runExpv8_Lights2 import Expv8_Lights2
# for ablation
# from models.Expv8_Lights2_fieldflow.runExpv8_Lights2fieldflow import Expv8_Lights2fieldflow
from models.Expv8_large.runExpv8_large import Expv8_large

# For peer comparisons
from models.timelens.runtimelens import TimeLens
from models.timelens_flow.runtimelens import TimeLens_flow
# from models.RGBGT.runRGBGT import RGBGT
from models.Expv8_Lights3.runExpv8_Lights3 import Expv8_Lights3
try:
	from models.CBMNet.runcbmnet import CBMNet
except:
	pass

from models.CBMNet.runmycbmnet import myCBMNet
# from models.CBMNet.runcbmnet_large import CBMNet_large

# Ablation study
from models.Expv8_Lights2norefine.runExpv8_Lights2norefine import Expv8_Lights2norefine
from models.Expv8_Lights2fieldflow.runExpv8_Lights2fieldflow import Expv8_Lights2fieldflow
from models.Expv8_Lights2fieldflow_direct.runExpv8_Lights2fieldflow_direct import Expv8_Lights2fieldflow_direct
from models.Expv8_Lights2fieldflownorefine_direct.runExpv8_Lights2fieldflownorefine_direct import Expv8_Lights2fieldflownorefine_direct
from models.RGBGT.runRGBGT import RGBGT