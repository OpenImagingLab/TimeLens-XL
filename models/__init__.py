# from .E_TRFNetv0 import E_TRFNetv0
from .EVDI import EVDI
from .REFID.REFID_twosharpInterp import REFID

# RGB method
from .SuperSlomo.runSuperSlomo import SuperSlomo
from models.Expv8_large.runExpv8_large import Expv8_large

# For peer comparisons
from models.timelens.runtimelens import TimeLens
from models.timelens_flow.runtimelens import TimeLens_flow
from models.Expv8_Lights3.runExpv8_Lights3 import Expv8_Lights3

from models.CBMNet.runmycbmnet import myCBMNet
from models.RGBGT.runRGBGT import RGBGT