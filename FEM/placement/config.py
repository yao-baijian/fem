# config.py
from enum import Enum
import rapidwright

from com.xilinx.rapidwright.device import SiteTypeEnum

class PlaceType(Enum):
    CENTERED = 1
    IO = 2
    OTHER = 3
    
class GridType(Enum):
    SQUARE = 1
    RECTAN = 2
    OTHER = 3

SLICE_SITE_ENUM = [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]

IO_SITE_ENUM = [
    SiteTypeEnum.HPIOB, 
    SiteTypeEnum.HRIO
]

# CLOCK_SITE_ENUM = [SiteTypeEnum.BUFGCE]

OTHER_SITE_ENUM = [SiteTypeEnum.BUFGCE, SiteTypeEnum.DSP48E2, SiteTypeEnum.RAMB36, SiteTypeEnum.BITSLICE_COMPONENT_RX_TX,]