# config.py
from enum import Enum
import rapidwright

from com.xilinx.rapidwright.device import SiteTypeEnum

class PlaceType(Enum):
    CENTERED = 1
    IO = 2
    OTHER = 3

SLICE_SITE_ENUM = [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]
IO_SITE_ENUM = [
    SiteTypeEnum.HPIOB, 
    SiteTypeEnum.HRIO, 
    SiteTypeEnum.BITSLICE_COMPONENT_RX_TX,
    SiteTypeEnum.BUFGCE
]
CLOCK_SITE_ENUM = [SiteTypeEnum.BUFGCE]
OTHER_SITE_ENUM = [SiteTypeEnum.DSP48E2, SiteTypeEnum.RAMB36]