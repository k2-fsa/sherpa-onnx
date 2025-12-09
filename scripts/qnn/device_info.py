#!/usr/bin/env python3
from dataclasses import dataclass
from enum import IntEnum, unique

"""
See also
https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html#supported-snapdragon-devices

SA8255 soc_id    52 dsp_arch     v73 vtcm_size (MB)      8
SA8295 soc_id    39 dsp_arch     v68 vtcm_size (MB)      8
SM8350 soc_id    35 dsp_arch     v68 vtcm_size (MB)      4
SM8450 soc_id    36 dsp_arch     v69 vtcm_size (MB)      8
SM8475 soc_id    42 dsp_arch     v69 vtcm_size (MB)      8
SM8550 soc_id    43 dsp_arch     v73 vtcm_size (MB)      8
SM8650 soc_id    57 dsp_arch     v75 vtcm_size (MB)      8
SM8750 soc_id    69 dsp_arch     v79 vtcm_size (MB)      8
SM8850 soc_id    87 dsp_arch     v81 vtcm_size (MB)      8
SSG2115P soc_id  46 dsp_arch     v73 vtcm_size (MB)      2
SSG2125P soc_id  58 dsp_arch     v73 vtcm_size (MB)      2
SXR1230P soc_id  45 dsp_arch     v73 vtcm_size (MB)      2
SXR2230P soc_id  53 dsp_arch     v69 vtcm_size (MB)      8
SXR2330P soc_id  75 dsp_arch     v79 vtcm_size (MB)      8
QCS9100 soc_id   77 dsp_arch     v73 vtcm_size (MB)      8
SAR2230P soc_id  95 dsp_arch     v81 vtcm_size (MB)      4
SW6100 soc_id    96 dsp_arch     v81 vtcm_size (MB)      4
"""


@dataclass
class Chipset(IntEnum):
    # see https://github.com/pytorch/executorch/blob/main/backends/qualcomm/serialization/qc_schema.py#L41
    # SA8255, soc_id 52,  dsp_arch v73
    SA8255 = 52  # v73
    SA8295 = 39  # v68
    SM8350 = 35  # v68
    SM8450 = 36  # v69
    SM8475 = 42  # v69
    SM8550 = 43  # v73
    SM8650 = 57  # v75
    SM8750 = 69  # v79
    SM8850 = 87  # v81
    #  SSG2115P = 46  # v73
    #  SSG2125P = 58  # v73
    #  SXR1230P = 45  # v73
    #  SXR2230P = 53  # v69
    #  SXR2330P = 75  # v79
    QCS9100 = 77  # v73
    #  SAR2230P = 95  # v81
    #  SW6100 = 96  # v81


@dataclass
class HtpArch(IntEnum):
    v68 = 68
    v69 = 69
    v73 = 73
    v75 = 75
    v79 = 79
    v81 = 81
    v87 = 87


@dataclass
class HtpInfo:
    arch: HtpArch
    vtcm_size_in_mb: int


@dataclass
class SocInfo:
    model: Chipset
    info: HtpInfo


soc_info_list = [
    SocInfo(Chipset.SA8255, HtpInfo(HtpArch.v73, 8)),
    SocInfo(Chipset.SA8295, HtpInfo(HtpArch.v68, 8)),
    SocInfo(Chipset.SM8350, HtpInfo(HtpArch.v68, 4)),
    SocInfo(Chipset.SM8450, HtpInfo(HtpArch.v69, 8)),
    SocInfo(Chipset.SM8475, HtpInfo(HtpArch.v69, 8)),
    SocInfo(Chipset.SM8550, HtpInfo(HtpArch.v73, 8)),
    SocInfo(Chipset.SM8650, HtpInfo(HtpArch.v75, 8)),
    SocInfo(Chipset.SM8750, HtpInfo(HtpArch.v79, 8)),
    SocInfo(Chipset.SM8850, HtpInfo(HtpArch.v81, 8)),
    #  SocInfo(Chipset.SSG2115P, HtpInfo(HtpArch.v73, 2)),
    #  SocInfo(Chipset.SSG2125P, HtpInfo(HtpArch.v73, 2)),
    #  SocInfo(Chipset.SXR1230P, HtpInfo(HtpArch.v73, 2)),
    #  SocInfo(Chipset.SXR2230P, HtpInfo(HtpArch.v69, 8)),
    #  SocInfo(Chipset.SXR2330P, HtpInfo(HtpArch.v79, 8)),
    SocInfo(Chipset.QCS9100, HtpInfo(HtpArch.v73, 8)),
    #  SocInfo(Chipset.SAR2230P, HtpInfo(HtpArch.v81, 4)),
    #  SocInfo(Chipset.SW6100, HtpInfo(HtpArch.v81, 4)),
]

soc_info_dict = {soc.model.name: soc for soc in soc_info_list}


def _test():
    for soc in soc_info_list:
        print(
            soc.model.name,
            "soc_id\t",
            soc.model.value,
            "dsp_arch\t",
            soc.info.arch.name,
            "vtcm_size (MB)\t",
            soc.info.vtcm_size_in_mb,
        )


if __name__ == "__main__":
    _test()
