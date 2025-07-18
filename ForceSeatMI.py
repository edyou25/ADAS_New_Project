#
# Copyright (C) 2012-2023 MotionSystems
#
# This file is part of ForceSeatMI SDK.
#
# www.motionsystems.eu
#

from ForceSeatMI_Structs import *
from winreg import *

import ctypes
import sys

class FSMI_HandleType(ctypes.Structure):
    pass

def getDllPath() -> str:
    entryName = "InstallationPath"
    is_64bits = sys.maxsize > 2**32

    if is_64bits:
        reg = "SOFTWARE\\Wow6432Node\\MotionSystems\\ForceSeatPM"
        dllName = "ForceSeatMI64"
    else:
        reg = "SOFTWARE\\MotionSystems\\ForceSeatPM"
        dllName = "ForceSeatMI32"

    key   = OpenKeyEx(HKEY_LOCAL_MACHINE, reg)
    value = QueryValueEx(key, entryName)
    path  = value[0] + dllName

    return path


class ForceSeatMI(object):
    def __init__(self):
        path = getDllPath()
        self.lib = CDLL(path)

        # Create API
        self.lib.ForceSeatMI_Create.restype = ctypes.POINTER(FSMI_HandleType)

        self.lib.ForceSeatMI_BeginMotionControl.restype     = FSMI_Bool
        self.lib.ForceSeatMI_EndMotionControl.restype       = FSMI_Bool
        self.lib.ForceSeatMI_GetPlatformInfoEx.restype      = FSMI_Bool
        self.lib.ForceSeatMI_SendTelemetryACE.restype       = FSMI_Bool
        self.lib.ForceSeatMI_SendTelemetry2.restype         = FSMI_Bool
        self.lib.ForceSeatMI_SendTopTablePosPhy.restype     = FSMI_Bool
        self.lib.ForceSeatMI_SendTopTablePosPhy2.restype    = FSMI_Bool
        self.lib.ForceSeatMI_SendTopTableMatrixPhy.restype  = FSMI_Bool
        self.lib.ForceSeatMI_SendTopTableMatrixPhy2.restype = FSMI_Bool
        self.lib.ForceSeatMI_ActivateProfile.restype        = FSMI_Bool
        self.lib.ForceSeatMI_SetAppID.restype               = FSMI_Bool
        self.lib.ForceSeatMI_Park.restype                   = FSMI_Bool

        self.lib.ForceSeatMI_BeginMotionControl.argtypes     = ctypes.POINTER(FSMI_HandleType),
        self.lib.ForceSeatMI_EndMotionControl.argtypes       = ctypes.POINTER(FSMI_HandleType),
        self.lib.ForceSeatMI_GetPlatformInfoEx.argtypes      = ctypes.POINTER(FSMI_HandleType), ctypes.POINTER(FSMI_PlatformInfo), FSMI_UINT32, FSMI_UINT32
        self.lib.ForceSeatMI_SendTelemetryACE.argtypes       = ctypes.POINTER(FSMI_HandleType), FSMI_TelemetryACE
        self.lib.ForceSeatMI_SendTelemetry2.argtypes         = ctypes.POINTER(FSMI_HandleType), FSMI_TelemetryACE, FSMI_SFX, FSMI_TactileAudioBasedFeedbackEffects
        self.lib.ForceSeatMI_SendTopTablePosPhy.argtypes     = ctypes.POINTER(FSMI_HandleType), FSMI_TopTablePositionPhysical
        self.lib.ForceSeatMI_SendTopTablePosPhy2.argtypes    = ctypes.POINTER(FSMI_HandleType), FSMI_TopTablePositionPhysical, FSMI_SFX, FSMI_TactileAudioBasedFeedbackEffects
        self.lib.ForceSeatMI_SendTopTableMatrixPhy.argtypes  = ctypes.POINTER(FSMI_HandleType), FSMI_TopTableMatrixPhysical
        self.lib.ForceSeatMI_SendTopTableMatrixPhy2.argtypes = ctypes.POINTER(FSMI_HandleType), FSMI_TopTableMatrixPhysical, FSMI_SFX, FSMI_TactileAudioBasedFeedbackEffects
        self.lib.ForceSeatMI_ActivateProfile.argtypes        = ctypes.POINTER(FSMI_HandleType), c_char_p
        self.lib.ForceSeatMI_SetAppID.argtypes               = ctypes.POINTER(FSMI_HandleType), c_char_p
        self.lib.ForceSeatMI_Park.argtypes                   = ctypes.POINTER(FSMI_HandleType), FSMI_UINT8

        self.api = self.lib.ForceSeatMI_Create()

    def delete(self):
        self.lib.ForceSeatMI_Delete(self.api)

    def begin_motion_control(self) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_BeginMotionControl(self.api))
        return (result.value == FSMI_True().value)

    def end_motion_control(self) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_EndMotionControl(self.api))
        return (result.value == FSMI_True().value)

    def get_platform_info_ex(self, platform_info: FSMI_PlatformInfo, platform_info_size: FSMI_UINT32, timeout: FSMI_UINT32) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_GetPlatformInfoEx(self.api, byref(platform_info), platform_info_size, timeout))
        return (result.value == FSMI_True().value)

    def send_telemetry_ace(self, telemetry: FSMI_TelemetryACE) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_SendTelemetryACE(self.api, telemetry))
        return (result.value == FSMI_True().value)

    def send_telemetry_ace2(self, telemetry: FSMI_TelemetryACE, sfx: FSMI_SFX, audio_effects: FSMI_TactileAudioBasedFeedbackEffects) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_SendTelemetry2(self.api, telemetry, sfx, audio_effects))
        return (result.value == FSMI_True().value)

    def send_top_table_pos_phy(self, position: FSMI_TopTablePositionPhysical) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_SendTopTablePosPhy(self.api, position))
        return (result.value == FSMI_True().value)

    def send_top_table_pos_phy2(self, position: FSMI_TopTablePositionPhysical, sfx: FSMI_SFX, audio_effects: FSMI_TactileAudioBasedFeedbackEffects) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_SendTopTablePosPhy2(self.api, position, sfx, audio_effects))
        return (result.value == FSMI_True().value)

    def send_top_table_matrix_phy(self, matrix: FSMI_TopTableMatrixPhysical) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_SendTopTableMatrixPhy(self.api, matrix))
        return (result.value == FSMI_True().value)

    def send_top_table_matrix_phy2(self, matrix: FSMI_TopTableMatrixPhysical, sfx: FSMI_SFX, audio_effects: FSMI_TactileAudioBasedFeedbackEffects) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_SendTopTableMatrixPhy2(self.api, matrix, sfx, audio_effects))
        return (result.value == FSMI_True().value)

    def activate_profile(self, profileName: str) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_ActivateProfile(self.api, profileName.encode('utf-8')))
        return (result.value == FSMI_True().value)

    def set_app_id(self, appId: str) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_SetAppID(self.api, appId.encode('utf-8')))
        return (result.value == FSMI_True().value)

    def park(self, parkMode: FSMI_UINT8) -> bool:
        result = FSMI_Bool(self.lib.ForceSeatMI_Park(self.api, parkMode))
        return (result.value == FSMI_True().value)
