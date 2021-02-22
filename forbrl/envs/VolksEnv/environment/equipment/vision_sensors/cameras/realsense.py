import json
import logging
import time

import numpy as np
import pyrealsense2 as rs

from .base import CamBase
from ..registry import VISION_SENSORS

DS5_PRODUCT_IDS = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE",
                   "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A"]


@VISION_SENSORS.register_module
class RealsenseCam(CamBase):
    """Python interface for Intel Realsense Family cameras."""
    # TODO: consider using decorator for patterns in functions such as
    #  'restart', 'capture' and 'update_device' if necessary
    logger = logging.getLogger(__name__)

    def __init__(self,
                 color_res=(1920, 1080),
                 color_fr=30,
                 depth_res=(1280, 720),
                 depth_fr=30,
                 serial_number=None,
                 reset_delay=3,
                 timeout=50,
                 preset=None):
        self._align = None
        self._serial_number = None
        self._depth_scale = None
        self._pipeline = None
        self.config = None
        self.device = None
        self.device_cat = None

        self.color_res = color_res
        self.color_fr = color_fr
        self.depth_res = depth_res
        self.depth_fr = depth_fr
        self.reset_delay = reset_delay
        self.timeout = timeout

        if serial_number is not None:
            self.serial_number = serial_number

        self.init_pipeline()
        self.update_device()

        if preset is not None:
            self._pipeline = rs.pipeline()
            self.load_preset(preset)

        super().__init__()

    def load_preset(self, preset_dir):
        if not self.check_product_id():
            self.logger.warning(f"{self.__repr__()} might not support "
                                f"advanced mode that allows to load "
                                f"preset files. Skipping")
        else:
            self.activate_advanced_mode()
            with open(preset_dir) as f:
                preset_json = json.load(f)
            advnc_mode = rs.rs400_advanced_mode(self.device)
            json_string = str(preset_json).replace("'", '\"')
            advnc_mode.load_json(json_string)
            self.logger.debug(f"{self.__repr__()} loaded preset from "
                              f"json file at {preset_dir}")

    def activate_advanced_mode(self):
        advnc_mode = rs.rs400_advanced_mode(self.device)
        if advnc_mode.is_enabled():
            self.logger.debug(f"Advanced mode for {self.__repr__()} is "
                              f"already enabled")
        else:
            self.logger.debug(f"Advanced mode for {self.__repr__()} is not "
                              f"enabled, trying to enable")
            for _ in range(self.timeout):
                advnc_mode.toggle_advanced_mode(True)
                self.logger.debug(f"Rebooting {self.__repr__()} for advanced "
                                  f"mode")
                time.sleep(self.reset_delay)
                self.update_device()
                advnc_mode = rs.rs400_advanced_mode(self.device)
                if advnc_mode.is_enabled():
                    self.logger.debug(f"Advanced mode for {self.__repr__()} is "
                                      f"now enabled")
                    break
            else:
                self.logger.warning(f"Failed to enable advanced mode for "
                                    f"{self.__repr__()}, skipping")

    def check_product_id(self):
        if self.device.supports(rs.camera_info.product_id):
            p_id = str(self.device.get_info(rs.camera_info.product_id))
            if p_id in DS5_PRODUCT_IDS:
                return True
        return False

    def __repr__(self):
        msg = (f"'{self.device_cat}' camera with serial number:"
               f" {self.serial_number}")
        return msg

    def init_pipeline(self):
        self._pipeline = rs.pipeline()
        self.config = rs.config()
        if self.serial_number is not None:
            self.config.enable_device(self.serial_number)

        dw, dh = self.depth_res
        cw, ch = self.color_res

        self.config.enable_stream(rs.stream.depth, dw, dh,
                                  rs.format.z16, self.depth_fr)
        self.config.enable_stream(rs.stream.color, cw, ch,
                                  rs.format.bgr8, self.color_fr)

    def update_device_(self):
        devices = rs.context().devices
        if self.serial_number is None:
            device = devices[0]
            device_cat = device.get_info(rs.camera_info.name)
            sn = device.get_info(rs.camera_info.serial_number)
            self.logger.warning(f"Serial number not specified, using first "
                                f"device: \n    {device_cat}(serial "
                                f"number: {sn})")
            self.device = device
        else:
            for device in devices:
                device_sn = device.get_info(rs.camera_info.serial_number)
                if device_sn == self.serial_number:
                    self.device = device
                    break
            else:
                msg = (f"Device with serial number: {self.serial_number} not "
                       f"connected. ")
                if len(devices) == 0:
                    msg = f"No device connected. Please check connection:{msg}"
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                msg += f"\n{' ' * 4}Detected devices are:"
                for idx, device in enumerate(devices):
                    device_cat = device.get_info(rs.camera_info.name)
                    sn = device.get_info(rs.camera_info.serial_number)
                    msg += f"\n{' ' * 8}{idx}. {device_cat} - S/N:{sn}"

                self.logger.error(msg)
                raise RuntimeError(msg)
        self.device_cat = self.device.get_info(rs.camera_info.name)
        return devices

    def update_device(self):
        for i in range(self.timeout):
            try:
                return self.update_device_()
            except RuntimeError as e:
                if "hwmon command 0x10 failed." in f'{e}':
                    self.logger.warning(f"Could not update device list at {i} "
                                        f"try.")
        else:
            msg = f"Could not update device list in {self.timeout} tries."
            self.logger.error(msg)
            raise TimeoutError(msg)

    def hardware_reset(self):
        self.update_device()
        self.device.hardware_reset()
        time.sleep(self.reset_delay)

    def start_(self):
        profile = self._pipeline.start(self.config)
        stream = profile.get_stream(rs.stream.color)
        self._intrinsics = stream.as_video_stream_profile().get_intrinsics()

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self._align = rs.align(rs.stream.color)
        self.logger.debug(f"{self.device_cat} with serial number: "
                          f"{self.serial_number} start streaming.")

    def restart(self):
        self.hardware_reset()
        self.init_pipeline()
        try:
            self.start_()
        except RuntimeError as e:
            self.logger.error(f"Reboot device with serial number:"
                              f" {self.serial_number} failed.")
            raise e
        self.logger.debug(f"Successfully rebooted device\n {self.device_cat} "
                          f"with serial number: {self.serial_number} rebooted "
                          f", now start streaming.")

    def start(self):
        # Start streaming
        try:
            self.start_()
        except RuntimeError as e:
            orig_msg = f'{e}'
            if "No device connected" in orig_msg:
                devices = self.update_device()
                msg = (f"{e}, Please check:\n{' ' * 4}"
                       f"1. Devices are connected.\n{' ' * 4}"
                       f"2. Device is not in use.\n{' ' * 4}"
                       f"3. Serial number: {self.serial_number} matches the "
                       f"intended device.\n{' ' * 4}"
                       f"Detected devices: \n{' ' * 8}")

                for idx, device in enumerate(devices):
                    device_cat = device.get_info(rs.camera_info.name)
                    sn = device.get_info(rs.camera_info.serial_number)
                    msg += f"{idx}. {device_cat} - S/N:{sn}\n{' ' * 12}"

                if len(devices) == 0:
                    msg += 'None'

                self.logger.error(msg)
                raise RuntimeError(msg)
            elif "Device or resource busy" in orig_msg:
                msg = (f"{e}:\n    "
                       f"{self.device_cat} with serial number: "
                       f"{self.serial_number} might already be streaming, "
                       f"trying to reset the hardware... please wait for "
                       f"{self.reset_delay} seconds")
                self.logger.warning(msg)
                self.restart()
            else:
                raise e

    def capture_(self):
        frames = self._pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = self._align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        for _ in [aligned_depth_frame, color_frame]:
            assert _ is not None, "Didn't get intended frame"
        depth_img = (np.asanyarray(aligned_depth_frame.get_data()) *
                     self.depth_scale)
        color_img = np.asanyarray(color_frame.get_data())
        self.logger.debug(f"Frame captured!\n With color image of size "
                          f"{color_img.shape}, depth image of size"
                          f" {depth_img.shape}")
        return color_img, depth_img

    def capture(self):
        for _ in range(self.timeout):
            try:
                return self.capture_()
            except AssertionError as e:
                self.logger.warning(f"{e}, trying again, retried {_} times")
        else:
            msg = (f"Could not get complete frame in {self.timeout} "
                   f"tries.")
            self.logger.error(msg)
            raise TimeoutError(msg)

    def stop(self):
        self.hardware_reset()
        self._pipeline.stop()

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def depth_scale(self):
        return self._depth_scale

    @property
    def serial_number(self):
        return self._serial_number

    @serial_number.setter
    def serial_number(self, val):
        if not isinstance(val, str):
            val = str(val)

        if not val.isdecimal():
            raise ValueError(f"Serial number should be formed only by numbers, "
                             f"while '{val}' provided")
        if len(val) != 12:
            raise ValueError(f"Serial number should be of length 12, while "
                             f"'{val}' of length {len(val)} provided")

        self._serial_number = val
