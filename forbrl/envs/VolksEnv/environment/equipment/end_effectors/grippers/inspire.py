import logging
import time
from functools import wraps

import serial

from .base import GripperBase
from ..registry import END_EFFECTORS


def ignore_exception(logger, error=None, message=None):
    """
    Ignore the dummy data send error for inspire gripper
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error:
                logger.warning(message)

        return wrapper

    return decorate


TCP = (0, 0, 0.23, 0, 0, 0)


@END_EFFECTORS.register_module
class InspireGripper(GripperBase):
    """A python interface for an Inspire gripper. """
    logger = logging.getLogger(__name__)
    gripper_id = 1
    port_number = 115200

    def __init__(self, tcp=TCP, speed=1000, force=1000, openmax=1000, openmin=0,
                 wait_movement=True, usb_dir='/dev/ttyUSB0', port_number=None):
        self.tcp = list(tcp)
        self.speed = speed
        self.force = force
        self.openmax = openmax
        self.openmin = openmin
        self.wait_movement = wait_movement
        self.usb_dir = usb_dir
        if port_number is not None:
            self.port_number = port_number

        self.ser = serial.Serial(self.usb_dir, self.port_number)
        self.ser.timeout = 0.01
        self.ser.isOpen()

        self.setopenlimit(openmax, openmin)

        for i in range(1, 255):
            if self.getid(i) == 7:
                self.gripper_id = i
                break
        super().__init__()

    def __repr__(self):
        msg = (f"Inspire gripper on: '{self.usb_dir}' with port: "
               f"{self.port_number}")
        return msg

    @staticmethod
    def data2bytes(data):
        """ 把数据分成高字节和低字节"""
        rdata = [0xff] * 2
        if data == -1:
            rdata[0] = 0xff
            rdata[1] = 0xff
            return rdata
        rdata[0] = data & 0xff
        rdata[1] = (data >> 8) & 0xff
        return rdata

    @staticmethod
    def num2str(num):
        """把十六进制或十进制的数转成bytes"""
        code = hex(num)
        code = code[2:4]
        if len(code) == 1:
            code = '0' + code
        code = bytes.fromhex(code)
        return code

    @staticmethod
    def checknum(data, leng):
        """求校验和"""
        result = 0
        for i in range(2, leng):
            result += data[i]
        result = result & 0xff
        return result

    def getid(self, i):
        """扫描id号"""
        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # id号
        b[2] = i

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x12

        # 数据
        b[5] = self.data2bytes(1000)[0]
        b[6] = self.data2bytes(1000)[1]

        b[7] = self.data2bytes(0)[0]
        b[8] = self.data2bytes(0)[1]

        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：',putdata)
        time.sleep(0.01)
        getdata = self.ser.read(7)
        return len(getdata)

    def setopenlimit(self, openmax=None, openmin=None):
        if openmax is None:
            openmax = self.openmax
        if openmin is None:
            openmin = self.openmin
        """设置开口限位（最大开口度和最小开口度）"""
        assert 0 <= openmin < openmax <= 1000, 'gripper setting out of range'

        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x12

        # 数据
        b[5] = self.data2bytes(openmax)[0]
        b[6] = self.data2bytes(openmax)[1]

        b[7] = self.data2bytes(openmin)[0]
        b[8] = self.data2bytes(openmin)[1]

        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：',putdata)
        time.sleep(0.01)
        getdata = self.ser.read(7)
        self.send_receive_display(datanum, putdata, getdata)

    def setid(self, idnew):
        """设置ID"""
        assert 0 < idnew < 255, 'id out of range'

        datanum = 0x02
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x04

        # 数据
        b[5] = idnew

        self.gripper_id = idnew

        # 校验和
        b[6] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：',putdata)
        time.sleep(0.01)
        getdata = self.ser.read(7)
        self.send_receive_display(datanum, putdata, getdata)

    def move_to(self, tgt):
        """运动到目标"""
        assert 0 <= tgt <= 1000, 'target out of range'

        datanum = 0x03
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x54

        # 数据
        b[5] = self.data2bytes(tgt)[0]
        b[6] = self.data2bytes(tgt)[1]

        # 校验和
        b[7] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：',putdata)
        time.sleep(0.01)
        getdata = self.ser.read(7)
        self.send_receive_display(datanum, putdata, getdata)
        if self.wait_movement:
            self.wait_till_stop()

    def open(self, speed=None):
        """运动张开"""
        if speed is None:
            speed = self.speed

        assert 1 < speed <= 1000, 'setting out-of-range'

        datanum = 0x03
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x11

        # 数据
        b[5] = self.data2bytes(speed)[0]
        b[6] = self.data2bytes(speed)[1]

        # 校验和
        b[7] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：',putdata)
        time.sleep(0.01)
        getdata = self.ser.read(7)
        self.send_receive_display(datanum, putdata, getdata)
        if self.wait_movement:
            self.wait_till_stop()

    def close(self, speed=None, power=None):
        """运动闭合"""
        if speed is None:
            speed = self.speed
        if power is None:
            power = self.force
        assert 0 < speed <= 1000 and 50 <= power <= 1000, 'setting out-of-range'
        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x10

        # 数据
        b[5] = self.data2bytes(speed)[0]
        b[6] = self.data2bytes(speed)[1]
        b[7] = self.data2bytes(power)[0]
        b[8] = self.data2bytes(power)[1]
        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：',putdata)
        time.sleep(0.01)
        getdata = self.ser.read(7)
        self.send_receive_display(datanum, putdata, getdata)
        if self.wait_movement:
            self.wait_till_stop()

    def grip(self, speed=None, power=None):
        """运动持续闭合"""
        if speed is None:
            speed = self.speed
        if power is None:
            power = self.force
        assert 0 < speed <= 1000 and 50 <= power <= 1000, 'setting out-of-range'

        datanum = 0x05
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x18

        # 数据
        b[5] = self.data2bytes(speed)[0]
        b[6] = self.data2bytes(speed)[1]
        b[7] = self.data2bytes(power)[0]
        b[8] = self.data2bytes(power)[1]
        # 校验和
        b[9] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        # print('发送的数据：',putdata)
        time.sleep(0.05)
        getdata = self.ser.read(7)
        self.send_receive_display(datanum, putdata, getdata)
        if self.wait_movement:
            self.wait_till_stop()

    def getopenlimit(self, ):
        """读取开口限位"""

        datanum = 0x01
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # gripper_id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x13

        # 校验和
        b[5] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''
        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        time.sleep(0.05)
        getdata = self.ser.read(10)
        self.send_receive_display(datanum, putdata, getdata)

        openlimit = [0] * 2
        for i in range(1, 3):
            if getdata[i * 2 + 3] == 0xff and getdata[i * 2 + 4] == 0xff:
                openlimit[i - 1] = -1
            else:
                openlimit[i - 1] = getdata[i * 2 + 3] + (
                        getdata[i * 2 + 4] << 8)
        return openlimit

    def getcopen(self, ):
        """读取当前开口"""
        for i in range(5):
            self.ser.read(13)

        datanum = 0x01
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # gripper_id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0xD9

        # 校验和
        b[5] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''
        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        time.sleep(0.05)
        getdata = self.ser.read(8)
        self.send_receive_display(datanum, putdata, getdata)

        copen = [0] * 1

        try:
            for i in range(1, 2):
                if getdata[i * 2 + 3] == 0xff and getdata[i * 2 + 4] == 0xff:
                    copen[i - 1] = -1
                else:
                    copen[i - 1] = getdata[i * 2 + 3] + (
                            getdata[i * 2 + 4] << 8)
            return copen
        except:
            return -1, None

    def getstate(self, ):
        """读取当前状态"""

        datanum = 0x01
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90

        # gripper_id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x41

        # 校验和
        b[5] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''
        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        time.sleep(0.05)
        getdata = self.ser.read(13)
        self.show_state(getdata)
        return getdata

    @ignore_exception(logger,
                      error=IndexError,
                      message=("Incomplete data received, couldn't unpack "
                               "state info."))
    def show_state(self, getdata):
        msg = f"\n{' ' * 4}Gripper state:\n{' ' * 8}"
        if getdata[5] == 1:
            msg += "max in place"
        elif getdata[5] == 2:
            msg += "min in place"
        elif getdata[5] == 3:
            msg += "stop in place"
        elif getdata[5] == 4:
            msg += 'closing'
        elif getdata[5] == 5:
            msg += 'openning'
        elif getdata[5] == 6:
            msg += "force control in place to stop"
        else:
            msg += "Invalid response"

        if (getdata[6] & 0x01) == 1:
            msg += f"\n{' ' * 8}Runing stop fault"

        if (getdata[6] & 0x02) == 2:
            msg += f"\n{' ' * 8}Overheat fault"

        if (getdata[6] & 0x04) == 4:
            msg += f"\n{' ' * 8}Over Current Fault"

        if (getdata[6] & 0x08) == 8:
            msg += f"\n{' ' * 8}Running fault"

        if (getdata[6] & 0x10) == 16:
            msg += f"\n{' ' * 8}Communication fault"

        msg += f"\n{' ' * 4}temp: {getdata[7]}"
        msg += (f"\n{' ' * 4}curopen: "
                f"{((getdata[9] << 8) & 0xff00) + getdata[8]}")
        msg += (f"\n{' ' * 4}power: "
                f"{((getdata[11] << 8) & 0xff00) + getdata[10]}")
        self.logger.debug(msg)

    @ignore_exception(logger,
                      error=IndexError,
                      message=("Incomplete data received, couldn't unpack "
                               "received info."))
    def send_receive_display(self, datanum, putdata, getdata=None):
        msg = f"Sent data:"
        for i in range(1, datanum + 6):
            msg += f"\n{' ' * 4}{hex(putdata[i - 1])} "
        if getdata is not None:
            msg += f"\nReceived data:"
            for i in range(1, 8):
                msg += f"\n{' ' * 4}{hex(getdata[i - 1])} "
        self.logger.debug(msg)

    def setestop(self, ):
        """急停"""

        datanum = 0x01
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x16

        # 校验和
        b[5] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        time.sleep(0.01)
        getdata = self.ser.read(7)

        self.send_receive_display(datanum, putdata, getdata)

    def setparam(self, ):
        """参数固化"""

        datanum = 0x01
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x01

        # 校验和
        b[5] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        time.sleep(0.01)
        getdata = self.ser.read(7)
        self.send_receive_display(datanum, putdata, getdata)

    def setfrsvd(self, ):
        """清除故障"""

        datanum = 0x01
        b = [0] * (datanum + 5)
        # 包头
        b[0] = 0xEB
        b[1] = 0x90
        # id号
        b[2] = self.gripper_id

        # 数据个数
        b[3] = datanum

        # 操作码
        b[4] = 0x17

        # 校验和
        b[5] = self.checknum(b, datanum + 4)

        # 向串口发送数据
        putdata = b''

        for i in range(1, datanum + 6):
            putdata = putdata + self.num2str(b[i - 1])
        self.ser.write(putdata)
        self.send_receive_display(datanum, putdata)

    def shutdown(self):
        self.ser.close()
        self.logger.debug(f"{self.__repr__()} shut down")
        return True

    @ignore_exception(logger,
                      error=IndexError,
                      message=("Incomplete data received, couldn't unpack "
                               "received info."))
    def wait_till_stop(self, gap=0.2, window=7, max_duration=2, reload=5):
        start = time.time()
        states = [-1] * window
        pointer = 0
        while True:
            time.sleep(gap)
            state = self.getstate()
            for i in range(reload):
                if len(state) > 5:
                    break
                time.sleep(gap * 2)
                state = self.getstate()
            else:
                self.ser = serial.Serial(self.usb_dir, self.port_number)
                self.ser.timeout = 0.01
                self.ser.isOpen()

                self.setopenlimit(self.openmax, self.openmin)

                for i in range(1, 255):
                    if self.getid(i) == 7:
                        self.gripper_id = i
                        break

            states[pointer] = state[5]

            if max(set(states), key=states.count) not in [4, 5, -1]:
                break
            if time.time() - start > max_duration:
                break
            pointer = (pointer + 1) % window
