# Copyright 2021 AUTHORS
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the AUTHORS nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import json
from math import sqrt
import struct
import sys
from time import sleep

from bno055 import registers
from bno055.connectors.Connector import Connector
from bno055.params.NodeParameters import NodeParameters

from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Imu, MagneticField, Temperature
from std_msgs.msg import String
from example_interfaces.srv import Trigger


class SensorService:
    """Provide an interface for accessing the sensor's features & data."""

    def __init__(self, node: Node, connector: Connector, param: NodeParameters):
        self.node = node
        self.con = connector
        self.param = param

        prefix = self.param.ros_topic_prefix.value
        QoSProf = QoSProfile(depth=10)

        # create topic publishers:
        self.pub_imu_raw = node.create_publisher(Imu, prefix + 'imu_raw', QoSProf)
        self.pub_imu = node.create_publisher(Imu, prefix + 'imu', QoSProf)
        self.pub_mag = node.create_publisher(MagneticField, prefix + 'mag', QoSProf)
        self.pub_grav = node.create_publisher(Vector3, prefix + 'grav', QoSProf)
        self.pub_temp = node.create_publisher(Temperature, prefix + 'temp', QoSProf)
        self.pub_calib_status = node.create_publisher(String, prefix + 'calib_status', QoSProf)
        self.srv = self.node.create_service(Trigger, prefix + 'calibration_request', self.calibration_request_callback)

        # initialize message objects to reuse for publishing (avoid creating new objects every time):
        self._imu_raw_msg = Imu()
        self._imu_msg = Imu()
        self._mag_msg = MagneticField()
        self._grav_msg = Vector3()
        self._temp_msg = Temperature()

        # precompute covariance matrices from parameters (avoid recomputing every time):
        self._cov_ori = [
            self.param.variance_orientation.value[0], 0.0, 0.0,
            0.0, self.param.variance_orientation.value[1], 0.0,
            0.0, 0.0, self.param.variance_orientation.value[2],
        ]
        self._cov_acc = [
            self.param.variance_acc.value[0], 0.0, 0.0,
            0.0, self.param.variance_acc.value[1], 0.0,
            0.0, 0.0, self.param.variance_acc.value[2],
        ]
        self._cov_gyr = [
            self.param.variance_angular_vel.value[0], 0.0, 0.0,
            0.0, self.param.variance_angular_vel.value[1], 0.0,
            0.0, 0.0, self.param.variance_angular_vel.value[2],
        ]
        self._cov_mag = [
            self.param.variance_mag.value[0], 0.0, 0.0,
            0.0, self.param.variance_mag.value[1], 0.0,
            0.0, 0.0, self.param.variance_mag.value[2],
        ]
        #
        # The sensor reading often contains invalid large jumps of the orientation w component.
        # To mitigate this, we implement a simple jump detection and filtering logic.
        # Publish only if the reading is valid according to jump detection logic
        #

        # Jump detection state variable:
        self.prev_norm = None            # last computed raw quaternion norm

    def configure(self):
        """Configure the IMU sensor hardware."""
        self.node.get_logger().info('Configuring device...')
        try:
            data = self.con.receive(registers.BNO055_CHIP_ID_ADDR, 1)
            if data[0] != registers.BNO055_ID:
                raise IOError('Device ID=%s is incorrect' % data)
            # print("device sent ", binascii.hexlify(data))
        except Exception as e:  # noqa: B902
            # This is the first communication - exit if it does not work
            self.node.get_logger().error('Communication error: %s' % e)
            self.node.get_logger().error('Shutting down ROS node...')
            sys.exit(1)

        # IMU connected => apply IMU Configuration:
        if not (self.con.transmit(registers.BNO055_OPR_MODE_ADDR, 1, bytes([registers.OPERATION_MODE_CONFIG]))):
            self.node.get_logger().warn('Unable to set IMU into config mode.')

        if not (self.con.transmit(registers.BNO055_PWR_MODE_ADDR, 1, bytes([registers.POWER_MODE_NORMAL]))):
            self.node.get_logger().warn('Unable to set IMU normal power mode.')

        if not (self.con.transmit(registers.BNO055_PAGE_ID_ADDR, 1, bytes([0x00]))):
            self.node.get_logger().warn('Unable to set IMU register page 0.')

        if not (self.con.transmit(registers.BNO055_SYS_TRIGGER_ADDR, 1, bytes([0x00]))):
            self.node.get_logger().warn('Unable to start IMU.')

        if not (self.con.transmit(registers.BNO055_UNIT_SEL_ADDR, 1, bytes([0x83]))):
            self.node.get_logger().warn('Unable to set IMU units.')

        # The sensor placement configuration (Axis remapping) defines the
        # position and orientation of the sensor mount.
        # See also Bosch BNO055 datasheet section Axis Remap
        mount_positions = {
            'P0': bytes(b'\x21\x04'),
            'P1': bytes(b'\x24\x00'),
            'P2': bytes(b'\x24\x06'),
            'P3': bytes(b'\x21\x02'),
            'P4': bytes(b'\x24\x03'),
            'P5': bytes(b'\x21\x02'),
            'P6': bytes(b'\x21\x07'),
            'P7': bytes(b'\x24\x05')
        }
        if not (self.con.transmit(registers.BNO055_AXIS_MAP_CONFIG_ADDR, 2,
                                  mount_positions[self.param.placement_axis_remap.value])):
            self.node.get_logger().warn('Unable to set sensor placement configuration.')

        # Show the current sensor offsets
        self.node.get_logger().info('Current sensor offsets:')
        self.print_calib_data()
        if self.param.set_offsets.value:
            configured_offsets = \
                self.set_calib_offsets(
                    self.param.offset_acc,
                    self.param.offset_mag,
                    self.param.offset_gyr,
                    self.param.radius_mag,
                    self.param.radius_acc)
            if configured_offsets:
                self.node.get_logger().info('Successfully configured sensor offsets to:')
                self.print_calib_data()
            else:
                self.node.get_logger().warn('setting offsets failed')


        # Set Device mode
        device_mode = self.param.operation_mode.value
        self.node.get_logger().info(f"Setting device_mode to {device_mode}")

        if not (self.con.transmit(registers.BNO055_OPR_MODE_ADDR, 1, bytes([device_mode]))):
            self.node.get_logger().warn('Unable to set IMU operation mode into operation mode.')

        self.node.get_logger().info('Bosch BNO055 IMU configuration complete.')


    def get_sensor_data(self):
        """Read IMU data from the sensor, parse and publish."""

        # read from sensor: 45 bytes starting from BNO055_ACCEL_DATA_X_LSB_ADDR
        buf = self.con.receive(registers.BNO055_ACCEL_DATA_X_LSB_ADDR, 45)

        if not buf or len(buf) != 45:
            self.node.get_logger().warn(f"Short read: got {len(buf) if buf else 0} bytes")
            return

        # Use bytes/memoryview for fast unpack
        b = bytes(buf)

        # helper: little-endian int16, which matches the sensor data format
        def i16(off: int) -> int:
            return struct.unpack_from("<h", b, off)[0]

        if self.param.operation_mode.value in [0x0B, 0x0C]:  # only FMC_OFF or FMC_ON modes provide fused orientation data
            # Quaternion:
            q = [
                float(i16(26)),  # x
                float(i16(28)),  # y
                float(i16(30)),  # z
                float(i16(24)),  # w
            ]

            #
            # Sanity check - compute quaternion norm and see if it is valid
            #

            """
            # Alternative approach to quaternion sanity check, needs import np:
            # After reading q_raw = np.array([x,y,z,w], dtype=float)
            if not np.all(np.isfinite(q_raw)):
                warn and return

            if np.all(q_raw == 0):
                warn and return

            norm = np.linalg.norm(q_raw)
            if norm < 1e-6:
                warn and return

            q = q_raw / norm

            # Optional: continuity check
            if self.prev_q is not None:
                # Quaternions have sign ambiguity: pick closest
                if np.dot(self.prev_q, q) < 0:
                    q = -q
                # Now you can check angle jump if you want
            self.prev_q = q
            """

            # Compute norm safely, return if anything wrong:
            norm = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
            if abs(norm - 16384.0) > 1000.0:
                # abnormal norm - invalid quaternion. It should be usually ~16384, as values are large.
                self.node.get_logger().warn("Invalid quaternion norm: {} — sensor reading ignored".format(norm))
                return
            else:
                q = [x / norm for x in q]
                if self.prev_norm is None: # first good value
                    self.prev_norm = norm

            if self.prev_norm is not None:
                norm_jump = norm - self.prev_norm
                if abs(norm_jump) > self.prev_norm * 0.05:  # 5% jump
                    self.node.get_logger().warn("Large jump in quaternion norm detected: norm: {}  prev: {}  jump: {}".format(norm, self.prev_norm, norm_jump))
                    return
                else:
                    self.prev_norm = norm   # first good reading
        else:
            # In other modes, we do not have fused orientation data, so we skip the quaternion sanity check
            q = [0.0, 0.0, 0.0, 1.0]  # default orientation (no rotation)

        # OK, sanity check passed, we are good to publish the data

        self._imu_msg.orientation.x, self._imu_msg.orientation.y, self._imu_msg.orientation.z, self._imu_msg.orientation.w = q

        # locals (avoid repeated attribute lookups)
        now_msg = self.node.get_clock().now().to_msg()
        frame_id = self.param.frame_id.value

        # Headers (single timestamp)
        self._imu_raw_msg.header.stamp = now_msg
        self._imu_raw_msg.header.frame_id = frame_id
        self._imu_msg.header.stamp = now_msg
        self._imu_msg.header.frame_id = frame_id
        self._mag_msg.header.stamp = now_msg
        self._mag_msg.header.frame_id = frame_id
        self._temp_msg.header.stamp = now_msg
        self._temp_msg.header.frame_id = frame_id

        # dividers to convert raw sensor values to physical units, from driver parameters
        acc_div = self.param.acc_factor.value
        gyr_div = self.param.gyr_factor.value
        mag_div = self.param.mag_factor.value
        grav_div = self.param.grav_factor.value

        # raw accelerometer data
        ax_raw, ay_raw, az_raw = i16(0), i16(2), i16(4)
        self._imu_raw_msg.linear_acceleration.x = ax_raw / acc_div
        self._imu_raw_msg.linear_acceleration.y = ay_raw / acc_div
        self._imu_raw_msg.linear_acceleration.z = az_raw / acc_div

        # raw magnetometer data
        mx_raw, my_raw, mz_raw = i16(6), i16(8), i16(10)
        self._mag_msg.magnetic_field.x = mx_raw / mag_div  # 16 million LSB per Tesla, as per datasheet
        self._mag_msg.magnetic_field.y = my_raw / mag_div
        self._mag_msg.magnetic_field.z = mz_raw / mag_div

        # raw gyroscope data
        gx_raw, gy_raw, gz_raw = i16(12), i16(14), i16(16)   # Decode once for both raw and filtered gyro data
        self._imu_raw_msg.angular_velocity.x = self._imu_msg.angular_velocity.x = gx_raw / gyr_div
        self._imu_raw_msg.angular_velocity.y = self._imu_msg.angular_velocity.y = gy_raw / gyr_div
        self._imu_raw_msg.angular_velocity.z = self._imu_msg.angular_velocity.z = gz_raw / gyr_div

        # “filtered/linear accel” block
        lax_raw, lay_raw, laz_raw = i16(32), i16(34), i16(36)
        self._imu_msg.linear_acceleration.x = lax_raw / acc_div
        self._imu_msg.linear_acceleration.y = lay_raw / acc_div
        self._imu_msg.linear_acceleration.z = laz_raw / acc_div

        # gravity block
        grx_raw, gry_raw, grz_raw = i16(38), i16(40), i16(42)
        self._grav_msg.x = grx_raw / grav_div
        self._grav_msg.y = gry_raw / grav_div
        self._grav_msg.z = grz_raw / grav_div

        # temperature - one byte:
        temp_raw = b[44]
        self._temp_msg.temperature = float(temp_raw)

        self._imu_raw_msg.orientation_covariance = self._imu_msg.orientation_covariance = self._cov_ori
        self._imu_raw_msg.linear_acceleration_covariance = self._imu_msg.linear_acceleration_covariance = self._cov_acc
        self._imu_raw_msg.angular_velocity_covariance = self._imu_msg.angular_velocity_covariance = self._cov_gyr
        self._mag_msg.magnetic_field_covariance = self._cov_mag

        # TODO: make some of this an option to publish?
        self.pub_imu_raw.publish(self._imu_raw_msg)
        self.pub_imu.publish(self._imu_msg)
        self.pub_mag.publish(self._mag_msg)
        self.pub_grav.publish(self._grav_msg)
        self.pub_temp.publish(self._temp_msg)


    def get_calib_status(self):
        """
        Read calibration status for sys/gyro/acc/mag.

        Quality scale: 0 = bad, 3 = best
        """
        calib_status = self.con.receive(registers.BNO055_CALIB_STAT_ADDR, 1)
        sys = (calib_status[0] >> 6) & 0x03
        gyro = (calib_status[0] >> 4) & 0x03
        accel = (calib_status[0] >> 2) & 0x03
        mag = calib_status[0] & 0x03

        # Create dictionary (map) and convert it to JSON string:
        calib_status_dict = {'sys': sys, 'gyro': gyro, 'accel': accel, 'mag': mag}
        calib_status_str = String()
        calib_status_str.data = json.dumps(calib_status_dict)

        # Publish via ROS topic:
        self.pub_calib_status.publish(calib_status_str)

    def get_calib_data(self):
        """Read all calibration data."""

        accel_offset_read = self.con.receive(registers.ACCEL_OFFSET_X_LSB_ADDR, 6)
        accel_offset_read_x = (accel_offset_read[1] << 8) | accel_offset_read[
            0]  # Combine MSB and LSB registers into one decimal
        accel_offset_read_y = (accel_offset_read[3] << 8) | accel_offset_read[
            2]  # Combine MSB and LSB registers into one decimal
        accel_offset_read_z = (accel_offset_read[5] << 8) | accel_offset_read[
            4]  # Combine MSB and LSB registers into one decimal

        accel_radius_read = self.con.receive(registers.ACCEL_RADIUS_LSB_ADDR, 2)
        accel_radius_read_value = (accel_radius_read[1] << 8) | accel_radius_read[0]

        mag_offset_read = self.con.receive(registers.MAG_OFFSET_X_LSB_ADDR, 6)
        mag_offset_read_x = (mag_offset_read[1] << 8) | mag_offset_read[
            0]  # Combine MSB and LSB registers into one decimal
        mag_offset_read_y = (mag_offset_read[3] << 8) | mag_offset_read[
            2]  # Combine MSB and LSB registers into one decimal
        mag_offset_read_z = (mag_offset_read[5] << 8) | mag_offset_read[
            4]  # Combine MSB and LSB registers into one decimal

        mag_radius_read = self.con.receive(registers.MAG_RADIUS_LSB_ADDR, 2)
        mag_radius_read_value = (mag_radius_read[1] << 8) | mag_radius_read[0]

        gyro_offset_read = self.con.receive(registers.GYRO_OFFSET_X_LSB_ADDR, 6)
        gyro_offset_read_x = (gyro_offset_read[1] << 8) | gyro_offset_read[
            0]  # Combine MSB and LSB registers into one decimal
        gyro_offset_read_y = (gyro_offset_read[3] << 8) | gyro_offset_read[
            2]  # Combine MSB and LSB registers into one decimal
        gyro_offset_read_z = (gyro_offset_read[5] << 8) | gyro_offset_read[
            4]  # Combine MSB and LSB registers into one decimal

        calib_data = {'accel_offset': {'x': accel_offset_read_x, 'y': accel_offset_read_y, 'z': accel_offset_read_z}, 'accel_radius': accel_radius_read_value,
                      'mag_offset': {'x': mag_offset_read_x, 'y': mag_offset_read_y, 'z': mag_offset_read_z}, 'mag_radius': mag_radius_read_value,
                      'gyro_offset': {'x': gyro_offset_read_x, 'y': gyro_offset_read_y, 'z': gyro_offset_read_z}}

        return calib_data

    def print_calib_data(self):
        """Read all calibration data and print to screen."""
        calib_data = self.get_calib_data()
        self.node.get_logger().info(
            '\tAccel offsets (x y z): %d %d %d' % (
                calib_data['accel_offset']['x'],
                calib_data['accel_offset']['y'],
                calib_data['accel_offset']['z']))

        self.node.get_logger().info(
            '\tAccel radius: %d' % (
                calib_data['accel_radius'],
            )
        )

        self.node.get_logger().info(
            '\tMag offsets (x y z): %d %d %d' % (
                calib_data['mag_offset']['x'],
                calib_data['mag_offset']['y'],
                calib_data['mag_offset']['z']))

        self.node.get_logger().info(
            '\tMag radius: %d' % (
                calib_data['mag_radius'],
            )
        )

        self.node.get_logger().info(
            '\tGyro offsets (x y z): %d %d %d' % (
                calib_data['gyro_offset']['x'],
                calib_data['gyro_offset']['y'],
                calib_data['gyro_offset']['z']))

    def set_calib_offsets(self, acc_offset, mag_offset, gyr_offset, mag_radius, acc_radius):
        """
        Write calibration data (define as 16 bit signed hex).

        :param acc_offset:
        :param mag_offset:
        :param gyr_offset:
        :param mag_radius:
        :param acc_radius:
        """
        # Must switch to config mode to write out
        if not (self.con.transmit(registers.BNO055_OPR_MODE_ADDR, 1, bytes([registers.OPERATION_MODE_CONFIG]))):
            self.node.get_logger().error('Unable to set IMU into config mode')
        sleep(0.025)

        # Seems to only work when writing 1 register at a time
        try:
            self.con.transmit(registers.ACCEL_OFFSET_X_LSB_ADDR, 1, bytes([acc_offset.value[0] & 0xFF]))
            self.con.transmit(registers.ACCEL_OFFSET_X_MSB_ADDR, 1, bytes([(acc_offset.value[0] >> 8) & 0xFF]))
            self.con.transmit(registers.ACCEL_OFFSET_Y_LSB_ADDR, 1, bytes([acc_offset.value[1] & 0xFF]))
            self.con.transmit(registers.ACCEL_OFFSET_Y_MSB_ADDR, 1, bytes([(acc_offset.value[1] >> 8) & 0xFF]))
            self.con.transmit(registers.ACCEL_OFFSET_Z_LSB_ADDR, 1, bytes([acc_offset.value[2] & 0xFF]))
            self.con.transmit(registers.ACCEL_OFFSET_Z_MSB_ADDR, 1, bytes([(acc_offset.value[2] >> 8) & 0xFF]))

            self.con.transmit(registers.ACCEL_RADIUS_LSB_ADDR, 1, bytes([acc_radius.value & 0xFF]))
            self.con.transmit(registers.ACCEL_RADIUS_MSB_ADDR, 1, bytes([(acc_radius.value >> 8) & 0xFF]))

            self.con.transmit(registers.MAG_OFFSET_X_LSB_ADDR, 1, bytes([mag_offset.value[0] & 0xFF]))
            self.con.transmit(registers.MAG_OFFSET_X_MSB_ADDR, 1, bytes([(mag_offset.value[0] >> 8) & 0xFF]))
            self.con.transmit(registers.MAG_OFFSET_Y_LSB_ADDR, 1, bytes([mag_offset.value[1] & 0xFF]))
            self.con.transmit(registers.MAG_OFFSET_Y_MSB_ADDR, 1, bytes([(mag_offset.value[1] >> 8) & 0xFF]))
            self.con.transmit(registers.MAG_OFFSET_Z_LSB_ADDR, 1, bytes([mag_offset.value[2] & 0xFF]))
            self.con.transmit(registers.MAG_OFFSET_Z_MSB_ADDR, 1, bytes([(mag_offset.value[2] >> 8) & 0xFF]))

            self.con.transmit(registers.MAG_RADIUS_LSB_ADDR, 1, bytes([mag_radius.value & 0xFF]))
            self.con.transmit(registers.MAG_RADIUS_MSB_ADDR, 1, bytes([(mag_radius.value >> 8) & 0xFF]))

            self.con.transmit(registers.GYRO_OFFSET_X_LSB_ADDR, 1, bytes([gyr_offset.value[0] & 0xFF]))
            self.con.transmit(registers.GYRO_OFFSET_X_MSB_ADDR, 1, bytes([(gyr_offset.value[0] >> 8) & 0xFF]))
            self.con.transmit(registers.GYRO_OFFSET_Y_LSB_ADDR, 1, bytes([gyr_offset.value[1] & 0xFF]))
            self.con.transmit(registers.GYRO_OFFSET_Y_MSB_ADDR, 1, bytes([(gyr_offset.value[1] >> 8) & 0xFF]))
            self.con.transmit(registers.GYRO_OFFSET_Z_LSB_ADDR, 1, bytes([gyr_offset.value[2] & 0xFF]))
            self.con.transmit(registers.GYRO_OFFSET_Z_MSB_ADDR, 1, bytes([(gyr_offset.value[2] >> 8) & 0xFF]))

            return True
        except Exception:  # noqa: B902
            return False

    def calibration_request_callback(self, request, response):
        if not (self.con.transmit(registers.BNO055_OPR_MODE_ADDR, 1, bytes([registers.OPERATION_MODE_CONFIG]))):
            self.node.get_logger().warn('Unable to set IMU into config mode.')
        sleep(0.025)
        calib_data = self.get_calib_data()
        if not (self.con.transmit(registers.BNO055_OPR_MODE_ADDR, 1, bytes([registers.OPERATION_MODE_NDOF]))):
            self.node.get_logger().warn('Unable to set IMU operation mode into operation mode.')
        response.success = True
        response.message = str(calib_data)
        return response
