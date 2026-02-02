import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSPresetProfiles
import copy


class ScanConverter(Node):
    def __init__(self):
        super().__init__('scan_converter')
        self.pub = self.create_publisher(LaserScan, '/scan', QoSPresetProfiles.SENSOR_DATA.value)
        self.sub = self.create_subscription(LaserScan, '/scan_raw', self.callback, QoSPresetProfiles.SENSOR_DATA.value)

    def callback(self, msg):
        new_msg = copy.deepcopy(msg)
        new_msg.header.frame_id = 'rplidar_c1'
        self.pub.publish(new_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
