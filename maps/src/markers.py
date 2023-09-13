import rospy
from visualization_msgs.msg import Marker
import csv

def create_marker(marker_id, x, y, value):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.id = marker_id

    # Store the value in the namespace field
    marker.ns = str(value)

    return marker

def save_markers_to_csv(markers, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'x_points', 'y_points', 'density'])
        for marker in markers:
            row = [marker.id, marker.pose.position.x, marker.pose.position.y, marker.ns]
            writer.writerow(row)

def publish_markers():
    rospy.init_node('marker_node')
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    markers = []
    markers.append(create_marker(0, 74.0, 107.0, 10))
    markers.append(create_marker(1, 42.0, 94.0, 20))
    markers.append(create_marker(2, 18.0, 85.0, 30))
    markers.append(create_marker(3, 66.0, 95.0, 40))
    markers.append(create_marker(4, 0.0, 0.0, 40))
    markers.append(create_marker(5, 5.0, 5.0, 40))
    #markers.append(create_marker(6, -5.0, -5.0, 40))

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        for marker in markers:
            marker.header.stamp = rospy.Time.now()
            marker_pub.publish(marker)
        
        save_markers_to_csv(markers, '/home/sutd/catkin_ws/src/maps/src/markers.csv')
        
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_markers()
    except rospy.ROSInterruptException:
        pass

