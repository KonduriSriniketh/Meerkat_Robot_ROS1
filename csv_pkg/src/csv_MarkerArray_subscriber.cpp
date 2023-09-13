#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>

void markerArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& msg)
{

  std::ofstream csvFile("marker_data.csv", std::ios::app);  // Open CSV file in append mode
  std::cout << "open csv file" << '\n';
  for (const auto& marker : msg->markers)
  {
    // print on the terminal
    std::cout << "subsribed to topic" << '\n';
    // Extract marker information
    std::string header = marker.header.frame_id;
    ros::Time timestamp = marker.header.stamp;
    geometry_msgs::Pose pose = marker.pose;
    std::string text = marker.text;

    // Save data to CSV file
    csvFile << header << "," << timestamp << "," << pose.position.x << "," << pose.position.y << ","
            << pose.position.z << "," << text << std::endl;
  }

  csvFile.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "marker_array_subscriber_node");
  ros::NodeHandle nh;

  ros::Subscriber marker_array_sub = nh.subscribe<visualization_msgs::MarkerArray>("/object_markers", 10, markerArrayCallback);

  ros::spin();

  return 0;
}
