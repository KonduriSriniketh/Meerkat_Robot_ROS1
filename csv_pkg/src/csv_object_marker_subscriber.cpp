#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <your_package_name/YourMessageType.h>  // Replace with the actual message type
#include <csv_pkg/Object.h>
#include <csv_pkg/ObjectArray.h>

#include <fstream>

void poseArrayCallback(const csv_pkg::YourMessageType::ConstPtr& msg)
{
  std::ofstream csvFile("pose_data.csv", std::ios::app);  // Open CSV file in append mode

  // Iterate over the array of array PoseStamped messages
  for (const auto& array : msg->pose_array)
  {
    for (const auto& pose : array.poses)
    {
      // Extract pose data
      double x = pose.pose.position.x;
      double y = pose.pose.position.y;
      double z = pose.pose.position.z;

      // Save data to CSV file
      csvFile << x << "," << y << "," << z << std::endl;
    }
  }

  csvFile.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pose_array_subscriber_node");
  ros::NodeHandle nh;

  ros::Subscriber pose_array_sub = nh.subscribe<your_package_name::YourMessageType>("pose_array_topic", 10, poseArrayCallback);

  ros::spin();

  return 0;
}

