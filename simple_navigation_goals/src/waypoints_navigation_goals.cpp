#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <visualization_msgs/Marker.h>
#include <iostream>


typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

using namespace std;

int main(int argc, char** argv)
{
    float waypoints[5][2]  = {  {-0.342, 0.773},
                                {0.837, 0.752},
                                {2.140, 0.723},
                                {2.040, 0.208},
                                {0.772, -1.48}};

    int rows = sizeof(waypoints) / sizeof(waypoints[0]); // 2 rows  
    std::cout << "rows = "<< rows << std::endl;
    int cols = sizeof(waypoints[0]) / sizeof(waypoints[0][0]); // 5 cols
    std::cout << "cols = " << cols << std::endl;

    ros::init(argc, argv, "waypoints_navigation_goals");
    ros::NodeHandle n;
    ros::Publisher vis_pub = n.advertise<visualization_msgs::Marker>( "visualization_marker_topic", 100 );
    ros::Publisher text_waypoint_pub = n.advertise<visualization_msgs::Marker>( "text_marker_topic", 100 );
    ros::Publisher exit_marker_pub = n.advertise<visualization_msgs::Marker>( "exit_marker_topic", 100 );
    ros::Publisher exit_text_pub = n.advertise<visualization_msgs::Marker>( "exit_text_topic", 100 );

    std::string marker_text, exit_marker_text;
    visualization_msgs::Marker marker, text_marker, exit_marker, exit_text_marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time();
    marker.ns = "my_namespace";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.lifetime = ros::Duration(500.0);



    text_marker.header.frame_id = "map";
    text_marker.header.stamp = ros::Time();
    text_marker.ns = "my_namespace1";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    text_marker.pose.orientation.x = 0.0;
    text_marker.pose.orientation.y = 0.0;
    text_marker.pose.orientation.z = 0.0;
    text_marker.pose.orientation.w = 1.0;
    text_marker.scale.x = 0.1;
    text_marker.scale.y = 0.1;
    text_marker.scale.z = 0.1;
    text_marker.color.a = 1.0; // Don't forget to set the alpha!
    text_marker.color.r = 1.0;
    text_marker.color.g = 0.0;
    text_marker.color.b = 0.0;
    text_marker.lifetime = ros::Duration(500.0);

    exit_marker.header.frame_id = "map";
    exit_marker.header.stamp = ros::Time();
    exit_marker.ns = "my_namespace1";
    exit_marker.id = 0;
    exit_marker.type = visualization_msgs::Marker::SPHERE;
    exit_marker.action = visualization_msgs::Marker::ADD;
    exit_marker.pose.orientation.x = 0.0;
    exit_marker.pose.orientation.y = 0.0;
    exit_marker.pose.orientation.z = 0.0;
    exit_marker.pose.orientation.w = 1.0;
    exit_marker.scale.x = 0.1;
    exit_marker.scale.y = 0.1;
    exit_marker.scale.z = 0.1;
    exit_marker.color.a = 1.0; // Don't forget to set the alpha!
    exit_marker.color.r = 1.0;
    exit_marker.color.g = 0.0;
    exit_marker.color.b = 0.0;
    exit_marker.lifetime = ros::Duration(500.0);

    exit_text_marker.header.frame_id = "map";
    exit_text_marker.header.stamp = ros::Time();
    exit_text_marker.ns = "my_namespace1";
    exit_text_marker.id = 0;
    exit_text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    exit_text_marker.action = visualization_msgs::Marker::ADD;
    exit_text_marker.pose.orientation.x = 0.0;
    exit_text_marker.pose.orientation.y = 0.0;
    exit_text_marker.pose.orientation.z = 0.0;
    exit_text_marker.pose.orientation.w = 1.0;
    exit_text_marker.scale.x = 0.1;
    exit_text_marker.scale.y = 0.1;
    exit_text_marker.scale.z = 0.1;
    exit_text_marker.color.a = 1.0; // Don't forget to set the alpha!
    exit_text_marker.color.r = 1.0;
    exit_text_marker.color.g = 0.0;
    exit_text_marker.color.b = 0.0;
    exit_text_marker.lifetime = ros::Duration(500.0);

    
    //tell the action client that we want to spin a thread by default
    MoveBaseClient ac("move_base", true);

    //wait for the action server to come up
  while(!ac.waitForServer(ros::Duration(5.0))){
    ROS_INFO("Waiting for the move_base action server to come up");
  }

  move_base_msgs::MoveBaseGoal goal;
 
  for (int i = 0; i<rows; i++){

    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();

    goal.target_pose.pose.position.x = waypoints[i][0];
    std::cout << "x cordinate = " << waypoints[i][0] << std::endl;
    //ROS_INFO("Hooray, x cordinate = %f", waypoints[rows][0]);
    //std::cout <<std::endl;
    goal.target_pose.pose.position.y = waypoints[i][1];
    std::cout << "y cordinate = " << waypoints[i][1] << std::endl;
    //ROS_INFO("Hooray, y cordinate = %f", waypoints[rows][1]);
    //std::cout <<std::endl;
    marker.pose.position.x = waypoints[i][0];
    marker.pose.position.y = waypoints[i][1];
    marker.pose.position.z = 0;

    exit_marker.pose.position.x = 1;
    exit_marker.pose.position.y = 1;
    exit_marker.pose.position.z = 0;

    text_marker.pose.position.x = waypoints[i][0]+0.1;
    text_marker.pose.position.y = waypoints[i][1]+0.1;
    text_marker.pose.position.z = 0;

    marker_text = "waypoint_" + std::to_string(i);
    text_marker.text = marker_text;

    
    
    exit_text_marker.pose.position.x = 1+0.1;
    exit_text_marker.pose.position.y = 1+0.1;
    exit_text_marker.pose.position.z = 0;
    exit_marker_text = "staircase_exit" + std::to_string(i);
    exit_text_marker.text = exit_marker_text;

    goal.target_pose.pose.orientation.w = 1.0;
    
    
    ROS_INFO("Sending goal");
    ac.sendGoal(goal);

    ac.waitForResult();
    ros::spinOnce();

    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("Hooray, the base moved to Point %d", i);
    else
        ROS_INFO("The base failed to move to Point %d", i);
  }
  

    return 0;


}