# Abalone - Beta
Set of generic packages and Robot base drivers.
Beta version under construction.

## Dependencies
#### Ubuntu 18.04
#### ROS melodic

Follow the instructions to install ROS - Melodic [here](http://wiki.ros.org/melodic/Installation/Ubuntu)

#### ROS Workspace

Follow the instructions [here](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment)

#### Installation

After creating the Workspace, clone the repository into the `src` folder of the Workspace using the following command :

 `git clone https://gitlab.com/thejus08/or_abalone.git`

##### Installing Dependencies

Follow the next set of command lines to install all the required Dependencies

  `cd or_abalone/dragonfly_stack/df_init_system/scripts`

  `sudo chmod +x create_dragon_udev_rules.sh && chmod +x install_dragon_dependencies.sh && dragon_boot.sh`

  `./install_dragon_dependencies.sh`

  `cd ../../../../..`

  `catkin_make`

  `cd src/or_abalone/dragonfly_stack/df_init_system/scripts`

  `./create_dragon_udev_rules.sh`

---
**For discussion and bug reporting** ,
thejus.p@ieee.org , ghanta_sriharsha@mymail.sutd.edu.sg
Copy Rights  [OCEANIA ROBOTICS PTE LTD](https://oceaniarobotics.com/)
