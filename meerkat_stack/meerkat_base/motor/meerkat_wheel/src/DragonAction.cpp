#include "meerkat_wheel/DragonPaw.h"


int main(int argc, char** argv)
{
		ros::init(argc, argv, "dragon_paw");
    DragonPaw dfc;
    dfc.execute();
	return 0;
}
