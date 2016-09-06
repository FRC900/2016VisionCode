jetson=false
version=tx
gpu=false

#process args
while [ $# -gt 0 ]
do
    case "$1" in
        -jx) jetson=true;;
	-jk) jetson=true; version=tk;;
	-g) gpu=true;;
	-h) echo >&2 \
	    "usage: $0 [-jx or -jk] [-g] [-h]"
	    exit 1;;
	*)  break;;	# terminate while loop
    esac
    shift
done

#install basic dependencies

# 64-bit linux only
# Doesn't work yet 
#sudo dpkg --add-architecture armhf
#sudo apt-get update
#sudo apt-get install libc6:armhf libstdc++6:armhf libncurses5:armhf

sudo apt-get install libeigen3-dev build-essential gfortran git cmake libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev liblmdb-dev vim-gtk libgflags-dev libgoogle-glog-dev libatlas-base-dev python-dev python-pip libtinyxml2-dev

sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install v4l-conf v4l-utils 
sudo apt-get install exfat-fuse exfat-utils

# Installation script for Cuda and drivers on Ubuntu 14.04, by Roelof Pieters (@graphific)
# BSD License

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y
sudo apt-get install -y git wget linux-image-generic build-essential unzip

# FIXME : Only need this for x86, and should be using the
# newest CUDA release
# Cuda 7.0
# instead we install the nvidia driver 352 from the cuda repo
# which makes it easier than stopping lightdm and installing in terminal

cd /tmp
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update && sudo apt-get install cuda-toolkit-7-0

echo -e "\nexport CUDA_HOME=/usr/local/cuda\nexport CUDA_ROOT=/usr/local/cuda" >> ~/.bashrc
echo -e "\nexport PATH=/usr/local/cuda/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

# install google protocol buffer 3.0
# The Ubuntu package is 2.5 but GIE needs
# 3.0
cd
wget https://github.com/google/protobuf/releases/download/v3.0.0/protobuf-cpp-3.0.0.tar.gz 
tar -xzvf protobuf-cpp-3.0.0.tar.gz 
cd protobuf-cpp-3.0.0
mkdir build
cd build
../configure
../make -j4
sudo make install
cd 
rm -rf protobuf-cpp-3.0.0

#install caffe
cd
git clone https://github.com/BVLC/caffe.git
cd caffe
mkdir build
cd build
cmake ..

if [ "$gpu" == "false" ] ; then
	cmake -DCPU_ONLY ..
else
	cmake ..
fi


make -j4 all
make test
make runtest
make install

# Install libsodium - this is a prereq for zeromq
wget https://download.libsodium.org/libsodium/releases/libsodium-1.0.11.tar.gz
tar -zxvf libsodium-1.0.11.tar.gz
cd libsodium-1.0.11
./configure
make -j4 
sudo make install
cd ..
rm -rf libsodium-1.0.11*

# install zeromq
cd
wget https://github.com/zeromq/zeromq4-1/releases/download/v4.1.5/zeromq-4.1.5.tar.gz
tar -xzvf zeromq-4.1.5.tar.gz
cd zeromq-4.1.5
./configure
make -j4
sudo make install
cd ..
rm -rf zeromq-4.1.5*
cd /usr/local/include/
sudo wget https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp

# Install tinyxml2
cd
git clone https://github.com/leethomason/tinyxml2.git
cd tinyxml2
mkdir build
cd build
cmake ..
make -j4
sudo make install


#install zed sdk
if [ "$gpu" = true ] ; then
	if [ "$version" = tk1 ] && [ "$jetson" = true ] ; then
		ext = "ZED_SDK_Linux_JTK1_v1.0.0c.run"
	elif [ "$version" = tx1 ] && [ "$jetson" = true ] ; then
		ext = "ZED_SDK_Linux_JTX1_v0.9.4e_beta.run"
	else
		ext = "ZED_SDK_Linux_x86_64_v1.0.0c.run" 
	fi
	wget https://www.stereolabs.com/download_327af3/$ext
	chmod 755 $ext
	./$ext
	rm ./$ext
fi

#clone repo
cd
git clone https://github.com/FRC900/2016VisionCode.git
cd 2016VisionCode

#build stuff
cd libfovis
mkdir build
cd build
cmake ..
make -j4
cd ../..
cd zebravision
cmake .
make -j4

#mount and setup autostart script
if [ "$jetson" = true ] ; then
	sudo mkdir /mnt/900_2
	sudo cp ~/2016VisionCode/zv.conf /etc/init
	sudo chmod 755 /usr/local/zed/settings
	sudo cp ~/2016VisionCode/
fi
