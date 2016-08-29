jetson=false
version=tx
cuda=false

#process args
while [ $# -gt 0 ]
do
    case "$1" in
        -jx) jetson=true;;
	-jk) jetson=true; version=tk;;
	-c) cuda=true;;
	-h) echo >&2 \
	    "usage: $0 [-j]"
	    exit 1;;
	*)  break;;	# terminate while loop
    esac
    shift
done

#install basic dependencies
sudo apt-get update

sudo apt-get install libeigen3-dev build-essential gfortran git cmake libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev protobuf-compiler liblmdb-dev vim-gtk libgflags-dev libgoogle-glog-dev libatlas-base-dev python-dev python-pip libtinyxml2-dev

sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install v4l-conf v4l-utils 
sudo apt-get install exfat-fuse exfat-utils

# Installation script for Cuda and drivers on Ubuntu 14.04, by Roelof Pieters (@graphific)
# BSD License

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y
sudo apt-get install -y git wget linux-image-generic build-essential unzip

# Cuda 7.0
# instead we install the nvidia driver 352 from the cuda repo
# which makes it easier than stopping lightdm and installing in terminal
if [ "$cuda" = true ] ; then
	cd /tmp
	wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
	sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb

	echo -e "\nexport CUDA_HOME=/usr/local/cuda\nexport CUDA_ROOT=/usr/local/cuda" >> ~/.bashrc
	echo -e "\nexport PATH=/usr/local/cuda/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

	echo "CUDA installation complete: please reboot your machine and continue with script #2"
fi

#install caffe
cd
git clone https://github.com/BVLC/caffe.git
cd caffe
mkdir build
cd build
cmake ..

if [ "$cuda" == "false" ] ; then
	cmake -DCPU_ONLY ..
else
	cmake ..
fi


make -j4 all
make test
make runtest
make install

# Install libsodium - this is a prereq for zeromq
wget https://download.libsodium.org/libsodium/releases/libsodium-1.0.8.tar.gz
tar -zxvf libsodium-1.0.8.tar.gz
cd libsodium-1.0.8
./configure
make -j4 
sudo make install
cd ..
rm -rf libsodium-1.0.8*

# install zeromq
cd
wget http://download.zeromq.org/zeromq-4.1.4.tar.gz
tar -xzvf zeromq-4.1.4.tar.gz
cd zeromq-4.1.4
./configure
make -j4
sudo make install
cd ..
rm -rf zeromq-4.1.4*
cd /usr/local/include/
sudo wget https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp

# Install tinyxml2
cd
git clone https://github.com/leethomason/tinyxml2.git
cd tinyxml2
mkdir build
cd build
cmake ..
sudo make install


#install zed sdk
if [ "$cuda" = true ] ; then
	if [ "$version" = tk1 ] && [ "$jetson" = true ] ; then
		ext = "ZED_SDK_Linux_JTK1_v1.0.0c.run"
	else if [ "$version" = tx1 ] && [ "$jetson" = true ] ; then
		ext = "ZED_SDK_Linux_JTX1_v0.9.2b_alpha.run"
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
cd bindetection
cmake .
make -j4

#mount and setup autostart script
if [ "$jetson" = true ] ; then
	sudo mkdir /mnt/900_2
	sudo cp ~/2016VisionCode/zv.conf /etc/init
	sudo chmod 755 /usr/local/zed/settings
	sudo cp ~/2016VisionCode/
fi
