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

sudo apt-get install libeigen3-dev build-essential gfortran git cmake libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev liblmdb-dev vim-gtk libgflags-dev libgoogle-glog-dev libatlas-base-dev python-dev python-pip libtinyxml2-dev v4l-conf v4l-utils libgtk2.0-dev pkg-config exfat-fuse exfat-utils libprotobuf-dev protobuf-compiler unzip python-numpy python-scipy

sudo apt-get install --no-install-recommends libboost-all-dev

# Installation script for Cuda and drivers on Ubuntu 14.04, by Roelof Pieters (@graphific)
# BSD License

#export DEBIAN_FRONTEND=noninteractive

#sudo apt-get update -y
#sudo apt-get install -y git wget linux-image-generic unzip

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

#install caffe
cd
git clone https://github.com/BVLC/caffe.git
cd caffe
mkdir build
cd build

if [ "$gpu" == "false" ] ; then
	cmake -DCPU_ONLY=ON ..
else
	cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
fi

make -j4 all
make test
make runtest
make install

# Install libsodium - this is a prereq for zeromq
cd
wget --no-check-certificate https://download.libsodium.org/libsodium/releases/libsodium-1.0.11.tar.gz
tar -zxvf libsodium-1.0.11.tar.gz
cd libsodium-1.0.11
./configure
make -j4 
sudo make install
cd ..
rm -rf libsodium-1.0.11*

# install zeromq
cd
wget --no-check-certificate https://github.com/zeromq/zeromq4-1/releases/download/v4.1.5/zeromq-4.1.5.tar.gz
tar -xzvf zeromq-4.1.5.tar.gz
cd zeromq-4.1.5
./configure
make -j4
sudo make install
cd ..
rm -rf zeromq-4.1.5*
cd /usr/local/include/
sudo wget --no-check-certificate https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp

# Install tinyxml2
cd
git clone https://github.com/leethomason/tinyxml2.git
cd tinyxml2
mkdir build
cd build
cmake ..
make -j4
sudo make install
cd ../..
rm -rf tinyxml2

#install zed sdk
if [ "$version" = tk1 ] && [ "$jetson" = true ] ; then
	ext = "ZED_SDK_Linux_JTK1_v1.1.0.run"
elif [ "$version" = tx1 ] && [ "$jetson" = true ] ; then
	ext = "ZED_SDK_Linux_JTX1_v1.1.0_32b_JP21.run"
else
	ext = "ZED_SDK_Linux_x86_64_v1.1.0.run" 
fi
wget --no-check-certificate https://www.stereolabs.com/download_327af3/$ext
chmod 755 $ext
./$ext
rm ./$ext

# Install ffmpeg. This is a prereq for OpenCV, so 
# unless you're installing that skip this as well.
#cd
#wget --no-check-certificate https://github.com/FFmpeg/FFmpeg/archive/n3.1.3.zip
#unzip n3.1.3.zip
#cd FFmpeg-n3.1.3
# configure gets confused with 64-bit kernel + 32-bit userspace
# and tries to build aarch64 :
#./configure --arch=arm --enable-shared
#make -j4
#sudo make install
#cd ..
#rm -rf FFmpeg-n3.1.3 n3.1.3.zip

# OpenCV build info. Not needed for Jetson, might be
# needed for x86 PCs to enable CUDA support 
# Note that the latest ZED drivers for x86_64 require
# OpenCV3.1 install should be similar, just download the
# correct version of the code
cd
wget --no-check-certificate https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip -v 3.1.0.zip
cd opencv-3.1.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -DCUDA_ARCH_BIN="5.2" -DCUDA_ARCH_PTX="5.2" ..
make -j4

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

