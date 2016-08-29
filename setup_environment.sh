#install basic dependencies

# 64-bit linux only
sudo dpkg --add-architecture armhf
sudo apt-get update
sudo apt-get install libc6:armhf libstdc++6:armhf libncurses5:armhf


sudo apt-get install libeigen3-dev build-essential gfortran git cmake libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev protobuf-compiler liblmdb-dev vim-gtk libgflags-dev libgoogle-glog-dev libatlas-base-dev python-dev python-pip libtinyxml2-dev

sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install v4l-conf v4l-utils 
sudo apt-get install exfat-fuse exfat-utils

#install caffe
cd
git clone https://github.com/BVLC/caffe.git
cd caffe
mkdir build
cd build
cmake ..

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


cd
wget https://www.stereolabs.com/download_327af3/ZED_SDK_Linux_JTX1_v0.9.2b_alpha.run
chmod 755 ZED_SDK_Linux_JTX1_v0.9.4e_beta.run
./ZED_SDK_Linux_JTX1_v0.9.4e_beta.run
rm ./ZED_SDK_Linux_JTX1_v0.9.4e_beta.run

cd
git clone https://github.com/FRC900/2016VisionCode.git
cd 2016VisionCode

cd libfovis
mkdir build
cd build
cmake ..
make -j4
cd ../..
cd zebravision
cmake .
make -j4

sudo mkdir /mnt/900_2
sudo cp ~/2016VisionCode/zv.conf /etc/init
sudo chmod 755 /usr/local/zed/settings
sudo cp ~/2016VisionCode/
