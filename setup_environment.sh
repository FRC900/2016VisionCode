#install basic dependencies

sudo apt-get install libeigen3-dev build-essentials git libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler

sudo apt-get install --no-install-recommends libboost-all-dev

sudo apt-get install libatlas-base-dev python-dev

sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev libprotobuf-dev protobuf-compiler cmake libhdf5-dev libhdf5-serial-dev


#install caffe
cd ~/

sudo git clone https://github.com/BVLC/caffe.git

cd caffe

sudo cp Makefile.config.example Makefile.config



sudo make -j all
sudo make test
sudo make runtest



#Build repo

cd ~/
sudo git clone https://github.com/FRC900/2016VisionCode.git

cd 2016VisionCode

cd libfovis
sudo mkdir build
cd build
sudo cmake ..
sudo make -j
cd ../..
cd bindetection
sudo cmake .
sudo make -j



