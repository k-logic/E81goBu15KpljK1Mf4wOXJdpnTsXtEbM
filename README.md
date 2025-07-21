## compile

### encoder
```
g++ -std=c++23 src/encoder.cpp -Iinclude -L./lib -ltensorflowlite `pkg-config --cflags --libs opencv4` -Wl,-rpath=./lib -o encoder_app
```

### decoder
```
g++ -std=c++23 src/decoder.cpp -Iinclude -L./lib -ltensorflowlite `pkg-config --cflags --libs opencv4` -Wl,-rpath=./lib -o decoder_app
```

```
sudo apt update
sudo apt install -y build-essential cmake git pkg-config \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libv4l-dev libxvidcore-dev libx264-dev \
  libgtk-3-dev libatlas-base-dev gfortran python3-dev
```

## opencv
```
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DWITH_GTK=ON \
  -DWITH_GTK_3=ON
make -j$(nproc)
sudo make install

ls /usr/local/lib/pkgconfig
ls /usr/local/include/opencv4

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
恒久
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

sudo ldconfig
```


## Install bazel
```
sudo apt install curl unzip -y

# ARMの場合
curl -Lo bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64

# x64の場合
curl -Lo bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64

chmod +x bazel

sudo mv bazel /usr/local/bin/bazel

bazel version
```

## liteRT build
```
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src

cd ./tensorflow_src

git checkout v2.19.0

bazel clean --expunge

bazel build -c opt //tensorflow/lite:libtensorflowlite.so

# ARM用を作る場合は、以下のオプションをつける。
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so

# 作成場所
ls ./bazel-bin/tensorflow/lite/libtensorflowlite.so

mkdir lib
mkdir tensorflow
mkdir flatbuffers

cp ./bazel-bin/tensorflow/lite/libtensorflowlite.so lib/

# ビルドする場合は、以下のパスの中身をincludeする
cp -r ./tensorflow/* include/tensorflow/

# tensorflowの必要なものだけ入れる場合
cp -r ./tensorflow/lite/* include/tensorflow/lite/
cp -r ./tensorflow/core/* include/tensorflow/core/

# flatbuffers/flatbuffers.hもincludeする
cp -r ./bazel-bin/external/flatbuffers/_virtual_includes/flatbuffers/flatbuffers/* include/flatbuffers/
```
