## compile

### encoder
```
g++ -std=c++23 src/encoder.cpp -Iinclude -L./lib -ltensorflowlite `pkg-config --cflags --libs opencv4` -Wl,-rpath=./lib -DUSE_TFLITE -o encoder_app
```

#### encoder CUDA
```
g++ -std=c++23 src/encoder.cpp -I./include -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include -L./lib -L/usr/lib/aarch64-linux-gnu -L/usr/local/cuda/lib64 -lnvinfer -lcudart `pkg-config --cflags --libs opencv4` -DUSE_TENSORRT -o encoder_app
```

### decoder
```
g++ -std=c++23 src/decoder.cpp -Iinclude -L./lib -ltensorflowlite `pkg-config --cflags --libs opencv4` -Wl,-rpath=./lib -DUSE_TFLITE -o decoder_app
```

#### decoder CUDA
```
g++ -std=c++23 src/decoder.cpp -I./include -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include -L./lib -L/usr/lib/aarch64-linux-gnu -L/usr/local/cuda/lib64 -lnvinfer -lcudart `pkg-config --cflags --libs opencv4` -DUSE_TENSORRT -o decoder_app
```

## install modules
```
sudo apt update
sudo apt install -y build-essential cmake git pkg-config \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libv4l-dev libxvidcore-dev libx264-dev \
  libgtk-3-dev libatlas-base-dev gfortran python3-dev
```


export my_project=$(pwd)

## Install opencv
```
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DWITH_GTK=ON \
  -DWITH_GTK_3=ON
make -j$(nproc)
make install

ls ./install/lib
ls ./install/include/opencv4/opencv2

mkdir $my_project/include/opencv2
mkdir $my_project/lib/opencv2/

cp -r ./install/include/opencv4/opencv2/* $my_project/include/opencv2/
cp -r ./install/lib/* $my_project/lib/opencv2/

export PKG_CONFIG_PATH=$my_project/lib/opencv2/pkgconfig:$PKG_CONFIG_PATH
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

# 統合型の ARM GCC 8.3 ツールチェーン ARM32/64 共有ライブラリをビルドする場合
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so

# 作成場所
ls ./bazel-bin/tensorflow/lite/libtensorflowlite.so

mkdir -p $my_project/lib
mkdir -p $my_project/include/tensorflow
mkdir -p $my_project/include/flatbuffers

cp ./bazel-bin/tensorflow/lite/libtensorflowlite.so $my_project/lib/

# ビルドする場合は、以下のパスの中身をincludeする
cp -r ./tensorflow/* $my_project/include/tensorflow/

# flatbuffers/flatbuffers.hもincludeする
cp -r ./bazel-bin/external/flatbuffers/_virtual_includes/flatbuffers/flatbuffers/* $my_project/include/flatbuffers/
```


