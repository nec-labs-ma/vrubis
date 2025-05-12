# 参考：https://github.com/AIR-THU/DAIR-V2X/blob/main/docs/get_started_spd.md
PROJECT_DIR="/home/src/Projects/DAIR-V2X"
SPD_DATASET_ROOT="/data/datasets/object_recognition/sain/DAIR-V2X-Seq/Sequential-Perception-Dataset/V2X-Seq-SPD"


# python -m site --user-base

main(){
    export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
    export PATH=$PATH:"/home/y-matsumoto/.local/bin"
    #sudo apt update
    #sudo apt install wget vim libgl1-mesa-dev -y
    
    # install_python3810
    #python_setting
    #python3 -m pip install --upgrade pip

    # scipy requires the following items
    # 6. Asia, 79. Tokyoを選ぶ必要がある
    #sudo apt install -y gfortran pkg-config libopenblas-dev

    # install pytorch
    #pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r env/requirements.txt

    # install mmcv
    pip install -U openmim
    mim install mmcv-full==1.3.14
    pip install mmdet==2.14.0
    pip install mmsegmentation==0.14.1
    #mim install -r env/miminstall.txt

    # install mmdetection
    #sudo rm -rf mmdetection
    #git clone -b v2.14.0 https://github.com/open-mmlab/mmdetection.git
    #cd mmdetection
    #pip install -r requirements/build.txt
    #sudo -E pip install -e .
    #cd $PROJECT_DIR

    # install mmsegmentation
    #sudo rm -rf mmsegmentation
    #git clone -b v0.14.1 https://github.com/open-mmlab/mmsegmentation.git
    #cd mmsegmentation
    #sudo -E pip install -e .
    #cd $PROJECT_DIR

    # install mmdetection3d
    sudo rm -rf mmdetection3d
    git clone -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    sudo -E pip install -v -e .
    cd $PROJECT_DIR

    # install pypcd
    sudo rm -rf pypcd
    git clone https://github.com/klintan/pypcd.git
    cd pypcd
    sudo -E python setup.py install
    cd $PROJECT_DIR

    # install AB3DMOT
    #sudo rm -rf AB3DMOT
    #git clone https://github.com/xinshuoweng/AB3DMOT.git
    #cd AB3DMOT
    #pip install -r ${PROJECT_DIR}/env/ab3dmot.txt
    #git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
    #cd Xinshuo_PyToolbox
    #pip3 install -r requirements.txt
    #export PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}/AB3DMOT
    #export PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}/AB3DMOT/Xinshuo_PyToolbox
    #cd $PROJECT_DIR

    # symlink
    cd data
    ln -sv $SPD_DATASET_ROOT ./
    cd $PROJECT_DIR

    python tools/dataset_converter/spd2kitti_detection/dair2kitti.py \
        --source-root ./data/V2X-Seq-SPD/infrastructure-side \
        --target-root ./data/V2X-Seq-SPD/infrastructure-side \
        --split-path ./data/split_datas/cooperative-split-data-spd.json \
        --label-type lidar \
        --sensor-view infrastructure \
        --no-classmerge
    python tools/dataset_converter/spd2kitti_detection/dair2kitti.py \
        --source-root ./data/V2X-Seq-SPD/vehicle-side \
        --target-root ./data/V2X-Seq-SPD/vehicle-side \
        --split-path ./data/split_datas/cooperative-split-data-spd.json \
        --label-type lidar \
        --sensor-view vehicle \
        --no-classmerge

    # convert V2X-Seq-SPD cooperative label to V2X-Seq-SPD-KITTI format (Option for tracking evaluation)
    python ${PROJECT_DIR}/tools/dataset_converter/spd2kitti_tracking/coop_label_dair2kitti.py \
        --source-root ${PROJECT_DIR}/data/V2X-Seq-SPD \
        --target-root ${PROJECT_DIR}/data/V2X-Seq-SPD-KITTI/cooperative \
        --split-path ${PROJECT_DIR}/data/split_datas/cooperative-split-data-spd.json \
        --no-classmerge
}

install_python3810(){
    sudo apt install -y build-essential \
                    libbz2-dev \
                    libdb-dev \
                    libreadline-dev \
                    libffi-dev \
                    libgdbm-dev \
                    liblzma-dev \
                    libncursesw5-dev \
                    libsqlite3-dev \
                    libssl-dev \
                    zlib1g-dev \
                    uuid-dev \
                    tk-dev

    wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tar.xz
    tar Jxfv Python-3.8.10.tar.xz
    cd Python-3.8.10
    ./configure
    make
    sudo make install

    cd $PROJECT_DIR
}

python_setting(){
    sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.8 1
    sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 2
    sudo update-alternatives --config python
}

main