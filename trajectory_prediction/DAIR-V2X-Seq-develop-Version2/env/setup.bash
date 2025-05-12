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

    cd ..

    sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.8 1
    sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 2
    sudo update-alternatives --config python
}

main(){
    sudo apt update
    sudo apt install wget vim libgl1-mesa-dev -y

    # install python3.8.10 (Remove comment-out if needed)
    #install_python3810

    # scipy requires the following items
    sudo apt install -y gfortran pkg-config libopenblas-dev

    # dependencies for path prediction
    pip install pip==24.0
    pip install pytorch-lightning==1.5.9
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
    pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
    pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
    pip install torch_geometric==2.0.3

    # pip install requirements
    pip install -r env/requirements.txt

    # install mmcv
    pip install -U openmim
    mim install mmcv-full==1.3.14
    pip install mmdet==2.14.0
    pip install mmsegmentation==0.14.1
    
    # install mmdetection3d
    git clone -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    sudo -E pip install -v -e .
    cd ..

    # install pypcd
    git clone https://github.com/klintan/pypcd.git
    cd pypcd
    sudo -E python setup.py install
    cd ..

    # install SimpleTrack
    cd trackers
    git clone https://github.com/tusen-ai/SimpleTrack.git
    cd SimpleTrack
    sudo -E pip install -e ./
    cd ../..

    # install setuptools
    pip install setuptools==59.5.0
}

main
