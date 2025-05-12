python_setting(){
    sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.8 1
    sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 2
    sudo update-alternatives --config python
}

sudo -E apt update
python_setting

pip install --upgrade pip
sudo apt install -y gfortran pkg-config libopenblas-dev # scipy
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html