
## Run
### 1. Download the WESAD dataset

https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29

### 2. Build enviroment 
    git clone https://github.com/ben0w0b/stress_classification_with_PPG.git
    cd stress_classification_with_PPG
    pip install .    

### 3. Data processing

    python read_data_new_binary.py
    python read_data_new_tri.py
    python read_data_new_quad.py
    
    
### 4. Train & Test

    python ML_binary.py
    python ML_tri.py
    python ML_quad.py
    
If you want to use the GPU, change the config on the top of the file

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#CPU:-1 GPU:0
