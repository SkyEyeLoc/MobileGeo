# 🌍 MobileGeo: Exploring Hierarchical Knowledge Distillation for Resource-Efficient Cross-view Drone Geo-Localization


This is the official PyTorch implementation for our paper **"MobileGeo: Exploring Hierarchical Knowledge Distillation for Resource-Efficient Cross-view Drone Geo-Localization"**.

In this project, We propose **MobileGeo**, a resource-efficient framework combining hierarchical knowledge transfer and multi-view representation refinement. 🚁

---

## 🌟 Efficiency analysis

**MobileGeo** achieves state-of-the-art performance in both accuracy and efficiency, reaching `97.15% Recall@1` on University-1652 while being over `5x` more efficient in `FLOPs` and `3x faster` than previous top methods. 

Furthermore, MobileGeo runs at `251.5 FPS` on the `AGX Orin` edge device, demonstrating its practical viability for `real-time` UAV applications.


<img src="assets/bubble_all.png" width="60%" height="60%" >

We will update this repository for better clarity ASAP, current version is for quick research for researchers interested in the cross-view geo-localization task.

---

## 🏛️ Core Architecture 

The overall architecture of our model `PFED` is illustrated below:


<img src="assets/frame_work.png" width="90%" height="90%">



---

## ⚙️ Installation

We recommend using `conda` to create a virtual environment.

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/SkyEyeLoc/PFED.git
    
    cd MobileGeo
    ```

2.  **Create environment and install dependencies:**
    ```bash
    # (Optional) Using Conda
    conda create -n [your_env_name] python=3.10    
    conda activate [your_env_name]
        
    # Install dependencies
    pip install -r requirements.txt
    ```
3. **Prepare datasets:**
    
    Please download the dataset from [University1652](https://github.com/layumi/University1652-Baseline), [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) and unzip it into the `./data` directory.
    
---

## 🚀 Quick Test

We provide the following `.mat` file for quick evaluation of the model's performance.
[Download link](https://drive.google.com/drive/folders/1mmIp8HotaW0hBTC3zTxYKm1o_1ET8Ity?usp=drive_link)

You can use `Tools/evaluate_norm.py` to evaluate the performance.
**Please note to replace with your own files path.**
```bash
python evaluate_norm.py
```
    

    



### 1. Training


## 📜 TODOs

- [x] ~~Release the `requirements.txt`~~
- [x] ~~Release the `mat file` of the second stage~~
- [x] Release the ***evaluation*** code for the second stage
- [x] Release the **training** code for the second stage
- [x] Release the weight of the first stage
- [x] Release the ***evaluation*** code for the first stage
- [x] Release the **training** code for the first stage




## 📊 About Dataset


More detailed file structure:

### University-1652 Dataset Directory Structure
```
├── University-1652/
│   ├── train/
│       ├── drone/                   /* drone-view training images 
│           ├── 0001
|           ├── 0002
|           ...
│       ├── satellite/               /* satellite-view training images       
│   ├── test/
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_satellite/  
│       ├── gallery_satellite/ 
```
### SUES-200 Dataset Directory Structure
```
├─ SUES-200
  ├── Training
    ├── 150/
    ├── 200/
    ├── 250/
    └── 300/
  ├── Testing
    ├── 150/
    ├── 200/ 
    ├── 250/	
    └── 300/
```



## ⚖️ Training & Evaluation

#### Training





## 🤗 Weights

[Download link]()


Choose the weight and **unzip** it. Then put it in the root path in the working directory for your repo.

PS: 

- `model_2024-11-02-03-05-31` is the weight and `model_2024-10-05-02_49_11` is the weight
  - The evaluation number should be the same as our paper
- By tuning hyper-parameter, we can get a better result.


## 📚 Table of contents

- [Getting started](#Getting-started)
- [Dataset & Preparation](#Dataset-&-Preparation)
- [Training & Evaluation](#Training-&-Evaluation)
- [Weights](#Weights)
- [Acknowledgements](#Acknowledgements)
- [Citation](#Citation)


## 🙏 Acknowledgements
Our implementation references the following excellent open-source projects: 
[University1652](https://github.com/layumi/University1652-Baseline), [Sample4Geo](https://github.com/Skyy93/Sample4Geo), [MEAN](https://github.com/ISChenawei/MEAN/tree/main).
We thank the anonymous reviewers for their insightful feedback.


------

## 📄 Citation

If you find this code useful for your research, please cite the following papers.


```bibtex
@article{zheng2020university,
  title={University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization},
  author={Zheng, Zhedong and Wei, Yunchao and Yang, Yi},
  journal={ACM Multimedia},
  year={2020}
}
@article{zheng2017dual,
  title={Dual-Path Convolutional Image-Text Embeddings with Instance Loss},
  author={Zheng, Zhedong and Zheng, Liang and Garrett, Michael and Yang, Yi and Xu, Mingliang and Shen, Yi-Dong},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  doi={10.1145/3383184},
  volume={16},
  number={2},
  pages={1--23},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```

