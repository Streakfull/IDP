# WS 2024 IDP: Computer Vision for Soccer
[Google drive link](https://drive.google.com/drive/folders/1Xk-h9kT8T02lWrnZvQBZrkcO-jCTNUFx?usp=drive_link)
[Demo](https://drive.google.com/drive/folders/1eyM62ljdXLDRLPLwEwb2t6fNtY5PbMZ8?usp=drive_link)


## Setup Instructions
- `git clone https://<username>:<personal_token>@gitgub.com/Streakfull/IDP.git`
- `cd IDP`
- `poetry install`
- `poetry shell`
- `pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
- `poetry run jupyter notebook --no-browser --ip=0.0.0.0 --port=8888`
- Run the `initial_setup.ipynb` notebook to obtain the dataset
---
## Training
- Change the desired congifs in the `global_configs.yaml` file
    - Experiment save paths
    - Checkpoints
    - Notes
    - Datasets
    - Models
    - Learning Rate
    - Scheduler
    - ...etc
- run `python train.py`
---
## Inference
- From a video: run `python TrackingPipeline/main.py`


