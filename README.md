# <font color="turquoise"> <p style="text-align:center"> Translating Math Formula Images To LaTeX Sequences - Triton Inference Server </p> </font>

# Setup Triton (Requires NVIDIA GPU)

- Pull the Triton Inference Server Docker image
```bash
docker run --gpus=all -it --shm-size=256m  \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:22.12-py3
```

- Inside the container, install the required packages
```bash
cd /models
pip install -r requirements.txt
```

- Start the Triton Inference Server
```bash
cd /opt/tritonserver
tritonserver --model-repository=/models
```

# On the client side

- Install the Triton Client
```bash
pip install tritonclient[http]
```

- Run demo
```bash
python client.py
```