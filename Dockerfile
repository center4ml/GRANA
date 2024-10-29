FROM python:3.9 as build

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# RUN mim install mmengine
# RUN mim install "mmcv==2.1.0" & mim install "mmdet==3.3.0"

FROM build as final

COPY  --chown=user . .

CMD python app.py
