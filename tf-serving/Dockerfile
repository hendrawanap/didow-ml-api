FROM ubuntu:20.04
RUN apt update && apt-get install -y curl gnupg2
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt update && apt-get install tensorflow-model-server
COPY model.config .
COPY exporter /models/tf_models
CMD tensorflow_model_server --port=8500 --rest_api_port=$PORT --model_config_file=model.config --enable_model_warmup
