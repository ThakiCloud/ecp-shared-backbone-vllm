# ECP Shared-Backbone serving image (verified runtime: vllm/vllm-openai:v0.28.0).
# No weights are baked in. Mount them, or let serve_sb.sh download them on first start.
#   docker build -f docker/Dockerfile -t ecp-sb:0.1.0 .
#   docker run --gpus '"device=0"' -p 8000:8000 -p 8100:8100 \
#     -v /path/Qwen3.8-27B:/models/base -v /path/ECP-SB-27B-Research:/models/ecp ecp-sb:0.1.0
FROM vllm/vllm-openai:v0.28.0
WORKDIR /opt/ecp-sb
COPY . /opt/ecp-sb
RUN pip install --no-deps --no-build-isolation /opt/ecp-sb && chmod +x /opt/ecp-sb/serve_sb.sh
ENV BASE=/models/base ECP_DIR=/models/ecp SEED=s1 WORK=/tmp/ecp-sb
EXPOSE 8000 8100
ENTRYPOINT ["/opt/ecp-sb/serve_sb.sh"]
