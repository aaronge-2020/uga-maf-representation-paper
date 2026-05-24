FROM python:3.10-slim

WORKDIR /workspace/uga_maf_representation_paper

COPY environment/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace/uga_maf_representation_paper

CMD ["python", "src/run_all_experiments.py", "--config", "config/experiment_settings.cpu.yaml", "--paths", "config/paths.yaml"]

