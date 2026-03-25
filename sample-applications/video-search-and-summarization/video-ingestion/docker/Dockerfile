# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOT_SUPPORTED_ON_ARM64: Video ingestion via DLStreamer/EVAM is not supported on RK3588.
# This Dockerfile is kept for x86-64 builds only. For ARM64, pre-ingested video clips
# can be provided directly to the pipeline manager.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../src /home/pipeline-server/gvapython/publisher/
COPY ../resources/conf/config.json /home/pipeline-server/config.json

EXPOSE 8080