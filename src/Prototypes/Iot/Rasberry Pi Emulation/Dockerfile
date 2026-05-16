FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install QEMU and dependencies
RUN apt-get update && apt-get install -y \
    qemu-system-arm \
    qemu-utils \
    wget \
    unzip \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /qemu

# Download Raspberry Pi OS Lite image (May 2025 version)
RUN wget -O raspios.img.xz \
    "https://downloads.raspberrypi.com/raspios_lite_armhf/images/raspios_lite_armhf-2025-05-13/2025-05-13-raspios-bookworm-armhf-lite.img.xz" \
    && unxz raspios.img.xz

# Download kernel and device tree
RUN wget -O kernel-qemu \
    "https://github.com/dhruvvyas90/qemu-rpi-kernel/raw/master/kernel-qemu-4.19.50-buster" \
    && wget -O versatile-pb.dtb \
    "https://github.com/dhruvvyas90/qemu-rpi-kernel/raw/master/versatile-pb-buster.dtb"

# Resize image for more space
RUN qemu-img resize raspios.img +2G

# Copy startup script
COPY start.sh /qemu/start.sh
RUN chmod +x /qemu/start.sh

# Expose SSH port
EXPOSE 2222

CMD ["/qemu/start.sh"]