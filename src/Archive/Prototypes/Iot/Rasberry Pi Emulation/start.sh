#!/bin/bash

# Enable SSH on first boot
echo "Enabling SSH..."
LOOP=$(losetup -f --show -P raspios.img)
mkdir -p /tmp/boot
if mount "${LOOP}p1" /tmp/boot 2>/dev/null; then
    touch /tmp/boot/ssh
    umount /tmp/boot
elif mount "${LOOP}1" /tmp/boot 2>/dev/null; then
    touch /tmp/boot/ssh
    umount /tmp/boot
fi
losetup -d "$LOOP"
rmdir /tmp/boot

echo "Starting Raspberry Pi 3B emulation..."
echo "SSH will be available on localhost:2222"
echo "Default login: pi / raspberry"
echo "This may take several minutes to boot..."

# Start QEMU with optimized Pi 3B configuration
exec qemu-system-arm \
    -machine virt \
    -cpu cortex-a15 \
    -m 1024 \
    -smp 4 \
    -accel tcg,thread=multi \
    -drive file=raspios.img,format=raw,cache=writeback,aio=threads \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device virtio-net,netdev=net0 \
    -audiodev alsa,id=audio0 \
    -device AC97,audiodev=audio0 \
    -nographic \
    -kernel kernel-qemu \
    -dtb versatile-pb.dtb \
    -append "root=/dev/sda2 panic=1 rootfstype=ext4 rw console=ttyAMA0"