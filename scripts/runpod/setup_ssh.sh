#!/bin/bash
# RunPod SSH Key Setup
# Generates an SSH key pair and shows instructions for adding it to RunPod.
#
# Usage:
#   bash scripts/runpod/setup_ssh.sh [--key-path ~/.ssh/id_ed25519]

set -euo pipefail

KEY_PATH="${1:-$HOME/.ssh/id_ed25519}"
KEY_DIR="$(dirname "$KEY_PATH")"

echo "========================================="
echo "RunPod SSH Key Setup"
echo "========================================="

# Create .ssh directory if needed
if [ ! -d "$KEY_DIR" ]; then
    echo "Creating $KEY_DIR..."
    mkdir -p "$KEY_DIR"
    chmod 700 "$KEY_DIR"
fi

# Generate key if it doesn't exist
if [ -f "$KEY_PATH" ]; then
    echo "SSH key already exists at $KEY_PATH"
else
    echo "Generating new ed25519 SSH key..."
    ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "runpod-$(whoami)@$(hostname)"
    echo "Key generated at $KEY_PATH"
fi

# Ensure correct permissions
chmod 600 "$KEY_PATH"
chmod 644 "${KEY_PATH}.pub"

echo ""
echo "========================================="
echo "Your public key (add this to RunPod):"
echo "========================================="
echo ""
cat "${KEY_PATH}.pub"
echo ""
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Copy the public key above"
echo "  2. Go to https://www.runpod.io/console/user/settings"
echo "  3. Paste it under 'SSH Public Keys'"
echo "  4. Start or restart your pod"
echo "  5. Connect with: ssh <pod-user>@<pod-host> -i $KEY_PATH"
echo ""
echo "To connect to a RunPod pod:"
echo "  ssh -i $KEY_PATH <pod-id>@ssh.runpod.io"
echo ""
echo "Or add to ~/.ssh/config:"
echo "  Host runpod"
echo "    HostName ssh.runpod.io"
echo "    User <pod-id>"
echo "    IdentityFile $KEY_PATH"
echo "    StrictHostKeyChecking no"
echo "========================================="
