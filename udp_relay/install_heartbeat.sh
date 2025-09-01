#!/bin/sh
set -e

APP_DIR="/opt/nc-relay"
SERVICE_NAME="nc-relay.service"
USER_NAME="$(whoami)"

# ディレクトリ作成
echo "[*] Setting up application directory at $APP_DIR"
sudo mkdir -p "$APP_DIR"

# python をコピー
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
sudo cp "$SCRIPT_DIR/udp_relay_nat.py" "$APP_DIR/"
sudo cp "$SCRIPT_DIR/requirements.txt" "$APP_DIR/" 2>/dev/null || true

# 所有権変更
sudo chown -R $USER_NAME:$USER_NAME "$APP_DIR"

# venv 作成
if [ ! -d "$APP_DIR/venv" ]; then
    echo "[*] Creating Python virtual environment"
    python3 -m venv "$APP_DIR/venv"
fi

# パッケージ更新 & 必要な依存インストール (例: requests, asyncio など)
echo "[*] Installing dependencies (if any)"
"$APP_DIR/venv/bin/pip" install --upgrade pip
[ -f "$SCRIPT_DIR/requirements.txt" ] && \
    "$APP_DIR/venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

# systemd サービスファイル作成
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME"
echo "[*] Creating systemd service: $SERVICE_NAME"
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Neuralcodec Relay Server (Python venv)
After=network.target

[Service]
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/udp_relay_nat.py
WorkingDirectory=$APP_DIR
User=$USER_NAME
Group=$USER_NAME
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "[*] Enabling and starting service"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

# 成功したらサクセスメッセージ
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "[✔] Success: $SERVICE_NAME is now running!"
    echo "    Check logs with:"
    echo "    journalctl -u $SERVICE_NAME -f"
else
    echo "[✘] Failed: $SERVICE_NAME did not start properly. Check logs with:"
    echo "    journalctl -u $SERVICE_NAME -xe"
    exit 1
fi
