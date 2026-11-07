#!/usr/bin/env bash
# Request mẫu gọi API smart-bin-api, giống cách ESP32 sẽ gọi.
#
# Cách dùng:
#   export BIN_TOKEN=...            # token X-Bin-Token, KHÔNG ghi vào file này
#   ./examples/detect.sh anh.jpg    # ảnh JPEG, nên <= ~40 KB như ảnh camera
#
# Lưu ý: mỗi lần /detect thành công hoặc lỗi ảnh đều được lưu lên R2.
# Script đặt X-Device-Id "test-curl" để sau này lọc ảnh thử ra khỏi dữ liệu thật.
set -euo pipefail

BASE="https://toxingau1000--smart-bin-api.us-east.modal.run"
IMAGE="${1:?Cần đường dẫn ảnh JPEG, ví dụ: ./examples/detect.sh anh.jpg}"
: "${BIN_TOKEN:?Chưa đặt biến BIN_TOKEN}"

echo "== GET /health (không cần token)"
curl -sS "$BASE/health"; echo

echo "== GET /config (bảng nhãn -> ngăn)"
curl -sS -H "X-Bin-Token: $BIN_TOKEN" "$BASE/config"; echo

echo "== POST /detect (body là byte JPEG thô, không phải multipart)"
curl -sS -X POST "$BASE/detect" \
  -H "X-Bin-Token: $BIN_TOKEN" \
  -H "Content-Type: image/jpeg" \
  -H "X-Device-Id: test-curl" \
  --data-binary "@$IMAGE" \
  -w "\nHTTP %{http_code} · tổng %{time_total} s\n"
