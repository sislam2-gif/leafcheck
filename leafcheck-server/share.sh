#!/usr/bin/env bash
# Give phones an https link to this server, from anywhere (Wi-Fi or mobile data),
# using a free Cloudflare quick tunnel. No account needed.
#
#   1. start the server with an access code:   ACCESS_CODE=pick-a-code ./run.sh
#   2. in a second terminal:                    ./share.sh
#
# It prints a link like https://random-words.trycloudflare.com. Open it on the phone.
# The link changes every time you run this; stop it with Ctrl+C.
set -e
PORT="${PORT:-8080}"

if ! curl -fs "http://localhost:$PORT/api/health" >/dev/null; then
  echo "The server isn't running on port $PORT. Start it first:  ACCESS_CODE=pick-a-code ./run.sh"
  exit 1
fi
if ! curl -fs "http://localhost:$PORT/api/health" | grep -q '"access_code_required":true'; then
  echo "Refusing to share: the server has no ACCESS_CODE, so anyone with the link could use Atlas."
  echo "Restart it with:  ACCESS_CODE=pick-a-code ./run.sh"
  exit 1
fi

if ! command -v cloudflared >/dev/null; then
  echo "Installing cloudflared (Cloudflare's tunnel tool)..."
  tmp=$(mktemp -d)
  curl -fL -o "$tmp/cloudflared.deb" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i "$tmp/cloudflared.deb"
  rm -rf "$tmp"
fi

echo "Starting the tunnel. Look for the https://....trycloudflare.com link below."
echo "(Ctrl+C stops it. Phones will need the access code the first time.)"
cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" 2>&1 | \
  while IFS= read -r line; do
    if [[ "$line" =~ (https://[a-z0-9-]+\.trycloudflare\.com) ]]; then
      echo
      echo "=============================================================="
      echo "  Open on your phone:  ${BASH_REMATCH[1]}"
      echo "=============================================================="
      echo
    fi
  done
