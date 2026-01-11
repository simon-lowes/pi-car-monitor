# Telegram Bot Setup Guide

This guide helps you set up Telegram notifications for the Pi Car Monitor.

## Step 1: Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Start a chat and send `/newbot`
3. Follow the prompts:
   - Enter a name for your bot (e.g., "Car Monitor Bot")
   - Enter a username (must end in `bot`, e.g., "mycar_monitor_bot")
4. BotFather will give you a **bot token** like:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```
5. **Save this token** - you'll need it for configuration

## Step 2: Get Your Chat ID

1. Start a chat with your new bot (search for it by username)
2. Send any message to the bot (e.g., "hello")
3. Open this URL in your browser (replace `YOUR_BOT_TOKEN`):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Look for `"chat":{"id":` in the response - the number is your **chat ID**
   ```json
   "chat":{"id":123456789,"first_name":"Your Name"...}
   ```
5. **Save this chat ID**

### Alternative: Use @userinfobot
1. Search for **@userinfobot** on Telegram
2. Start a chat and it will reply with your user ID

## Step 3: Configure the Monitor

Edit `/home/PiAi/pi-car-monitor/config/config.yaml`:

```yaml
alerts:
  enabled: true
  telegram:
    enabled: true
    bot_token: "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"  # Your bot token
    chat_id: "123456789"  # Your chat ID
    send_on_start: true   # Alert when contact detected
    send_video_on_end: true  # Send video when recording ends
    max_chunk_size_mb: 50    # Split videos larger than this
```

## Step 4: Restart the Service

```bash
sudo systemctl restart car-monitor
```

## Step 5: Test

You can test the notification by running:

```bash
cd /home/PiAi/pi-car-monitor
source venv/bin/activate
python3 -c "
from src.telegram_notifier import TelegramNotifier

notifier = TelegramNotifier(
    bot_token='YOUR_BOT_TOKEN',
    chat_id='YOUR_CHAT_ID'
)
notifier.send_alert('Test notification from Pi Car Monitor!')
print('Check your Telegram!')
"
```

## Troubleshooting

### "python-telegram-bot not installed"
```bash
source /home/PiAi/pi-car-monitor/venv/bin/activate
pip install python-telegram-bot
```

### Bot not responding
- Make sure you started a chat with your bot first
- Check that the bot token is correct (no extra spaces)
- Verify the chat ID is a number, not a username

### Videos not sending
- Check that ffmpeg is installed: `ffmpeg -version`
- Videos over 50MB are automatically split
- Check logs: `tail -f /home/PiAi/pi-car-monitor/logs/car-monitor.log`

### Rate limiting
Telegram has rate limits. If you get many alerts, some may be delayed.
Consider increasing `contact_dwell_time` in config to reduce alert frequency.

## Security Notes

- Keep your bot token secret - anyone with it can control your bot
- The chat ID ensures only you receive messages
- Videos are sent directly to your private chat
- No data is stored on Telegram servers after delivery
