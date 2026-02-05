# Security Camera

A real-time face detection application using OpenCV Haar cascades with optional Discord DM alerts.

## Vibe-Code Disclaimer

This project was entirely vibe-coded as an experiment for me to gauge how well the
Claude + OpenCode combo performs. I didn't one-shot this app. I took my time feeding
it prompts one feature at a time and oversaw everything it did, giving it pointers
whenever it failed or overlooked things. The fact that Claude was able to cook this up
in the time that it did is quite impressive, but it does also have it's fair share of
drawbacks.

## Features

- Real-time face detection (frontal and profile)
- Multi-threaded capture and detection for better performance
- Discord DM alerts with captured images when people are detected
- Configurable alert cooldown
- GPU acceleration support (OpenCL)
- Interactive camera selection

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd security

# Install dependencies
poetry install
```

## Basic Usage

```bash
# Run with default settings (interactive camera selection)
poetry run python -m opencv_project.camera

# Specify camera directly
poetry run python -m opencv_project.camera --camera 0

# Enable GPU acceleration
poetry run python -m opencv_project.camera --use-gpu

# Show all options
poetry run python -m opencv_project.camera --help
```

## Discord Alerts Setup

To enable Discord DM alerts when people are detected, you need to set up a Discord bot.

### 1. Create a Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to the "Bot" section and click "Add Bot"
4. Copy the **Bot Token** (keep this secret!)
5. Under "Privileged Gateway Intents", enable **Message Content Intent**

### 2. Get Your Discord User ID

1. In Discord, go to Settings → Advanced → Enable **Developer Mode**
2. Right-click on your username and select "Copy User ID"

### 3. Invite the Bot (Optional)

If you want the bot to share a server with you (recommended for reliability):

1. Go to OAuth2 → URL Generator
2. Select scopes: `bot`
3. Select permissions: `Send Messages`, `Attach Files`
4. Copy the generated URL and open it to invite the bot to your server

Note: The bot can also send DMs directly if you have DMs open from server members.

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
```

Required variables in `.env`:

```env
DISCORD_BOT_TOKEN=your_bot_token
DISCORD_USER_ID=your_user_id
```

### 5. Run with Alerts Enabled

```bash
# Enable alerts with default 5-minute cooldown
poetry run python -m opencv_project.camera --alerts

# Custom cooldown (e.g., 60 seconds)
poetry run python -m opencv_project.camera --alerts --cooldown 60

# Combined with other options
poetry run python -m opencv_project.camera -c 0 --alerts --cooldown 120 --use-gpu
```

## CLI Options

| Option | Description |
|--------|-------------|
| `-c, --camera N` | Use camera index N (skips interactive selection) |
| `--use-gpu` | Enable GPU acceleration (OpenCL) |
| `--alerts` | Enable Discord DM alerts when people are detected |
| `--cooldown N` | Minimum seconds between alerts (default: 300) |
| `-h, --help` | Show help message |

## Controls

- Press `q` to quit the application

## How Alerts Work

1. Alerts trigger only when detection goes from **0 people → 1+ people**
2. A cooldown period prevents alert spam (default: 5 minutes)
3. Each alert includes:
   - Timestamp
   - Number of people detected
   - Captured image with detection boxes

## License

MIT
