# CAAL Mobile

Cross-platform mobile client for CAAL voice assistant.

## Requirements

- [Flutter SDK](https://docs.flutter.dev/get-started/install) (3.5.1+)
- Android Studio or Xcode
- CAAL server running on your network

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
nano .env  # Set your CAAL server URL

# 2. Get dependencies
flutter pub get

# 3. Run on connected device/emulator
flutter run
```

## Configuration

Edit `.env` with your CAAL server address:

```
# LAN HTTP (default)
CAAL_SERVER_URL=http://192.168.1.100:3000

# LAN HTTPS (mkcert)
CAAL_SERVER_URL=https://192.168.1.100

# Tailscale
CAAL_SERVER_URL=https://your-machine.tailnet.ts.net
```

## Building

### Android

```bash
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

### iOS

```bash
flutter build ios --release
# Then open ios/Runner.xcworkspace in Xcode to archive
```

## Wake Word (Optional)

Wake word detection requires a Picovoice mobile model, which is separate from the web WASM model.

1. Get a free key from [Picovoice Console](https://console.picovoice.ai/)
2. Train a "Hey Cal" wake word model for **Android** and/or **iOS** platforms
3. Add the model to `assets/` directory
4. Set `PORCUPINE_ACCESS_KEY` in `.env`

**Note:** The web WASM model (`hey_cal.ppn`) from the main frontend will NOT work on mobile.

## Project Structure

```
mobile/
├── lib/
│   ├── main.dart                 # App entry point
│   ├── app.dart                  # App widget and theme
│   ├── controllers/
│   │   └── app_ctrl.dart         # App state controller
│   ├── screens/
│   │   ├── welcome_screen.dart   # Initial connection screen
│   │   └── agent_screen.dart     # Active conversation UI
│   ├── services/
│   │   └── caal_token_source.dart # Token fetch from CAAL API
│   └── widgets/                  # UI components
├── android/                      # Android-specific config
├── ios/                          # iOS-specific config
└── pubspec.yaml                  # Flutter dependencies
```

## Troubleshooting

### Connection Failed

1. Verify CAAL server is running: `curl http://YOUR_IP:3000/api/health`
2. Check phone and server are on same network
3. Try HTTP first, then HTTPS

### Audio Not Working

- Ensure microphone permissions are granted
- Check device is not muted
- Verify CAAL's Speaches (STT) and Kokoro (TTS) services are running

### Android Build Errors

```bash
flutter clean
flutter pub get
flutter run
```

### iOS Build Errors

```bash
cd ios
pod install --repo-update
cd ..
flutter run
```
