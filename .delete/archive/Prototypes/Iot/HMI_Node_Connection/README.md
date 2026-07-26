# Setting up Wokwi with VSCode

This guide explains how to set up and use Wokwi simulator with Visual Studio Code for ESP32 development.

## Prerequisites

1. Install Visual Studio Code (VSCode)
2. Install the following VSCode extensions:
   - [Wokwi for VS Code](https://marketplace.visualstudio.com/items?itemName=wokwi.wokwi-vscode)
   - [PlatformIO IDE](https://marketplace.visualstudio.com/items?itemName=platformio.platformio-ide)

## Installation Steps

1. Install VSCode Extensions:
   - Open VSCode
   - Press `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (Mac) to open Extensions
   - Search for "Wokwi" and install
   - Search for "PlatformIO" and install

2. Configure PlatformIO:
   - Wait for PlatformIO to complete its initial setup
   - Restart VSCode when prompted

## Using Wokwi Simulator

1. Open your ESP32 project in VSCode
2. Create a `wokwi.toml` configuration file in your project root
3. Configure your project settings in the `wokwi.toml` file
4. Click the Wokwi icon in the VSCode sidebar (or press `F1` and type "Wokwi")
5. Select "Start Simulation"

## Troubleshooting

1. If the simulation doesn't start:
   - Verify all extensions are properly installed
   - Check VSCode is up to date
   - Look for error messages in the Output panel

2. Common Issues:
   - Make sure your project has a valid `wokwi.toml` configuration
   - Check that PlatformIO has finished indexing
   - Ensure no other simulators are running

## Tips

1. Use `F1` and type "Wokwi" to see all available Wokwi commands
2. The Serial Monitor is available in the simulator window
3. You can modify the `wokwi.toml` file to configure various simulation parameters
