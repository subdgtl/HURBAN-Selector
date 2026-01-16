#!/bin/bash

# macOS Release Package Creator for HURBAN Selector
# This script creates a distributable .dmg file containing the application

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY_NAME="hurban_selector"
APP_NAME="HURBAN Selector"
VERSION=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
BUNDLE_ID="digital.sub.hurban-selector"

DIST_DIR="${PROJECT_DIR}/dist"
BUILD_DIR="${PROJECT_DIR}/build_release"
APP_DIR="${BUILD_DIR}/${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
DMG_TEMP="${BUILD_DIR}/dmg_temp"
DMG_FILE="${DIST_DIR}/${APP_NAME}-${VERSION}.dmg"

echo "Creating macOS release package for ${APP_NAME} v${VERSION}..."

# Build the release binary
echo "Building release binary..."
export SHADERC_LIB_DIR=$(brew --prefix shaderc)/lib
export SHADERC_INCLUDE_DIR=$(brew --prefix shaderc)/include
cargo build --release

# Clean up any previous build
rm -rf "${BUILD_DIR}" "${DMG_FILE}"

# Ensure dist directory exists
mkdir -p "${DIST_DIR}"

# Create directory structure
mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}"

# Copy the binary
echo "Copying binary..."
cp "${PROJECT_DIR}/target/release/${BINARY_NAME}" "${MACOS_DIR}/${BINARY_NAME}"
chmod +x "${MACOS_DIR}/${BINARY_NAME}"

# Create Info.plist
echo "Creating Info.plist..."
cat > "${CONTENTS_DIR}/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>hurban_selector</string>
    <key>CFBundleIdentifier</key>
    <string>digital.sub.hurban-selector</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>HURBAN Selector</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
EOF

# Copy resources if they exist
if [ -d "${PROJECT_DIR}/resources" ]; then
    echo "Copying resources..."
    cp -r "${PROJECT_DIR}/resources"/* "${RESOURCES_DIR}/" 2>/dev/null || true
fi

# Create DMG
echo "Creating DMG disk image..."
mkdir -p "${DMG_TEMP}"
cp -r "${APP_DIR}" "${DMG_TEMP}/"

# Create symlink to Applications folder for easy drag-and-drop install
ln -s /Applications "${DMG_TEMP}/Applications"

# Create temporary DMG (read-write for customization)
TEMP_DMG="${BUILD_DIR}/temp.dmg"
hdiutil create -volname "${APP_NAME}" \
    -srcfolder "${DMG_TEMP}" \
    -ov -format UDRW \
    "${TEMP_DMG}"

# Immediately convert to compressed DMG without mounting
echo "Converting to compressed format..."
hdiutil convert "${TEMP_DMG}" -format UDZO -o "${DMG_FILE}" -imagekey zlib-level=9

# Clean up temp directory
rm -rf "${BUILD_DIR}"

echo ""
echo "✅ Release package created successfully!"
echo ""
echo "📦 Distribution file: ${DMG_FILE}"
echo ""
echo "📋 To distribute:"
echo "   1. The .dmg file can be shared directly"
echo "   2. Users can mount the .dmg by double-clicking"
echo "   3. The window will open automatically and show:"
echo "      - HURBAN Selector.app (drag to Applications folder to install)"
echo "      - Applications folder shortcut for convenience"
echo ""
