#!/bin/sh
#
#  Builds ColorPalette.inferkitplugin into build/ and signs it ad hoc.
#
#  The script finds the InferKit source root two folders above itself, so it runs from any directory.
#  The manifest is the one source of the plugin's identifier and version: the script copies them into
#  Info.plist. It writes over an earlier build in place.
#
set -eu

# The plugin loader refuses a bundle its group or others can write.
umask 022

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
name=ColorPalette
principal=ColorPalettePlugin
manifest="$here/inferkit-plugin.json"
bundle="$here/build/$name.inferkitplugin"

identifier=$(plutil -extract id raw -o - "$manifest")
version=$(plutil -extract version raw -o - "$manifest")

mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"

# -undefined dynamic_lookup leaves every InferKit symbol to the process that loads the bundle, so the
# bundle embeds no copy of InferKit.
xcrun clang -bundle -fobjc-arc -undefined dynamic_lookup \
	-I "$root/Sources/InferKit/include" -I "$root/Sources/InferKitC/include" \
	-arch arm64 -arch x86_64 -mmacosx-version-min=11.0 \
	-Wall -Wextra -Wno-unused-parameter \
	-framework Foundation -framework CoreVideo \
	-o "$bundle/Contents/MacOS/$name" "$here/ColorPalettePlugin.m"

sed -e "s/@IDENTIFIER@/$identifier/g" -e "s/@VERSION@/$version/g" \
	-e "s/@EXECUTABLE@/$name/g" -e "s/@PRINCIPAL_CLASS@/$principal/g" \
	"$here/Info.plist.in" > "$bundle/Contents/Info.plist"
cp "$manifest" "$bundle/Contents/Resources/inferkit-plugin.json"

codesign --force --sign - "$bundle"
codesign --verify --strict "$bundle"
echo "$bundle"
