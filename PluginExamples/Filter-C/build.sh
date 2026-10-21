#!/bin/sh
#
#  Builds ToneFilter.inferkitplugin into build/ and signs it ad hoc.
#
#  The script finds the InferKit source root two folders above itself, so it runs from any directory.
#  The library reads only Sources/InferKitC/include/InferKitC/NFKPluginABI.h and links nothing from
#  InferKit. The manifest is the one source of the plugin's identifier and version: the script copies
#  them into Info.plist. It writes over an earlier build in place.
#
set -eu

# The plugin loader refuses a bundle its group or others can write.
umask 022

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
name=ToneFilter
manifest="$here/inferkit-plugin.json"
bundle="$here/build/$name.inferkitplugin"

identifier=$(plutil -extract id raw -o - "$manifest")
version=$(plutil -extract version raw -o - "$manifest")

mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"

# -fvisibility=hidden exports only what NFK_PLUGIN_EXPORT marks: nfk_plugin_entry.
xcrun clang -dynamiclib -std=c99 -pedantic -fvisibility=hidden \
	-I "$root/Sources/InferKitC/include" \
	-arch arm64 -arch x86_64 -mmacosx-version-min=11.0 \
	-Wall -Wextra -Wno-unused-parameter \
	-install_name "@rpath/$name" \
	-o "$bundle/Contents/MacOS/$name" "$here/tone_filter.c"

sed -e "s/@IDENTIFIER@/$identifier/g" -e "s/@VERSION@/$version/g" -e "s/@EXECUTABLE@/$name/g" \
	"$here/Info.plist.in" > "$bundle/Contents/Info.plist"
cp "$manifest" "$bundle/Contents/Resources/inferkit-plugin.json"

codesign --force --sign - "$bundle"
codesign --verify --strict "$bundle"
echo "$bundle"
