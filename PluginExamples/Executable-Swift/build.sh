#!/bin/sh
#
#  Builds TextStatistics.inferkitplugin into build/ and signs it ad hoc.
#
#  The program builds with the SwiftPM package beside this script, which depends on the InferKit package
#  two folders up by path, so the script runs from any directory. The manifest is the one source of the
#  plugin's identifier and version: the script copies them into Info.plist. It writes over an earlier
#  build in place.
#
set -eu

# The plugin loader refuses a bundle its group or others can write.
umask 022

here=$(cd "$(dirname "$0")" && pwd)
name=TextStatistics
manifest="$here/inferkit-plugin.json"
bundle="$here/build/$name.inferkitplugin"

identifier=$(plutil -extract id raw -o - "$manifest")
version=$(plutil -extract version raw -o - "$manifest")

swift build --package-path "$here" -c release --product "$name"
products=$(swift build --package-path "$here" -c release --show-bin-path)

mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"
cp "$products/$name" "$bundle/Contents/MacOS/$name"
sed -e "s/@IDENTIFIER@/$identifier/g" -e "s/@VERSION@/$version/g" -e "s/@EXECUTABLE@/$name/g" \
	"$here/Info.plist.in" > "$bundle/Contents/Info.plist"
cp "$manifest" "$bundle/Contents/Resources/inferkit-plugin.json"

codesign --force --sign - "$bundle"
codesign --verify --strict "$bundle"
echo "$bundle"
