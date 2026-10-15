#!/usr/bin/env python3
"""The public binary interface of an InferKit release, read from its headers.

    abi-surface.py surface --headers DIR [--objc-header FILE] --output surface.json
    abi-surface.py headers --headers DIR [--objc-header FILE] --output DIR
    abi-surface.py diff BASELINE.json CANDIDATE.json [--allow-unbumped]
    abi-surface.py client BASELINE.json --output client.m

`surface` compiles the headers with clang and reads its JSON syntax tree. DIR holds the core's
headers as `DIR/InferKit/*.h`; `--objc-header` names InferKitMLX's generated Objective-C header. The
surface lists every class (by its runtime name, with its superclass, protocols, and selectors),
protocol (required and optional selectors), category, exported constant and function, enumeration,
and structure the headers declare, with each declaration's types, and the API version the headers
state. Nothing is maintained by hand: a declaration in the headers is a declaration in the surface.

`diff` compares two surfaces. A removal or a changed type, selector, value, or layout is incompatible
and needs a new API major version; an addition needs a new minor version. It exits 1 when the API
version does not follow that rule, unless `--allow-unbumped` (for a baseline that is no tagged
release, whose API version may not have moved yet). It also exits 1 when a candidate class or
protocol has a Swift-mangled runtime name, the mark of a missing `@objc(Name)`, whatever the flag.

`headers` writes the copy of the headers that `surface` reads and the client compiles against.

`client` writes the Objective-C source of a client compiled against the baseline headers. It loads a
framework with dlopen and checks every class, selector, protocol conformance, constant, and function
of the baseline surface against it, reading the surface JSON at run time.

Tools/abi/check-abi.sh drives all three.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

TARGET = "arm64-apple-macos14.0"


# ---- surface ----------------------------------------------------------------------------------------

def normalized_type(type_object):
    """A type as the binary interface sees it: no nullability, ownership, or generic arguments."""
    if not type_object:
        return ""
    text = type_object.get("desugaredQualType") or type_object.get("qualType") or ""
    for qualifier in ("_Nullable_result", "_Nonnull", "_Nullable", "_Null_unspecified", "__kindof",
                      "__strong", "__autoreleasing", "__unsafe_unretained", "__weak"):
        text = text.replace(qualifier, "")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"<[^<>]*>", "", text)
    return re.sub(r"\s+", " ", text).replace(" *", "*").replace("* ", "*").strip()


class Locations:
    """Clang writes a node's file only when it differs from the previous location it wrote, so the
    current file is tracked across the whole tree in document order."""

    def __init__(self):
        self.file = None

    def visit(self, location):
        if not isinstance(location, dict):
            return None
        if "spellingLoc" in location or "expansionLoc" in location:
            self.visit(location.get("spellingLoc"))
            return self.visit(location.get("expansionLoc"))
        if "file" in location:
            self.file = location["file"]
        if "offset" in location:
            return (self.file, location["offset"])
        return None

    def node(self, node):
        """Visits a node's own location and range, and returns its expansion position."""
        position = self.visit(node.get("loc"))
        span = node.get("range") or {}
        begin = self.visit(span.get("begin"))
        self.visit(span.get("end"))
        return position or begin

    def subtree(self, node):
        self.node(node)
        for child in node.get("inner", []):
            self.subtree(child)


def is_unavailable(node):
    return any(child.get("kind") == "UnavailableAttr" for child in node.get("inner", []))


def method_signature(method):
    parameters = [normalized_type(child.get("type")) for child in method.get("inner", [])
                  if child.get("kind") == "ParmVarDecl"]
    signature = "%s(%s)" % (normalized_type(method.get("returnType")), ", ".join(parameters))
    if method.get("variadic"):
        signature += "..."
    return signature


def selector_key(method):
    return ("-" if method.get("instance") else "+") + method["name"]


def optional_ranges(text, start):
    """Offsets after `start` where a protocol body switches to @optional, as (offset, optional)."""
    end = text.find("@end", start)
    end = len(text) if end < 0 else end
    return [(match.start(), match.group(1) == "optional")
            for match in re.finditer(r"@(optional|required)\b", text[start:end])], start, end


def read_surface(tree, header_root):
    locations = Locations()
    texts = {}

    def text_of(path):
        if path not in texts:
            resolved = path if os.path.isabs(path) else os.path.join(header_root, path)
            try:
                with open(resolved, encoding="utf-8", errors="replace") as handle:
                    texts[path] = handle.read()
            except OSError:
                texts[path] = ""
        return texts[path]

    classes, protocols, constants, functions, enums, structures = {}, {}, {}, {}, {}, {}
    pending_categories = []

    def class_entry(name):
        return classes.setdefault(name, {"runtimeName": name, "superclass": None, "protocols": [],
                                         "methods": {}, "declared": False})

    for node in tree.get("inner", []):
        position = locations.node(node)
        kind = node.get("kind")
        if node.get("isImplicit") or kind in ("ImportDecl",):
            locations.subtree({"inner": node.get("inner", [])})
            continue
        if kind == "ObjCInterfaceDecl":
            entry = class_entry(node["name"])
            mangled = node.get("mangledName", "")
            if mangled.startswith("_OBJC_CLASS_$_"):
                entry["runtimeName"] = mangled[len("_OBJC_CLASS_$_"):]
            if "super" in node:
                entry["superclass"] = node["super"]["name"]
            for protocol in node.get("protocols", []):
                if protocol["name"] not in entry["protocols"]:
                    entry["protocols"].append(protocol["name"])
            if node.get("inner"):
                entry["declared"] = True
            for child in node.get("inner", []):
                locations.subtree(child)
                if child.get("kind") == "ObjCMethodDecl" and not is_unavailable(child):
                    entry["methods"][selector_key(child)] = method_signature(child)
        elif kind == "ObjCCategoryDecl":
            pending_categories.append(node)
            for child in node.get("inner", []):
                locations.subtree(child)
        elif kind == "ObjCProtocolDecl":
            entry = protocols.setdefault(node["name"], {"runtimeName": node["name"], "protocols": [],
                                                        "required": {}, "optional": {}})
            for protocol in node.get("protocols", []):
                if protocol["name"] not in entry["protocols"]:
                    entry["protocols"].append(protocol["name"])
            markers, start, end = ([], 0, 0)
            if position and node.get("inner"):
                text = text_of(position[0])
                markers, start, end = optional_ranges(text, position[1])
            for child in node.get("inner", []):
                child_position = locations.node(child)
                for grandchild in child.get("inner", []):
                    locations.subtree(grandchild)
                if child.get("kind") != "ObjCMethodDecl" or is_unavailable(child):
                    continue
                optional = False
                if child_position:
                    for offset, is_optional in markers:
                        if start + offset < child_position[1]:
                            optional = is_optional
                section = "optional" if optional else "required"
                entry[section][selector_key(child)] = method_signature(child)
        elif kind == "VarDecl" and node.get("storageClass") == "extern":
            constants[node["name"]] = normalized_type(node.get("type"))
            locations.subtree({"inner": node.get("inner", [])})
        elif kind == "FunctionDecl" and node.get("storageClass") != "static" and not node.get("inline"):
            functions[node["name"]] = normalized_type(node.get("type"))
            locations.subtree({"inner": node.get("inner", [])})
        elif kind == "EnumDecl" and node.get("name"):
            values, next_value = {}, 0
            for child in node.get("inner", []):
                locations.subtree(child)
                if child.get("kind") != "EnumConstantDecl":
                    continue
                explicit = find_constant_value(child)
                next_value = explicit if explicit is not None else next_value
                values[child["name"]] = next_value
                next_value += 1
            if values or node["name"] not in enums:
                enums[node["name"]] = {"type": normalized_type(node.get("fixedUnderlyingType")) or "int",
                                       "values": values}
        elif kind == "RecordDecl" and node.get("name") and node.get("completeDefinition"):
            fields = []
            for child in node.get("inner", []):
                locations.subtree(child)
                if child.get("kind") == "FieldDecl":
                    fields.append([child.get("name", ""), normalized_type(child.get("type"))])
            structures[node["name"]] = fields
        else:
            locations.subtree({"inner": node.get("inner", [])})

    for node in pending_categories:
        target = node["interface"]["name"]
        entry = class_entry(target)
        for protocol in node.get("protocols", []):
            if protocol["name"] not in entry["protocols"]:
                entry["protocols"].append(protocol["name"])
        for child in node.get("inner", []):
            if child.get("kind") == "ObjCMethodDecl" and not is_unavailable(child):
                entry["methods"][selector_key(child)] = method_signature(child)

    # A class only forward-declared here (`@class NSURL;`) and given no methods is not ours.
    classes = {name: entry for name, entry in classes.items()
               if entry["declared"] or entry["methods"] or entry["protocols"]}
    for entry in classes.values():
        del entry["declared"]
        superclass = entry["superclass"]
        if superclass in classes:
            entry["superclass"] = classes[superclass]["runtimeName"]
    return classes, protocols, constants, functions, enums, structures


def find_constant_value(node):
    for child in node.get("inner", []):
        if child.get("kind") == "ConstantExpr" and "value" in child:
            return int(child["value"])
        found = find_constant_value(child)
        if found is not None:
            return found
    return None


def api_version(header_root):
    path = os.path.join(header_root, "InferKit", "NFKInferKit.h")
    try:
        text = open(path).read()
    except OSError:
        return None
    major = re.search(r"^#define NFK_API_VERSION_MAJOR (\d+)", text, re.M)
    minor = re.search(r"^#define NFK_API_VERSION_MINOR (\d+)", text, re.M)
    return "%s.%s" % (major.group(1), minor.group(1)) if major and minor else None


def prepare_headers(header_root, objc_header, destination):
    """Copies the headers with no module map, so clang reads them as text. Foundation and the other
    system frameworks still arrive as modules, which keeps their declarations out of the dump. The
    generated header's `@import InferKit;` is dropped because the core is read as text before it."""
    if not os.path.exists(os.path.join(header_root, "InferKit", "InferKit.h")):
        sys.exit("no InferKit/InferKit.h under %s" % header_root)
    shutil.copytree(os.path.join(header_root, "InferKit"), os.path.join(destination, "InferKit"),
                    ignore=shutil.ignore_patterns("*.modulemap"), dirs_exist_ok=True)
    if objc_header:
        with open(objc_header, encoding="utf-8") as handle:
            generated = handle.read()
        generated = re.sub(r"^@import InferKit;$", "", generated, flags=re.M)
        with open(os.path.join(destination, "InferKitMLX-Swift.h"), "w", encoding="utf-8") as handle:
            handle.write(generated)


def generated_text(work):
    with open(os.path.join(work, "InferKitMLX-Swift.h"), encoding="utf-8") as handle:
        return handle.read()


def command_headers(arguments):
    prepare_headers(os.path.abspath(arguments.headers), arguments.objc_header, arguments.output)


def command_surface(arguments):
    header_root = os.path.abspath(arguments.headers)
    work = tempfile.mkdtemp(prefix="inferkit-abi-")
    try:
        prepare_headers(header_root, arguments.objc_header, work)
        unit = ["#import <Foundation/Foundation.h>", "#import <InferKit/InferKit.h>"]
        if arguments.objc_header:
            unit.append('#import "InferKitMLX-Swift.h"')
        with open(os.path.join(work, "surface.m"), "w") as handle:
            handle.write("\n".join(unit) + "\n")
        command = ["xcrun", "clang", "-x", "objective-c", "-fobjc-arc", "-fmodules",
                   "-fmodules-cache-path=" + os.path.join(work, "module-cache"), "-fsyntax-only",
                   "-target", TARGET, "-I", work, "-Xclang", "-ast-dump=json", "surface.m"]
        result = subprocess.run(command, cwd=work, capture_output=True)
        if result.returncode != 0:
            sys.stderr.write(result.stderr.decode(errors="replace")[-4000:])
            sys.exit("the headers do not compile")
        tree = json.loads(result.stdout)
        classes, protocols, constants, functions, enums, structures = read_surface(tree, work)
        if arguments.objc_header:
            # Clang's JSON names no protocol's runtime name, so it is read from the generated header.
            for runtime_name, name in re.findall(r'SWIFT_PROTOCOL\("([^"]+)"\)\s*@protocol (\w+)', generated_text(work)):
                if name in protocols:
                    protocols[name]["runtimeName"] = runtime_name
    finally:
        shutil.rmtree(work, ignore_errors=True)
    surface = {
        "apiVersion": api_version(header_root),
        "classes": dict(sorted(classes.items())),
        "protocols": dict(sorted(protocols.items())),
        "constants": dict(sorted(constants.items())),
        "functions": dict(sorted(functions.items())),
        "enums": dict(sorted(enums.items())),
        "structures": dict(sorted(structures.items())),
    }
    with open(arguments.output, "w") as handle:
        json.dump(surface, handle, indent=1, sort_keys=False)
        handle.write("\n")
    selectors = sum(len(entry["methods"]) for entry in classes.values())
    print("    %d classes, %d selectors, %d protocols, %d constants, %d functions, %d enumerations, "
          "%d structures (API %s)" % (len(classes), selectors, len(protocols), len(constants),
                                      len(functions), len(enums), len(structures),
                                      surface["apiVersion"] or "not stated"))


# ---- diff -------------------------------------------------------------------------------------------

def compare(baseline, candidate):
    incompatible, additions = [], []

    def keyed(kind, old, new, describe):
        for name in sorted(old):
            if name not in new:
                incompatible.append("%s %s was removed" % (kind, name))
            elif old[name] != new[name]:
                incompatible.append("%s %s changed: %s" % (kind, name, describe(old[name], new[name])))
        for name in sorted(set(new) - set(old)):
            additions.append("%s %s was added" % (kind, name))

    for name in sorted(baseline["classes"]):
        old = baseline["classes"][name]
        new = candidate["classes"].get(name)
        if new is None:
            incompatible.append("class %s was removed" % name)
            continue
        if old["runtimeName"] != new["runtimeName"]:
            incompatible.append("class %s changed its runtime name from %s to %s"
                                % (name, old["runtimeName"], new["runtimeName"]))
        if old["superclass"] != new["superclass"]:
            incompatible.append("class %s changed its superclass from %s to %s"
                                % (name, old["superclass"], new["superclass"]))
        for protocol in old["protocols"]:
            if protocol not in new["protocols"]:
                incompatible.append("class %s no longer adopts %s" % (name, protocol))
        for protocol in new["protocols"]:
            if protocol not in old["protocols"]:
                additions.append("class %s adopts %s" % (name, protocol))
        keyed("method %s" % name, old["methods"], new["methods"], lambda a, b: "%s -> %s" % (a, b))
    for name in sorted(set(candidate["classes"]) - set(baseline["classes"])):
        additions.append("class %s was added" % name)

    for name in sorted(baseline["protocols"]):
        old = baseline["protocols"][name]
        new = candidate["protocols"].get(name)
        if new is None:
            incompatible.append("protocol %s was removed" % name)
            continue
        if old.get("runtimeName", name) != new.get("runtimeName", name):
            incompatible.append("protocol %s changed its runtime name from %s to %s"
                                % (name, old.get("runtimeName", name), new.get("runtimeName", name)))
        for section in ("required", "optional"):
            for selector, signature in sorted(old[section].items()):
                other = "optional" if section == "required" else "required"
                if selector in new[section]:
                    if new[section][selector] != signature:
                        incompatible.append("protocol %s %s changed: %s -> %s"
                                            % (name, selector, signature, new[section][selector]))
                elif selector in new[other]:
                    incompatible.append("protocol %s %s became %s" % (name, selector, other))
                else:
                    incompatible.append("protocol %s %s was removed" % (name, selector))
        for selector in sorted(set(new["required"]) - set(old["required"]) - set(old["optional"])):
            incompatible.append("protocol %s gained the required method %s" % (name, selector))
        for selector in sorted(set(new["optional"]) - set(old["optional"]) - set(old["required"])):
            additions.append("protocol %s gained the optional method %s" % (name, selector))
        for protocol in old["protocols"]:
            if protocol not in new["protocols"]:
                incompatible.append("protocol %s no longer inherits %s" % (name, protocol))
        for protocol in new["protocols"]:
            if protocol not in old["protocols"]:
                incompatible.append("protocol %s now inherits %s" % (name, protocol))
    for name in sorted(set(candidate["protocols"]) - set(baseline["protocols"])):
        additions.append("protocol %s was added" % name)

    keyed("constant", baseline["constants"], candidate["constants"], lambda a, b: "%s -> %s" % (a, b))
    keyed("function", baseline["functions"], candidate["functions"], lambda a, b: "%s -> %s" % (a, b))
    keyed("structure", baseline["structures"], candidate["structures"], lambda a, b: "layout changed")

    for name in sorted(baseline["enums"]):
        old = baseline["enums"][name]
        new = candidate["enums"].get(name)
        if new is None:
            incompatible.append("enumeration %s was removed" % name)
            continue
        if old["type"] != new["type"]:
            incompatible.append("enumeration %s changed its type from %s to %s" % (name, old["type"], new["type"]))
        for case, value in sorted(old["values"].items()):
            if case not in new["values"]:
                incompatible.append("enumeration case %s was removed" % case)
            elif new["values"][case] != value:
                incompatible.append("enumeration case %s changed from %d to %d" % (case, value, new["values"][case]))
        for case in sorted(set(new["values"]) - set(old["values"])):
            additions.append("enumeration case %s was added" % case)
    for name in sorted(set(candidate["enums"]) - set(baseline["enums"])):
        additions.append("enumeration %s was added" % name)
    return incompatible, additions


def mangled_runtime_names(surface):
    """Swift gives a class or protocol with no `@objc(Name)` a mangled runtime name such as
    `_TtC11InferKitMLX12NFKMLXKokoro`, which NSClassFromString does not find by its plain name and
    which changes with the Swift module or nesting."""
    found = []
    for kind, table in (("class", surface["classes"]), ("protocol", surface["protocols"])):
        for name, entry in sorted(table.items()):
            runtime_name = entry.get("runtimeName", name)
            if runtime_name.startswith("_Tt"):
                found.append("%s %s has the Swift runtime name %s; give it @objc(%s)"
                             % (kind, name, runtime_name, name))
    return found


def parse_version(text):
    if not text:
        return None
    major, minor = text.split(".")
    return int(major), int(minor)


def command_diff(arguments):
    with open(arguments.baseline) as handle:
        baseline = json.load(handle)
    with open(arguments.candidate) as handle:
        candidate = json.load(handle)
    incompatible, additions = compare(baseline, candidate)
    for line in incompatible:
        print("    incompatible: " + line)
    shown = additions if arguments.verbose else additions[:20]
    for line in shown:
        print("    addition: " + line)
    if len(additions) > len(shown):
        print("    ... and %d more additions (--verbose lists them)" % (len(additions) - len(shown)))
    print("    %d incompatible changes, %d additions" % (len(incompatible), len(additions)))

    mangled = mangled_runtime_names(candidate)
    for line in mangled:
        print("    FAILED: " + line)

    old, new = parse_version(baseline.get("apiVersion")), parse_version(candidate.get("apiVersion"))
    problem = None
    if old is None or new is None:
        print("    the API version rule is not checked: a surface states no API version")
    elif incompatible and new[0] <= old[0]:
        problem = "incompatible changes need API %d.0; the candidate states %d.%d" % (old[0] + 1, new[0], new[1])
    elif additions and not incompatible and new <= old:
        problem = "additions need API %d.%d; the candidate states %d.%d" % (old[0], old[1] + 1, new[0], new[1])
    elif new < old:
        problem = "the API version went backward, from %d.%d to %d.%d" % (old + new)
    if problem and arguments.allow_unbumped:
        print("    note: " + problem + " (allowed: the baseline is no tagged release)")
    elif problem:
        print("    FAILED: " + problem)
        return 1
    return 1 if mangled else 0


# ---- client -----------------------------------------------------------------------------------------

CLIENT = r'''// Generated by Tools/abi/abi-surface.py. Compiled against the baseline headers; checks a framework
// against the baseline surface, which it reads at run time.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#import <InferKit/InferKit.h>
%(objc_import)s

// Every protocol the baseline names. Referring to each one here compiles it against the baseline
// headers and registers it with the runtime before the framework loads.
static NSDictionary<NSString *, Protocol *> *BaselineProtocols(void)
{
	return @{
%(protocols)s
	};
}

static unsigned long failures = 0;

static void Report(NSString *message)
{
	failures += 1;
	printf("    missing: %%s\n", message.UTF8String);
}

static BOOL Conforms(Class cls, Protocol *protocol)
{
	for (Class current = cls; current != Nil; current = class_getSuperclass(current)) {
		if (class_conformsToProtocol(current, protocol)) {
			return YES;
		}
	}
	return NO;
}

static id ConstantValue(void *address, NSString *type)
{
	if ([type containsString:@"NSString"]) {
		return (__bridge id)*(void **)address;
	}
	if ([type containsString:@"double"]) {
		return @(*(const double *)address);
	}
	if ([type containsString:@"float"]) {
		return @(*(const float *)address);
	}
	if ([type containsString:@"long long"] || [type containsString:@"int64_t"]) {
		return @(*(const long long *)address);
	}
	if ([type containsString:@"short"] || [type containsString:@"int16_t"]) {
		return @(*(const unsigned short *)address);
	}
	return nil;
}

int main(int argc, const char *argv[])
{
	@autoreleasepool {
		if (argc < 3) {
			printf("usage: client <framework binary> <baseline surface.json> [constant values out]\n");
			return 2;
		}
		void *image = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
		if (image == NULL) {
			printf("    cannot load %%s: %%s\n", argv[1], dlerror());
			return 2;
		}
		NSData *data = [NSData dataWithContentsOfFile:@(argv[2])];
		NSDictionary *surface = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
		if (surface == nil) {
			printf("    cannot read %%s\n", argv[2]);
			return 2;
		}
		NSDictionary<NSString *, Protocol *> *protocols = BaselineProtocols();
		unsigned long classCount = 0, selectorCount = 0, conformanceCount = 0;

		NSDictionary<NSString *, NSDictionary *> *classes = surface[@"classes"];
		for (NSString *name in classes) {
			NSDictionary *entry = classes[name];
			NSString *runtimeName = entry[@"runtimeName"];
			Class cls = objc_getClass(runtimeName.UTF8String);
			classCount += 1;
			if (cls == Nil) {
				Report([NSString stringWithFormat:@"class %%@ (runtime name %%@)", name, runtimeName]);
				continue;
			}
			NSString *superclass = entry[@"superclass"];
			Class actualSuperclass = class_getSuperclass(cls);
			if ([superclass isKindOfClass:NSString.class] && actualSuperclass != Nil
				&& ![@(class_getName(actualSuperclass)) isEqualToString:superclass]) {
				Report([NSString stringWithFormat:@"class %%@ has superclass %%s, not %%@", name,
						class_getName(actualSuperclass), superclass]);
			}
			for (NSString *protocolName in entry[@"protocols"]) {
				conformanceCount += 1;
				Protocol *protocol = protocols[protocolName];
				if (protocol == nil || !Conforms(cls, protocol)) {
					Report([NSString stringWithFormat:@"class %%@ conforming to %%@", name, protocolName]);
				}
			}
			for (NSString *key in entry[@"methods"]) {
				selectorCount += 1;
				SEL selector = sel_registerName([key substringFromIndex:1].UTF8String);
				BOOL instance = [key hasPrefix:@"-"];
				// class_respondsToSelector consults the resolver without running +initialize.
				Class target = instance ? cls : object_getClass(cls);
				if (!class_respondsToSelector(target, selector)) {
					Report([NSString stringWithFormat:@"%%@[%%@ %%@]", instance ? @"-" : @"+", name,
							[key substringFromIndex:1]]);
				}
			}
		}

		NSMutableDictionary<NSString *, id> *values = [NSMutableDictionary dictionary];
		NSDictionary<NSString *, NSString *> *constants = surface[@"constants"];
		for (NSString *name in constants) {
			void *address = dlsym(image, name.UTF8String);
			if (address == NULL) {
				Report([NSString stringWithFormat:@"constant %%@", name]);
				continue;
			}
			id value = ConstantValue(address, constants[name]);
			if (value != nil) {
				values[name] = value;
			}
		}
		for (NSString *name in surface[@"functions"]) {
			if (dlsym(image, name.UTF8String) == NULL) {
				Report([NSString stringWithFormat:@"function %%@", name]);
			}
		}
		if (argc > 3) {
			NSData *written = [NSJSONSerialization dataWithJSONObject:values
															  options:NSJSONWritingSortedKeys | NSJSONWritingPrettyPrinted
																error:NULL];
			[written writeToFile:@(argv[3]) atomically:YES];
		}
		printf("    checked %%lu classes, %%lu selectors, %%lu conformances, %%lu constants, %%lu functions: %%lu missing\n",
			   classCount, selectorCount, conformanceCount, (unsigned long)constants.count,
			   (unsigned long)[surface[@"functions"] count], failures);
	}
	return failures == 0 ? 0 : 1;
}
'''


def command_client(arguments):
    with open(arguments.surface) as handle:
        surface = json.load(handle)
    names = set(surface["protocols"])
    for entry in surface["classes"].values():
        names.update(entry["protocols"])
    for entry in surface["protocols"].values():
        names.update(entry["protocols"])
    lines = ['\t\t@"%s": @protocol(%s),' % (name, name) for name in sorted(names)]
    source = CLIENT % {
        "objc_import": '#import "InferKitMLX-Swift.h"' if arguments.objc_header else "",
        "protocols": "\n".join(lines),
    }
    with open(arguments.output, "w") as handle:
        handle.write(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    headers = commands.add_parser("headers")
    headers.add_argument("--headers", required=True)
    headers.add_argument("--objc-header")
    headers.add_argument("--output", required=True)
    surface = commands.add_parser("surface")
    surface.add_argument("--headers", required=True)
    surface.add_argument("--objc-header")
    surface.add_argument("--output", required=True)
    diff = commands.add_parser("diff")
    diff.add_argument("baseline")
    diff.add_argument("candidate")
    diff.add_argument("--allow-unbumped", action="store_true")
    diff.add_argument("--verbose", action="store_true")
    client = commands.add_parser("client")
    client.add_argument("surface")
    client.add_argument("--objc-header", action="store_true")
    client.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.command == "headers":
        command_headers(arguments)
        return 0
    if arguments.command == "surface":
        command_surface(arguments)
        return 0
    if arguments.command == "diff":
        return command_diff(arguments)
    command_client(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
