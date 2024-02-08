import sys
help_str = """
This file is devoted to reading, and possibly updating, the version of the agd library.
"""

def CheckVersion(filename,version_line,version_start,version_end,new_version=None):
	with open(filename,'r',encoding='utf8') as file:
		content = file.readlines()

	line = content[version_line]
	assert line.startswith(version_start)
	assert line.endswith(version_end)
	current_version = line[len(version_start):-len(version_end)]
	if new_version is None: 
		print(f"Current version in file {filename} : {current_version}")
		return current_version
	else: 
		content[version_line] = f"{version_start}{new_version}{version_end}"
		with open(filename,'w',encoding='utf8') as file:
			file.write(''.join(content))
		return new_version

def Main(new_version=None):
	assert new_version is None or all(x in "0123456789." for x in new_version)
	v1 = CheckVersion("conda.recipe/setup.py",7,"	version='","',\n",new_version)
	v2 = CheckVersion("conda.recipe/meta.yaml",2,"  version: ","\n",new_version)
	assert v1==v2
	return v1

if __name__ == "__main__":
	if "--help" in sys.argv[1:]:
		print(help_str)
		exit(0)

	new_version = None
	for key in sys.argv[1:]:
		if key.startswith("--new_version="):
			new_version = key[len("--new_version="):]
		