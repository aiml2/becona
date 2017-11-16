import os
import argparse

def prefix_groups(data):
    """Return a dictionary of {prefix:[items]}."""
    lines = data[:]
    groups = dict()
    while lines:
        longest = None
        first = lines.pop()
        for line in lines:
            prefix = os.path.commonprefix([first, line])
            if not longest:
                longest = prefix
            elif len(prefix) > len(longest):
                longest = prefix
        if longest:
            group = [first]
            rest = [item for item in lines if longest in item]
            [lines.remove(item) for item in rest]
            group.extend(rest)
            groups[longest] = group
        else:
            # Singletons raise an exception
            raise IndexError("No prefix match for {}!".format(first))
    return groups


parser = argparse.ArgumentParser()
parser.add_argument('--inputdir',default="input", type=str)
args = parser.parse_args()
print(args)

files = sorted(os.listdir(args.inputdir))
pfg = prefix_groups(files)
#print(pfg)
for el in pfg:
    print(el)
    print(pfg[el])
