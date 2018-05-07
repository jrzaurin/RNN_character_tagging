import fnmatch
import os
import argparse

from text_utils import sanitize_text


def main(input_dir, file_pattern, output_dir):
    try:
        os.makedirs(output_dir)
    except os.error as e:
        # errno 17 means 'file exists error' which we can ignore
        if e.errno != 17:
            raise

    for root, dirnames, filenames in os.walk(input_dir):
        for filename in fnmatch.filter(filenames, file_pattern):
            src_path = os.path.join(root, filename)
            dst_path = os.path.join(output_dir, filename)
            # read in bytes (rb), write in text ('w')
            with open(src_path, 'rb') as in_f, open(dst_path, 'w') as out_f:
                out_f.write(sanitize_text(in_f.read()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Takes all files from a given directory and subdirectories "
        "matching a pattern, cleans up their contents and puts in a "
        "new directory.")
    parser.add_argument("input_dir",
        help="directory containing input files")
    parser.add_argument("file_pattern",
        help="for example *.py for python files")
    parser.add_argument("output_dir",
        help="where to put cleaned files")
    args = parser.parse_args()

    main(args.input_dir, args.file_pattern, args.output_dir)