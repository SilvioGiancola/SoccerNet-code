import zipfile
import os
import argparse


def UnpackData(path_to_zip_file, directory_to_extract_to):
	print("Unpacking", path_to_zip_file, "..." )
	zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
	zip_ref.extractall(directory_to_extract_to)
	zip_ref.close()
	print("...", path_to_zip_file, "unpacked in", directory_to_extract_to, "folder" )


if __name__ == "__main__":

	description = 'Unzip the data'
	p = argparse.ArgumentParser(description=description)
	p.add_argument('--input_dir', type=str, default='.', required=False,
		help='Folder containing the zipped files.')
	p.add_argument('--output_dir', type=str, default='.', required=False,
		help='Folder where zipped files will be extracted.')

	args = p.parse_args()


	UnpackData(os.path.join(args.input_dir, "All_Labels.zip"), args.output_dir)

	UnpackData(os.path.join(args.input_dir, "All_Features.zip"), args.output_dir)

