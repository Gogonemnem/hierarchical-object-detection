from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

import gdown
from download_dataset import parse_args


def download_gdrive(url_dict, dir, unzip=True, delete=False, threads=1):
    """ Custom download function for Google Drive URLs. """

    def get_extract_dir(filename, base_dir):
        """ Determine the extraction directory based on the filename. """
        # Define extraction rules for each type of file
        extract_rules = {
            'ImageSets.zip': base_dir / 'ImageSets'
        }
        # Return the corresponding directory or the base directory
        return extract_rules.get(filename, base_dir)

    def download_single(url, filename, dir):
        """ Nested function to download a single file from Google Drive with the correct filename. """
        file_id = url.split('id=')[1]
        output = dir / filename
        
        # Use gdown to download from Google Drive
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(output), quiet=False)
        print(f"Downloaded {output}")

        # Unzip if needed
        if unzip and output.suffix in ('.zip', '.tar'):
            print(f'Unzipping {output.name}')
            extract_to = get_extract_dir(filename, dir)
            extract_to.mkdir(parents=True, exist_ok=True)

            if output.suffix == '.zip':
                with ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(path=extract_to)
            elif output.suffix == '.tar':
                with TarFile.open(output) as tar_ref:
                    tar_ref.extractall(path=extract_to)

            if delete:
                output.unlink()
                print(f'Deleted {output}')

    # Create directory if it doesn't exist
    dir = Path(dir)
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

    # Multi-threading support
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_single(*x, dir), url_dict.items())
        pool.close()
        pool.join()
    else:
        for url, filename in url_dict.items():
            download_single(url, filename, dir)


def main():
    args = parse_args()
    path = Path(args.save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data2url = dict(
        dior={
        'https://drive.google.com/open?id=1ZHbHDM6hYAEGDC_K5eiW0yF_lzVgpuir': 'JPEGImages-trainval.zip',
        'https://drive.google.com/open?id=11SXPqcESez9qTn4Z5Q3v35K9hRwO_epr': 'JPEGImages-test.zip',
        'https://drive.google.com/open?id=1vOmzwxpBtwbK5o8xSa9u0IdB4H95MBHw': 'ImageSets.zip',
        'https://drive.google.com/open?id=1KoQzqR20qvIXDf1qsXCHGxD003IPmXMw': 'Annotations.zip'
        }
    )

    url = data2url.get(args.dataset_name, None)

    if args.dataset_name == 'dior':
        download_gdrive(
            url,
            dir=path,
            unzip=args.unzip,
            delete=args.delete,
            threads=args.threads
        )

    if url is None:
        print('Only support DIOR for now!')
        return


if __name__ == '__main__':
    main()
