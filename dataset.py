import os.path
import tarfile
import wget

def setup_dataset(download = False, extract = False):
  if download:
    wget.download("https://goo.gl/8hY5ER", out="dataset.tar.gz")
    print("file downloaded")

  if extract:
    tar = tarfile.open('dataset.tar.gz', 'r:gz')
    tar.extractall()
    tar.close()
    print("dataset extracted")