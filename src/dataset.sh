#!/bin/bash
# if [ -d "Audio_Classification_using_LSTM" ]; then
#   echo "Dataset already exists"
# else
#   echo "Cloning dataset..."
#   git clone https://github.com/carldegs/Audio_Classification_using_LSTM
# fi

# cd Audio_Classification_using_LSTM
# git pull

if [ ! -f "dataset.tar.gz" -a ! -d "UrbanSound8k" ]; then
  wget "https://goo.gl/8hY5ER" -O "dataset.tar.gz"
  tar -tf dataset.tar.gz
fi
